from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:                                    # pragma: no cover (typing only)
    import numpy as np

from src.config.schema.camera import (
    RGBDDeviceRigConfig,
    SingleDeviceRigConfig,
    WebcamPairRigConfig,
)
from src.calibration.stereo.manager import StereoCam3D
from src.camera.setup.image_taking.frames import AnyFrame, StereoFrame
from src.camera.setup.image_taking.rgbd import (
    OpenCvRGBDStreamer,
    RealSenseRGBDStreamer,
)
from src.camera.setup.image_taking.single import SingleDeviceStreamer
from src.camera.setup.image_taking.webcam import WebcamPairStreamer

CameraRigConfig = WebcamPairRigConfig | SingleDeviceRigConfig | RGBDDeviceRigConfig
AnyStreamer = (
    WebcamPairStreamer | SingleDeviceStreamer | OpenCvRGBDStreamer | RealSenseRGBDStreamer
)

__all__ = [
    "FrameProvider",
    "FrameProviderStateError",
    "RigHandle",
    "UnknownCameraRigError",
]


class UnknownCameraRigError(KeyError):
    """Raised when a requested camera rig ID is not registered."""


class FrameProviderStateError(RuntimeError):
    """Raised when frames are requested before the provider is opened."""


class FrameProvider:
    """Unified rig-ID keyed frame provider.

    The provider owns lifecycle and rig indexing only. All image acquisition,
    splitting, cropping, resizing, and quality setup stays in the lower-level
    streamers under ``camera.setup``.
    """

    def __init__(
        self,
        rigs: list[CameraRigConfig],
        stereo: StereoCam3D | None = None,
    ) -> None:
        if not rigs:
            raise ValueError("At least one rig must be provided.")

        self.logger = logging.getLogger(__name__)
        self._rigs: dict[str, CameraRigConfig] = {}
        self._streamers: dict[str, AnyStreamer] = {}
        self._stereo = stereo
        self._stereo_rig_index: dict[str, int] = {}
        #: Which rigs are currently streaming. PER RIG rather than one flag, because a consumer that
        #: was handed a single rig must be able to give that one back without closing devices it was
        #: never given; see `RigHandle.release`.
        self._open_rigs: set[str] = set()

        stereo_idx = 0
        for rig in rigs:
            rig_id = rig.rig_id
            if rig_id in self._rigs:
                raise ValueError(f"Duplicate rig_id: {rig_id}")

            self._rigs[rig_id] = rig
            self._streamers[rig_id] = self._create_streamer(rig)

            if not isinstance(rig, RGBDDeviceRigConfig):
                self._stereo_rig_index[rig_id] = stereo_idx
                stereo_idx += 1

        self.logger.info("FrameProvider created with rigs: %s", list(self._rigs))

    @property
    def is_open(self) -> bool:
        """True when every rig is streaming.

        Derived from the per-rig set rather than a single flag, so a provider whose one rig has been
        handed back is correctly no longer "open", where a flag would have kept saying yes and let the next
        ``grab`` reach a released device.
        """
        return bool(self._rigs) and self._open_rigs == set(self._rigs)

    def open(self) -> None:
        """Open every streamer, rolling back already-opened ones on failure."""
        if self.is_open:
            return
        opened: list[str] = []
        try:
            for rig_id, streamer in self._streamers.items():
                if rig_id in self._open_rigs:
                    continue
                streamer.open()
                opened.append(rig_id)
                self._open_rigs.add(rig_id)
        except Exception:
            self.logger.exception("Failed to open rig; rolling back opened streamers")
            for rig_id in opened:
                try:
                    self._streamers[rig_id].release()
                except Exception as exc:  # pragma: no cover (defensive logging)
                    self.logger.warning("Rollback release of %s failed: %s", rig_id, exc)
                self._open_rigs.discard(rig_id)
            raise

    def open_rig(self, rig_id: str) -> None:
        """Open one rig and leave the others untouched. Idempotent.

        The counterpart to `release_rig`, and what lets a consumer route through this class without
        acquiring devices it does not use. A cell needs one RGB-D camera; `open()` would have claimed
        every configured rig, and on this project an opened camera that nobody closes is a defect with
        a history. Constructing a streamer touches no device, since every ``__init__`` under
        `camera.setup.image_taking` only stores config, so a provider may know every rig while
        holding only the ones it was asked for.
        """
        streamer = self._require_streamer(rig_id)
        if rig_id in self._open_rigs:
            return
        streamer.open()
        self._open_rigs.add(rig_id)

    def release(self) -> None:
        """Release every streamer. Safe to call repeatedly."""
        for rig_id in list(self._open_rigs):
            self.release_rig(rig_id)

    def release_rig(self, rig_id: str) -> None:
        """Release one rig and leave every other one streaming. Idempotent, and it never raises.

        This is what makes a shared provider safe to hand out. A consumer given one rig must be able
        to give that one back, and the console's only teardown path reaches through a built service to
        ``perception.close()`` (``api/lifecycle.release_perception``), so a handle that could not
        release would have turned the one call that closes a camera into a silent no-op, and a cell
        that is rebuilt would never get its device back. It must equally not be able to close a device
        it was never handed, which is why this takes an id rather than tearing down the provider.

        Never raises, because this is teardown: a camera that cannot be closed must not stop the thing
        that was closing it. It is logged, because a device that would not close is the reason the next
        open fails.
        """
        self._require_rig(rig_id)
        if rig_id not in self._open_rigs:
            return
        try:
            self._streamers[rig_id].release()
        except Exception as exc:  # noqa: BLE001 (teardown reports, it does not propagate)
            self.logger.warning("Release of rig %s failed: %s", rig_id, exc)
        self._open_rigs.discard(rig_id)

    def __enter__(self) -> FrameProvider:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        self.release()
        return False

    def grab(self, rig_id: str) -> AnyFrame:
        """Grab a raw frame from ``rig_id`` (a StereoFrame or RGBDFrame per the rig kind)."""
        streamer = self._require_streamer(rig_id)
        # PER RIG, not for the provider as a whole: one rig handed back must fail loudly here rather
        # than reach a released device, and it must not make the others unreachable.
        self._require_open(rig_id)
        return streamer.grab()

    def grab_rectified(self, rig_id: str) -> StereoFrame:
        """Grab a stereo frame and rectify it with ``StereoCam3D``."""
        self._require_rig(rig_id)
        if rig_id not in self._stereo_rig_index:
            raise ValueError(f"Rig {rig_id!r} is RGB-D and does not support stereo rectification")
        if self._stereo is None:
            raise RuntimeError("StereoCam3D is required for grab_rectified()")

        frame = self.grab(rig_id)
        if not isinstance(frame, StereoFrame):
            raise TypeError(f"Rig {rig_id!r} returned {type(frame).__name__}, expected StereoFrame")
        index = self._stereo_rig_index[rig_id]
        left_rectified, right_rectified = self._stereo.rectify(frame.left, frame.right, rig=index)
        return StereoFrame(left=left_rectified, right=right_rectified)

    @property
    def rig_ids(self) -> list[str]:
        """All registered rig identifiers in configuration order."""
        return list(self._rigs)

    def is_rgbd(self, rig_id: str) -> bool:
        """True if ``rig_id`` is an RGB-D device."""
        return isinstance(self._require_rig(rig_id), RGBDDeviceRigConfig)

    def is_stereo(self, rig_id: str) -> bool:
        """True if ``rig_id`` is a stereo rig."""
        self._require_rig(rig_id)
        return not self.is_rgbd(rig_id)

    def get_rig_config(self, rig_id: str) -> CameraRigConfig:
        """Return the original config object for ``rig_id``."""
        return self._require_rig(rig_id)

    def get_stereo_rig_index(self, rig_id: str) -> int:
        """Return the ``StereoCam3D`` rig index for ``rig_id``."""
        self._require_rig(rig_id)
        try:
            return self._stereo_rig_index[rig_id]
        except KeyError as exc:
            raise KeyError(f"Rig {rig_id!r} has no stereo index because it is RGB-D") from exc

    def get_intrinsics(self, rig_id: str) -> "np.ndarray | None":
        """The camera matrix ``rig_id`` reports, or ``None`` if its streamer does not have one.

        Added because its absence is why the pick PATH bypassed this class. Both consumers of a real
        camera need K, the perception adapter to unproject and `GraspCalculator` at construction,
        and the provider could hand out frames but not the matrix that gives them a metric meaning. So
        the robot path built its own `RealSenseRGBDStreamer` and every rig-keyed guarantee here was
        simply not on it. A stereo rig legitimately answers ``None``: its geometry lives in
        `StereoCam3D`, not in a single pinhole matrix.
        """
        streamer = self._require_streamer(rig_id)
        self._require_open(rig_id)
        read = getattr(streamer, "get_intrinsics", None)
        return read() if callable(read) else None

    def open_rig_ids(self) -> frozenset[str]:
        """Which rigs are streaming right now."""
        return frozenset(self._open_rigs)

    def get_distortion(self, rig_id: str) -> "np.ndarray | None":
        """The lens distortion coefficients ``rig_id`` reports, or ``None`` when it has none.

        Needed by hand-eye calibration, which is why it is here beside `get_intrinsics`.
        `calibration.rgbd_marker_source.RGBDArucoMarkerSource`, the piece that lets a fixed RGB-D
        camera see the ArUco board. It duck-types against ``grab`` / ``get_intrinsics`` /
        ``get_distortion``, and a handle missing the third would have made every camera addressable
        and none of them calibratable.
        """
        streamer = self._require_streamer(rig_id)
        self._require_open(rig_id)
        read = getattr(streamer, "get_distortion", None)
        return read() if callable(read) else None

    def rig(self, rig_id: str) -> "RigHandle":
        """A handle to one rig, shaped like the streamer its consumer already expects.

        This is the seam that lets every camera run through this class without any consumer learning
        about it: the handle answers ``grab()``, ``get_intrinsics()`` and ``release()``, which is
        exactly the surface `RealSenseVisionPerceptionSource` duck-types against. It can reach one
        rig and no other.
        """
        self._require_rig(rig_id)
        return RigHandle(self, rig_id)

    def _require_open(self, rig_id: str | None = None) -> None:
        if rig_id is None:
            if not self.is_open:
                raise FrameProviderStateError(
                    "FrameProvider is not open; call open() before grabbing frames")
            return
        if rig_id not in self._open_rigs:
            raise FrameProviderStateError(
                f"Rig {rig_id!r} is not open; call open() before grabbing frames "
                f"(open rigs: {sorted(self._open_rigs) or 'none'})")

    def _require_rig(self, rig_id: str) -> CameraRigConfig:
        try:
            return self._rigs[rig_id]
        except KeyError as exc:
            raise UnknownCameraRigError(f"Unknown rig_id: {rig_id!r}") from exc

    def _require_streamer(self, rig_id: str) -> AnyStreamer:
        self._require_rig(rig_id)
        return self._streamers[rig_id]

    @staticmethod
    def _create_streamer(rig: CameraRigConfig) -> AnyStreamer:
        if isinstance(rig, WebcamPairRigConfig):
            return WebcamPairStreamer(rig)
        if isinstance(rig, SingleDeviceRigConfig):
            return SingleDeviceStreamer(rig)
        if isinstance(rig, RGBDDeviceRigConfig):
            if rig.rgbd_backend == "realsense":
                return RealSenseRGBDStreamer(rig)
            return OpenCvRGBDStreamer(rig)
        raise ValueError(f"Unsupported rig type: {type(rig)!r}")


class RigHandle:
    """One rig of a :class:`FrameProvider`, shaped like the streamer its consumer already expects.

    Why a handle and not the provider itself. The robot pick path consumes a duck-typed streamer --
    ``grab()``, ``get_intrinsics()`` and ``release()``, and so does `datagen`'s camera probe, which
    deliberately passes a shim to exercise the adapter with no hardware present. Handing those a
    provider would have made the adapter learn about camera orchestration and broken the shim. Handing
    them a handle changes nothing on their side and still routes every frame through the one class
    that owns rig identity and lifecycle.

    It can reach one rig. ``release()`` gives that rig back and leaves every other one streaming --
    which is what makes a shared provider safe to hand out, and what keeps the console's only teardown
    path (``api/lifecycle.release_perception`` -> ``perception.close()``) doing what it says.
    """

    __slots__ = ("_provider", "rig_id")

    def __init__(self, provider: FrameProvider, rig_id: str) -> None:
        self._provider = provider
        self.rig_id = rig_id

    def __repr__(self) -> str:
        return f"RigHandle({self.rig_id!r})"

    def grab(self) -> AnyFrame:
        """One frame from this rig."""
        return self._provider.grab(self.rig_id)

    def get_intrinsics(self) -> "np.ndarray | None":
        """This rig's camera matrix, or ``None`` where the rig has no single pinhole matrix."""
        return self._provider.get_intrinsics(self.rig_id)

    def get_distortion(self) -> "np.ndarray | None":
        """This rig's distortion coefficients, or ``None``. Completes the surface hand-eye
        calibration duck-types against, so a handle is calibratable and not merely readable."""
        return self._provider.get_distortion(self.rig_id)

    def release(self) -> None:
        """Give this rig back. Idempotent, never raises, and it touches no other rig."""
        self._provider.release_rig(self.rig_id)

    @property
    def is_open(self) -> bool:
        return self.rig_id in self._provider.open_rig_ids()
