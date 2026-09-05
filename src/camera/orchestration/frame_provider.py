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
    """Frame provider keyed by rig id, over stereo, single-device and RGB-D rigs.

    It owns lifecycle and rig indexing only. Acquisition, splitting, cropping,
    resizing and quality setup stay in the streamers under ``camera.setup``, and
    :meth:`rig` hands one rig to a consumer as a `RigHandle`.
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
        #: Which rigs stream right now. Per rig rather than one flag, so a consumer handed a
        #: single rig gives that one back without closing devices it never held. See
        #: `RigHandle.release`.
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

        Read from the per-rig set, so a provider whose one rig was handed back reports false
        and the next ``grab`` on it fails instead of reaching a released device.
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

        The counterpart to `release_rig`, for a consumer that needs a single rig where
        `open()` claims every configured one. No ``__init__`` under
        `camera.setup.image_taking` touches a device, it only stores config, so a provider
        knows every rig while holding open only the ones it was asked for.
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

        Takes a rig id rather than tearing down the provider, so a consumer gives back the rig
        it was handed and cannot close one it was not. The console's only teardown path ends
        here, through a built service: ``api/lifecycle.release_perception`` ->
        ``perception.close()``.

        Teardown must not be stopped by a camera that will not close, so a failure is logged
        rather than raised. The log is where the reason for the next failed open shows up.
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
        # Per rig, not for the provider as a whole: a rig that was handed back fails here
        # rather than reaching a released device, and the rest stay reachable.
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
        """The camera matrix ``rig_id`` reports, or ``None`` if its streamer has none.

        The rig must be open. K is what gives a frame a metric meaning, and both consumers of
        a real camera need it: the perception adapter to unproject, and `GraspCalculator` at
        construction. Serving it here keeps them inside the rig-keyed lifecycle instead of
        opening a `RealSenseRGBDStreamer` of their own. A stereo rig answers ``None``, its
        geometry living in `StereoCam3D` rather than in a single pinhole matrix.
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

        The rig must be open. Hand-eye calibration needs the coefficients, which is why this
        sits beside `get_intrinsics`. `calibration.rgbd_marker_source.RGBDArucoMarkerSource`,
        which lets a fixed RGB-D camera see the ArUco board, duck-types against ``grab``,
        ``get_intrinsics`` and ``get_distortion``; a rig serving only the first two is
        addressable but cannot be calibrated.
        """
        streamer = self._require_streamer(rig_id)
        self._require_open(rig_id)
        read = getattr(streamer, "get_distortion", None)
        return read() if callable(read) else None

    def rig(self, rig_id: str) -> "RigHandle":
        """A :class:`RigHandle` to one rig, shaped like the streamer its consumer expects.

        The handle answers ``grab()``, ``get_intrinsics()`` and ``release()``, the surface
        `RealSenseVisionPerceptionSource` duck-types against, and reaches this rig and no other.
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
    """One rig of a :class:`FrameProvider`, shaped like the streamer its consumer expects.

    The robot pick path consumes a duck-typed streamer, ``grab()``, ``get_intrinsics()`` and
    ``release()``, and so does `datagen`'s camera probe, which passes a shim and runs with no
    hardware present. A handle serves that surface, so neither learns about camera
    orchestration while every frame goes through the class that owns rig identity and lifecycle.

    A handle reaches one rig. ``release()`` gives that rig back and leaves the others streaming.
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
        """This rig's distortion coefficients, or ``None`` where the rig reports none.

        Completes the surface hand-eye calibration duck-types against, so a handle can be
        calibrated and not only read.
        """
        return self._provider.get_distortion(self.rig_id)

    def release(self) -> None:
        """Give this rig back. Idempotent, never raises, and it touches no other rig."""
        self._provider.release_rig(self.rig_id)

    @property
    def is_open(self) -> bool:
        return self.rig_id in self._provider.open_rig_ids()
