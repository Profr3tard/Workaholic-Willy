"""Unified rig-ID keyed frame provider. For stereo rigs, optionally rectifies frames with a StereoCam3D instance."""

from __future__ import annotations

import logging
from typing import Literal

from config.schema.camera import (
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
        self._is_open = False

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
        """True when all streamers were successfully opened."""
        return self._is_open

    def open(self) -> None:
        """Open every streamer, rolling back already-opened ones on failure."""
        if self._is_open:
            return
        opened: list[str] = []
        try:
            for rig_id, streamer in self._streamers.items():
                streamer.open()
                opened.append(rig_id)
        except Exception:
            self.logger.exception("Failed to open rig; rolling back opened streamers")
            for rig_id in opened:
                try:
                    self._streamers[rig_id].release()
                except Exception as exc:  # pragma: no cover - defensive logging
                    self.logger.warning("Rollback release of %s failed: %s", rig_id, exc)
            raise
        self._is_open = True

    def release(self) -> None:
        """Release every streamer. Safe to call repeatedly."""
        if not self._is_open:
            return
        for rig_id, streamer in self._streamers.items():
            try:
                streamer.release()
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.warning("Release of rig %s failed: %s", rig_id, exc)
        self._is_open = False

    def __enter__(self) -> FrameProvider:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        self.release()
        return False

    def grab(self, rig_id: str) -> AnyFrame:
        """Grab a raw frame from ``rig_id`` (a StereoFrame or RGBDFrame per the rig kind)."""
        streamer = self._require_streamer(rig_id)
        self._require_open()
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

    def _require_open(self) -> None:
        if not self._is_open:
            raise FrameProviderStateError("FrameProvider is not open; call open() before grabbing frames")

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