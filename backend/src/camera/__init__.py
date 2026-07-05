"""Camera acquisition, setup, and runtime orchestration entry points."""

from __future__ import annotations

from .orchestration import (
    FrameProviderStateError,
    UnknownCameraRigError,
)

from .frame_provider import FrameProvider

from .pipeline import StereoCapturePipeline

from .setup.image_taking import AnyFrame, RGBDFrame, StereoFrame


__all__ = [
    "AnyFrame",
    "FrameProvider",
    "FrameProviderStateError",
    "RGBDFrame",
    "StereoCapturePipeline",
    "StereoFrame",
    "UnknownCameraRigError",
]