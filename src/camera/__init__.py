"""Camera acquisition, setup, and runtime orchestration."""

from __future__ import annotations

from .orchestration import FrameProvider, FrameProviderStateError, UnknownCameraRigError
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