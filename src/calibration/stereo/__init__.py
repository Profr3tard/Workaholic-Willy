"""Stereo calibration, rectification, reconstruction and marker pose.

`StereoCam3D` is the entry point: it builds one `StereoRigRunTime` per configured
rig through `StereoRigFactory` and addresses them by index.
"""

from __future__ import annotations

from .config import CalibrationResult, StereoRigConfig
from .factory import StereoRigFactory
from .manager import StereoCam3D
from .repository import StereoRigRepository
from .runtime import StereoRigRunTime

__all__ = [
	"CalibrationResult",
	"StereoCam3D",
	"StereoRigConfig",
	"StereoRigFactory",
	"StereoRigRepository",
	"StereoRigRunTime",
]
