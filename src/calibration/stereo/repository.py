"""Stereo rig repository for loading and saving calibration artifacts."""

from pathlib import Path
from typing import Optional

import numpy as np

from .config import CalibrationResult
from .sub_modules.calib_store import StereoCalibrationStore

__all__ = ["StereoRigRepository"]

class StereoRigRepository:
    """Explicit persistence boundary for stereo calibration artifacts."""

    def __init__(self, store: StereoCalibrationStore | None = None):
        self.store = store if store is not None else StereoCalibrationStore()
    
    def load(self, stereomap_file: str | Path) -> tuple[CalibrationResult, Optional[np.ndarray]]:
        return self.store.load(str(stereomap_file))
    
    def save(
        self,
        stereomap_file: str | Path,
        result: CalibrationResult,
        extrinsics: Optional[np.ndarray] = None,
    ) -> None:
        self.store.save(str(stereomap_file), result, extrinsics)