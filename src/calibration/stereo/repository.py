from pathlib import Path
from typing import Optional

import numpy as np

from .config import CalibrationResult
from .sub_modules.calib_store import StereoCalibrationStore

__all__ = ["StereoRigRepository"]

class StereoRigRepository:
    """Persistence boundary for stereo calibration artifacts.

    Takes the stereomap path as `str` or `Path` and hands it on as `str` to
    `StereoCalibrationStore`, which owns the file format and its schema version.
    """

    def __init__(self, store: StereoCalibrationStore | None = None):
        self.store = store if store is not None else StereoCalibrationStore()
    
    def load(self, stereomap_file: str | Path) -> tuple[CalibrationResult, Optional[np.ndarray]]:
        """Reads a stereomap: the calibration, and the CAMERA -> BASE matrix if it carries one."""
        return self.store.load(str(stereomap_file))
    
    def save(
        self,
        stereomap_file: str | Path,
        result: CalibrationResult,
        extrinsics: Optional[np.ndarray] = None,
    ) -> None:
        """Writes the calibration, and the extrinsics when given.

        The file is rewritten whole, so a save without extrinsics drops the
        CAMERA -> BASE matrix a previous save put there.
        """
        self.store.save(str(stereomap_file), result, extrinsics)