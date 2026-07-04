"""Backwards-compatible facade over the stereo calibrator + map store.

External code (``StereoRigRepository``, ``StereoRigFactory``) still imports
:class:`StereoCalibration` from this module. Internally the heavy lifting
is delegated to the focused helpers:

* :class:`StereoCalibrator` — chessboard detection + ``cv.stereoCalibrate``.
* :class:`StereoMapStore` — XML I/O for the resulting maps.

New code should depend on those classes directly.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from backend.src.calibration.stereo.config import CalibrationResult

from .calibrator import StereoCalibrator
from .map_store import StereoMapStore


class StereoCalibration:
    """Thin facade preserving the historical API surface."""

    def __init__(
        self,
        chessboard_size: Sequence[int],
        square_size_mm: float,
        rectify_alpha: float = 0.0,
    ) -> None:
        self._calibrator = StereoCalibrator(
            chessboard_size=chessboard_size,
            square_size_mm=square_size_mm,
            rectify_alpha=rectify_alpha,
        )
        self._store = StereoMapStore()
        self.result: Optional[CalibrationResult] = None

    # Read-only mirrors of the underlying calibrator's parameters so that
    # historical attribute access keeps working.
    @property
    def chessboard_size(self) -> Tuple[int, int]:
        return self._calibrator.chessboard_size

    @property
    def square_size_mm(self) -> float:
        return self._calibrator.square_size_mm

    @property
    def rectify_alpha(self) -> float:
        return self._calibrator.rectify_alpha

    # ------------------------------------------------------------------
    def calibrate(
        self,
        frame_size: Optional[Tuple[int, int]],
        left_glob: str,
        right_glob: str,
    ) -> CalibrationResult:
        self.result = self._calibrator.calibrate(frame_size, left_glob, right_glob)
        return self.result

    def save(self, filepath: str, T_cam_to_base: Optional[np.ndarray] = None) -> None:
        if self.result is None:
            raise RuntimeError("There is no calibration available to save.")
        self._store.save(filepath, self.result, T_cam_to_base)

    def load(
        self, filepath: str, test_output: bool = False
    ) -> Tuple[CalibrationResult, Optional[np.ndarray]]:
        result, T = self._store.load(filepath)
        self.result = result
        if test_output:
            self._store.print_diagnostics(result)
        return result, T


__all__ = ["CalibrationResult", "StereoCalibration"]
