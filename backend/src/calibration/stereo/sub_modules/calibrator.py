"""Chessboard-based stereo calibration.

``StereoCalibrator`` performs the heavy lifting required to turn a folder
of stereo image pairs into a :class:`CalibrationResult` (rectification
maps, projection matrices, the ``Q`` reprojection matrix, etc.). It is
deliberately stateless except for the calibration parameters themselves —
persistence is handled by :class:`StereoMapStore`.
"""

from __future__ import annotations

import glob
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2 as cv
import numpy as np

from backend.src.calibration.exceptions import StereoCalibrationError
from backend.src.calibration.helpers import proj_to_K
from backend.src.calibration.stereo.config import CalibrationResult

# OpenCV cornerSubPix iteration criteria.
_SUBPIX_CRIT = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# stereoCalibrate iteration criteria.
_STEREO_CRIT = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
# cornerSubPix search window half-size.
_SUBPIX_WIN = (11, 11)
# Minimum number of usable image pairs required for a stable calibration.
_MIN_PAIRS = 10


class StereoCalibrator:
    """Compute stereo intrinsics, extrinsics and rectification maps."""

    def __init__(
        self,
        chessboard_size: Sequence[int],
        square_size_mm: float,
        rectify_alpha: float = 0.0,
    ) -> None:
        self.chessboard_size: Tuple[int, int] = (
            int(chessboard_size[0]),
            int(chessboard_size[1]),
        )
        self.square_size_mm: float = float(square_size_mm)
        self.rectify_alpha: float = float(rectify_alpha)
        if self.chessboard_size[0] <= 0 or self.chessboard_size[1] <= 0:
            raise StereoCalibrationError("chessboard_size values must be positive")
        if self.square_size_mm <= 0.0:
            raise StereoCalibrationError("square_size_mm must be > 0")
        if not (0.0 <= self.rectify_alpha <= 1.0):
            raise StereoCalibrationError("rectify_alpha must be between 0 and 1")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def calibrate(
        self,
        frame_size: Optional[Tuple[int, int]],
        left_glob: str,
        right_glob: str,
    ) -> CalibrationResult:
        """Calibrate from globbed image folders. Returns a fresh result."""
        left_imgs = sorted(glob.glob(left_glob))
        right_imgs = sorted(glob.glob(right_glob))

        if not left_imgs or not right_imgs:
            raise StereoCalibrationError(
                "No calibration images found. "
                f"Looked for left={left_glob!r}, right={right_glob!r}."
            )
        if len(left_imgs) != len(right_imgs):
            raise StereoCalibrationError(
                "Number of left/right calibration images differs: "
                f"{len(left_imgs)} vs {len(right_imgs)}."
            )

        objp = self._object_points()
        objpoints: List[np.ndarray] = []
        imgpointsL: List[np.ndarray] = []
        imgpointsR: List[np.ndarray] = []

        for left_path, right_path in zip(left_imgs, right_imgs):
            pair = self._detect_pair(left_path, right_path)
            if pair is None:
                continue
            (gL_shape, cornersL, cornersR) = pair
            if frame_size is None:
                frame_size = (gL_shape[1], gL_shape[0])
            objpoints.append(objp)
            imgpointsL.append(cornersL)
            imgpointsR.append(cornersR)

        if len(objpoints) < _MIN_PAIRS:
            raise StereoCalibrationError(
                f"Too few usable calibration pairs: {len(objpoints)} (need >= {_MIN_PAIRS})."
            )
        assert frame_size is not None  # for the type checker

        return self._calibrate_from_correspondences(
            frame_size, objpoints, imgpointsL, imgpointsR
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _object_points(self) -> np.ndarray:
        """Generate the 3D coordinates of chessboard corners (z=0)."""
        cols, rows = self.chessboard_size
        objp = np.zeros((cols * rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objp *= self.square_size_mm
        return objp

    def _detect_pair(
        self, left_path: str, right_path: str
    ) -> Optional[Tuple[Tuple[int, int], np.ndarray, np.ndarray]]:
        """Detect chessboard corners in a single stereo pair."""
        L = cv.imread(left_path)
        R = cv.imread(right_path)
        if L is None or R is None:
            return None

        gL = cv.cvtColor(L, cv.COLOR_BGR2GRAY)
        gR = cv.cvtColor(R, cv.COLOR_BGR2GRAY)
        cols, rows = self.chessboard_size

        retL, cornersL = cv.findChessboardCorners(gL, (cols, rows))
        retR, cornersR = cv.findChessboardCorners(gR, (cols, rows))
        if not (retL and retR):
            return None

        cornersL = cv.cornerSubPix(gL, cornersL, _SUBPIX_WIN, (-1, -1), _SUBPIX_CRIT)
        cornersR = cv.cornerSubPix(gR, cornersR, _SUBPIX_WIN, (-1, -1), _SUBPIX_CRIT)
        return gL.shape, cornersL, cornersR

    def _calibrate_from_correspondences(
        self,
        frame_size: Tuple[int, int],
        objpoints: List[np.ndarray],
        imgpointsL: List[np.ndarray],
        imgpointsR: List[np.ndarray],
    ) -> CalibrationResult:
        """Run mono → stereo → rectify → maps from already-detected corners."""
        # Mono calibration for each camera independently.
        _, camL, distL, _, _ = cv.calibrateCamera(  # type: ignore[call-overload]
            objpoints, imgpointsL, frame_size, None, None
        )
        _, camR, distR, _, _ = cv.calibrateCamera(  # type: ignore[call-overload]
            objpoints, imgpointsR, frame_size, None, None
        )

        camL_opt, _ = cv.getOptimalNewCameraMatrix(camL, distL, frame_size, 1.0)
        camR_opt, _ = cv.getOptimalNewCameraMatrix(camR, distR, frame_size, 1.0)

        _, camL_opt, distL, camR_opt, distR, R, T, _E, _F = cv.stereoCalibrate(
            objpoints,
            imgpointsL,
            imgpointsR,
            camL_opt,
            distL,
            camR_opt,
            distR,
            frame_size,
            criteria=_STEREO_CRIT,
            flags=cv.CALIB_FIX_INTRINSIC,
        )

        rectL, rectR, projL, projR, Q, _roiL, _roiR = cv.stereoRectify(
            camL_opt,
            distL,
            camR_opt,
            distR,
            frame_size,
            R,
            T,
            alpha=self.rectify_alpha,
        )

        fx_rect = float(projR[0, 0])
        fov_x_deg = float(math.degrees(2 * math.atan(frame_size[0] / (2 * fx_rect))))

        stereoMapL_x, stereoMapL_y = cv.initUndistortRectifyMap(
            camL_opt, distL, rectL, projL, frame_size, cv.CV_16SC2
        )
        stereoMapR_x, stereoMapR_y = cv.initUndistortRectifyMap(
            camR_opt, distR, rectR, projR, frame_size, cv.CV_16SC2
        )

        return CalibrationResult(
            stereoMapL_x=stereoMapL_x,
            stereoMapL_y=stereoMapL_y,
            stereoMapR_x=stereoMapR_x,
            stereoMapR_y=stereoMapR_y,
            Q=Q,
            projL=projL,
            projR=projR,
            K_rect=proj_to_K(projL),
            fx_rect=fx_rect,
            fov_x_deg=fov_x_deg,
            frame_size=frame_size,
        )


# Path is only imported so callers can pass pathlib objects to glob.glob —
# OpenCV expects strings, but pathlib is more idiomatic upstream.
__all__ = ["StereoCalibrator", "Path"]
