"""ChArUco-based stereo calibration.

``StereoCalibrator`` turns a folder of stereo image pairs into a
:class:`CalibrationResult` (per-camera intrinsics, rectification rotations,
projection matrices and the ``Q`` reprojection matrix; the remap tables are
derived from those). A ChArUco board tolerates partial views and gives each
corner a unique id, so corners are matched across the stereo pair by id.
Persistence is handled by :class:`StereoCalibrationStore`.
"""

from __future__ import annotations

import glob
import time
from typing import List, Optional, Tuple

import cv2 as cv
import numpy as np

from src.calibration.constants import CALIBRATION_LOG_DIR, STEREO_CALIBRATOR_LOG_FILE
from src.calibration.exceptions import StereoCalibrationError
from src.calibration.stereo.config import CalibrationResult
from src.calibration.stereo.sub_modules.aruco_esti import resolve_aruco_dictionary
from src.utility.log_cfg import create_logger

# stereoCalibrate iteration criteria.
_STEREO_CRIT = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
# Minimum ChArUco corners a stereo pair must share (across L and R) to be usable.
_MIN_CORNERS = 6
# Minimum number of usable image pairs required for a stable calibration.
_MIN_PAIRS = 10

__all__ = ["StereoCalibrator"]

logger = create_logger("StereoCalibrator", STEREO_CALIBRATOR_LOG_FILE, log_dir=CALIBRATION_LOG_DIR)


class StereoCalibrator:
    """Compute stereo intrinsics, extrinsics and rectification from a ChArUco board."""

    def __init__(
        self,
        squares_x: int,
        squares_y: int,
        square_length_mm: float,
        marker_length_mm: float,
        aruco_dict_name: str,
        rectify_alpha: float = 0.0,
    ) -> None:
        if int(squares_x) <= 1 or int(squares_y) <= 1:
            raise StereoCalibrationError("squares_x and squares_y must be > 1")
        if not 0.0 < float(marker_length_mm) < float(square_length_mm):
            raise StereoCalibrationError("require 0 < marker_length_mm < square_length_mm")
        self.rectify_alpha = float(rectify_alpha)
        if not 0.0 <= self.rectify_alpha <= 1.0:
            raise StereoCalibrationError("rectify_alpha must be between 0 and 1")
        self._board = cv.aruco.CharucoBoard(
            (int(squares_x), int(squares_y)),
            float(square_length_mm),
            float(marker_length_mm),
            resolve_aruco_dictionary(aruco_dict_name),
        )
        self._detector = cv.aruco.CharucoDetector(self._board)
        # Board geometry sets the scale of every result below, so it is logged as configured.
        logger.debug(
            "ChArUco board %dx%d squares, square=%.2f mm, marker=%.2f mm, dict=%s, alpha=%.2f.",
            int(squares_x),
            int(squares_y),
            float(square_length_mm),
            float(marker_length_mm),
            aruco_dict_name,
            self.rectify_alpha,
        )

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

        logger.info(
            "Calibrating from %d image pairs (left=%s, right=%s).",
            len(left_imgs),
            left_glob,
            right_glob,
        )
        started = time.perf_counter()

        objpoints: List[np.ndarray] = []
        imgpointsL: List[np.ndarray] = []
        imgpointsR: List[np.ndarray] = []

        # Per-pair detection is the tight loop here, so rejects are counted and reported once.
        unusable = 0
        for left_path, right_path in zip(left_imgs, right_imgs):
            pair = self._detect_pair(left_path, right_path)
            if pair is None:
                unusable += 1
                continue
            (gL_shape, objp, cornersL, cornersR) = pair
            if frame_size is None:
                frame_size = (gL_shape[1], gL_shape[0])
            objpoints.append(objp)
            imgpointsL.append(cornersL)
            imgpointsR.append(cornersR)

        if unusable:
            # Degraded, not fatal: the solve continues on whatever is left. A board half out of
            # frame shows up as a high reject count long before the RMS looks wrong.
            logger.warning(
                "%d of %d image pairs unusable (board not found, or < %d corners shared L/R).",
                unusable,
                len(left_imgs),
                _MIN_CORNERS,
            )
        if len(objpoints) < _MIN_PAIRS:
            raise StereoCalibrationError(
                f"Too few usable calibration pairs: {len(objpoints)} (need >= {_MIN_PAIRS})."
            )
        assert frame_size is not None  # for the type checker

        result = self._calibrate_from_correspondences(
            frame_size, objpoints, imgpointsL, imgpointsR
        )
        logger.info(
            "Calibration done in %.1f s: %d usable pairs, frame_size=%s.",
            time.perf_counter() - started,
            len(objpoints),
            result.frame_size,
        )
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _detect_pair(
        self, left_path: str, right_path: str
    ) -> Optional[Tuple[Tuple[int, int], np.ndarray, np.ndarray, np.ndarray]]:
        """Detect ChArUco corners in a stereo pair and match them across L/R by id."""
        L = cv.imread(left_path)
        R = cv.imread(right_path)
        if L is None or R is None:
            return None

        gL = cv.cvtColor(L, cv.COLOR_BGR2GRAY)
        gR = cv.cvtColor(R, cv.COLOR_BGR2GRAY)
        cornersL, idsL, _, _ = self._detector.detectBoard(gL)
        cornersR, idsR, _, _ = self._detector.detectBoard(gR)
        if idsL is None or idsR is None or len(idsL) < _MIN_CORNERS or len(idsR) < _MIN_CORNERS:
            return None

        matched = self._match_by_id(cornersL, idsL, cornersR, idsR)
        if matched is None:
            return None
        objp, imgL, imgR = matched
        return gL.shape, objp, imgL, imgR

    def _match_by_id(
        self,
        cornersL: np.ndarray,
        idsL: np.ndarray,
        cornersR: np.ndarray,
        idsR: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Keep only ChArUco corners seen in both images; return (object, left, right) points."""
        board_corners = np.asarray(self._board.getChessboardCorners(), dtype=np.float32)
        ids_left = [int(i) for i in idsL.flatten()]
        ids_right = [int(i) for i in idsR.flatten()]
        index_l = {cid: k for k, cid in enumerate(ids_left)}
        index_r = {cid: k for k, cid in enumerate(ids_right)}
        common = [cid for cid in ids_left if cid in index_r]
        if len(common) < _MIN_CORNERS:
            return None
        obj = np.array([board_corners[cid] for cid in common], dtype=np.float32).reshape(-1, 1, 3)
        img_l = np.array([cornersL[index_l[cid]] for cid in common], dtype=np.float32).reshape(-1, 1, 2)
        img_r = np.array([cornersR[index_r[cid]] for cid in common], dtype=np.float32).reshape(-1, 1, 2)
        return obj, img_l, img_r

    def _calibrate_from_correspondences(
        self,
        frame_size: Tuple[int, int],
        objpoints: List[np.ndarray],
        imgpointsL: List[np.ndarray],
        imgpointsR: List[np.ndarray],
    ) -> CalibrationResult:
        """Run mono -> stereo -> rectify from already-matched corner correspondences."""
        rms_left, camL, distL, _, _ = cv.calibrateCamera(  # type: ignore[call-overload]
            objpoints, imgpointsL, frame_size, None, None
        )
        rms_right, camR, distR, _, _ = cv.calibrateCamera(  # type: ignore[call-overload]
            objpoints, imgpointsR, frame_size, None, None
        )

        camL_opt, _ = cv.getOptimalNewCameraMatrix(camL, distL, frame_size, 1.0)
        camR_opt, _ = cv.getOptimalNewCameraMatrix(camR, distR, frame_size, 1.0)

        stereo_rms, camL_opt, distL, camR_opt, distR, R, T, _E, _F = cv.stereoCalibrate(
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

        # Reprojection RMS is in pixels, the baseline in the board's unit (mm). Logging both
        # turns "the rig is off" into a number comparable against the last calibration.
        logger.info(
            "Reprojection RMS: left=%.3f px, right=%.3f px, stereo=%.3f px; baseline=%.1f mm.",
            float(rms_left),
            float(rms_right),
            float(stereo_rms),
            float(np.linalg.norm(np.asarray(T, dtype=np.float64))),
        )

        return CalibrationResult(
            camL=camL_opt,
            distL=distL,
            rectL=rectL,
            projL=projL,
            camR=camR_opt,
            distR=distR,
            rectR=rectR,
            projR=projR,
            Q=Q,
            frame_size=frame_size,
        )
