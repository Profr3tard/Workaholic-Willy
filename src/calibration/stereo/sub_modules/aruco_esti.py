"""ArUco marker pose estimation in rectified left frames."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import cv2 as cv
import numpy as np

from src.calibration.exceptions import StereoCalibrationError

logger = logging.getLogger(__name__)

__all__ = ["ArucoPoseEstimator", "resolve_aruco_dictionary"]


def resolve_aruco_dictionary(name: str):
    """Resolve a dictionary name (``'DICT_5X5_100'`` or ``'5X5_100'``) to a cv2 ArUco dictionary."""
    d = name.upper().strip()
    if not d.startswith("DICT_"):
        d = "DICT_" + d
    if not hasattr(cv.aruco, d):
        raise StereoCalibrationError(f"Unknown ArUco dictionary: {name}")
    return cv.aruco.getPredefinedDictionary(getattr(cv.aruco, d))


def _rvec_tvec_to_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rotation, _ = cv.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3))
    translation = np.asarray(tvec, dtype=np.float64).reshape(3)
    T_cam_to_marker = np.eye(4, dtype=np.float64)
    T_cam_to_marker[:3, :3] = rotation
    T_cam_to_marker[:3, 3] = translation
    return T_cam_to_marker


class ArucoPoseEstimator:
    """Detects ArUco markers and returns ``T_cam_to_marker`` 4x4 transforms."""

    def __init__(
        self,
        marker_length_mm: float = 50.0,
        dict_name: str = "DICT_5X5_100",
    ) -> None:
        if not hasattr(cv, "aruco"):
            raise StereoCalibrationError(
                "cv2.aruco not found. Please install opencv-contrib-python."
            )

        self.marker_length = float(marker_length_mm)
        if self.marker_length <= 0.0:
            raise StereoCalibrationError("marker_length_mm must be > 0")
        self.aruco_dict = resolve_aruco_dictionary(dict_name)
        self.aruco_params = cv.aruco.DetectorParameters()
        # Sub-pixel corner refinement markedly improves solvePnP pose accuracy on fiducials.
        self.aruco_params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Reusable grayscale buffer for the realtime path.
        self._gray: Optional[np.ndarray] = None

    def estimate(
        self,
        rect_left_bgr: np.ndarray,
        K_rect: Optional[np.ndarray],
        dist_rect: np.ndarray,
        target_id: Optional[int] = None,
    ):
        """Detect markers and estimate their pose relative to the camera.

        Args:
            rect_left_bgr: rectified left BGR frame.
            K_rect: rectified intrinsics (3x3). Required.
            dist_rect: distortion coefficients (zeros for rectified input).
            target_id: if given, return only that marker's transform (or None).

        Returns:
            dict[int, 4x4] when ``target_id`` is None; otherwise a single
            4x4 matrix or ``None`` if the marker was not detected.
        """
        if K_rect is None:
            raise StereoCalibrationError("K_rect is not available (set rectified intrinsics).")
        K = np.asarray(K_rect, dtype=np.float64)
        if K.shape != (3, 3) or not np.all(np.isfinite(K)):
            raise StereoCalibrationError("K_rect must be a finite 3x3 matrix")
        dist = np.asarray(dist_rect, dtype=np.float64).reshape(-1)

        gray = self._to_gray(rect_left_bgr)
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            logger.debug("ArUco: no markers detected.")
            return {} if target_id is None else None

        # cv.aruco.estimatePoseSingleMarkers was removed in OpenCV >= 4.7 (we pin 4.13). Pose each
        # detected marker with cv.solvePnP over its known square corners (mirrors willy_sim/hand_eye.py).
        half = self.marker_length / 2.0
        # cv2.aruco corner order is TL, TR, BR, BL -> object points with the marker +Z out of plane.
        obj = np.array(
            [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
            dtype=np.float64,
        )
        result: Dict[int, np.ndarray] = {}
        for i, mid in enumerate(ids.flatten()):
            img_pts = np.asarray(corners[i], dtype=np.float64).reshape(4, 2)
            ok, rvec, tvec = cv.solvePnP(obj, img_pts, K, dist, flags=cv.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                logger.debug("ArUco: solvePnP failed for marker %d.", int(mid))
                continue
            result[int(mid)] = _rvec_tvec_to_matrix(rvec, tvec)

        if target_id is None:
            return result
        return result.get(int(target_id))

    # ------------------------------------------------------------------
    def _to_gray(self, bgr: np.ndarray) -> np.ndarray:
        if bgr.ndim == 2:
            return bgr
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise StereoCalibrationError(
                f"rect_left_bgr must be grayscale or BGR, got shape {bgr.shape}"
            )
        h, w = bgr.shape[:2]
        if self._gray is None or self._gray.shape != (h, w):
            self._gray = np.empty((h, w), dtype=np.uint8)
        cv.cvtColor(bgr, cv.COLOR_BGR2GRAY, dst=self._gray)
        return self._gray