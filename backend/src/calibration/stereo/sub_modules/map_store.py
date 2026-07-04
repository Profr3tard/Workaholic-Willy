"""Persistence layer for stereo calibration artefacts.

A ``StereoMapStore`` reads/writes a :class:`CalibrationResult` (and an
optional ``T_cam_to_base`` extrinsics matrix) to an OpenCV ``FileStorage``
XML/YAML file. Splitting it out of the calibration logic makes the math
side easier to test in isolation and lets us swap in a different format
later without touching the calibrator.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2 as cv
import numpy as np

from backend.src.calibration.exceptions import StereoCalibrationError
from backend.src.calibration.stereo.config import CalibrationResult
from backend.src.geometry.validation import validate_homogeneous_matrix

logger = logging.getLogger(__name__)


class StereoMapStore:
    """Read/write stereo rectification maps + Q matrix + optional extrinsics."""

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def save(
        self,
        filepath: str,
        result: CalibrationResult,
        T_cam_to_base: Optional[np.ndarray] = None,
    ) -> None:
        target = Path(filepath)
        target.parent.mkdir(parents=True, exist_ok=True)
        fs = cv.FileStorage(str(target), cv.FileStorage_WRITE)
        if not fs.isOpened():
            raise OSError(f"Could not open '{filepath}' for writing.")
        try:
            fs.write("stereoMapL_x", result.stereoMapL_x)
            fs.write("stereoMapL_y", result.stereoMapL_y)
            fs.write("stereoMapR_x", result.stereoMapR_x)
            fs.write("stereoMapR_y", result.stereoMapR_y)
            fs.write("Q", result.Q)
            fs.write("projL", result.projL)
            fs.write("projR", result.projR)
            fs.write("K_rect", result.K_rect)
            fs.write("frame_size_w", int(result.frame_size[0]))
            fs.write("frame_size_h", int(result.frame_size[1]))

            if T_cam_to_base is not None:
                T_valid = validate_homogeneous_matrix(T_cam_to_base, name="T_cam_to_base")
                fs.write("T_cam_to_base", T_valid)
                fs.write("T_cam_to_base_valid", 1)
            else:
                fs.write("T_cam_to_base_valid", 0)
        finally:
            fs.release()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    def load(
        self, filepath: str
    ) -> Tuple[CalibrationResult, Optional[np.ndarray]]:
        fs = cv.FileStorage(filepath, cv.FileStorage_READ)
        if not fs.isOpened():
            raise OSError(f"Could not open '{filepath}' for reading.")
        try:
            stereoMapL_x = self._require_mat(fs, "stereoMapL_x")
            stereoMapL_y = self._require_mat(fs, "stereoMapL_y")
            stereoMapR_x = self._require_mat(fs, "stereoMapR_x")
            stereoMapR_y = self._require_mat(fs, "stereoMapR_y")
            Q = self._require_mat(fs, "Q")
            projL = self._require_mat(fs, "projL")
            projR = self._require_mat(fs, "projR")
            K_rect = self._require_mat(fs, "K_rect")

            fw = int(fs.getNode("frame_size_w").real())
            fh = int(fs.getNode("frame_size_h").real())
            if fw <= 0 or fh <= 0:
                raise StereoCalibrationError("Calibration file has invalid frame_size")

            valid_node = fs.getNode("T_cam_to_base_valid")
            if not valid_node.empty() and int(valid_node.real()) == 1:
                T_cam_to_base = validate_homogeneous_matrix(
                    self._require_mat(fs, "T_cam_to_base"),
                    name="T_cam_to_base",
                )
            else:
                T_cam_to_base = None
        finally:
            fs.release()

        result = CalibrationResult(
            stereoMapL_x=stereoMapL_x,
            stereoMapL_y=stereoMapL_y,
            stereoMapR_x=stereoMapR_x,
            stereoMapR_y=stereoMapR_y,
            Q=Q,
            projL=projL,
            projR=projR,
            K_rect=K_rect,
            fx_rect=float(projR[0, 0]),
            fov_x_deg=float(np.degrees(2.0 * np.arctan(fw / (2.0 * float(projR[0, 0]))))),
            frame_size=(fw, fh),
        )
        return result, T_cam_to_base

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    @staticmethod
    def print_diagnostics(result: CalibrationResult) -> None:
        """Log the basic geometric sanity-checks for a loaded calibration."""
        Q = result.Q
        P2 = result.projR
        fx = float(P2[0, 0])
        Tx = float(P2[0, 3])
        baseline_from_p = -Tx / fx if fx else float("nan")
        baseline_from_q = -1.0 / Q[3, 2] if Q[3, 2] else float("nan")
        logger.info("fx (px)         = %.2f", fx)
        logger.info("baseline (P2)   = %.1f mm", baseline_from_p)
        logger.info("baseline (Q)    = %.1f mm", baseline_from_q)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _require_mat(fs: "cv.FileStorage", key: str) -> np.ndarray:
        node = fs.getNode(key)
        if node.empty():
            raise StereoCalibrationError(f"Calibration file is missing required key: {key}")
        mat = node.mat()
        if mat is None or mat.size == 0:
            raise StereoCalibrationError(f"Calibration file has empty matrix for key: {key}")
        return mat


__all__ = ["StereoMapStore"]
