"""
Umeyama rigid-body point registration.

Pure-numpy; no OpenCV dependency. Finds the rotation ``R`` and translation
``t`` such that ``dst ≈ R @ src + t`` in a least-squares sense.

The algorithm is the closed-form SVD solution described in Umeyama (1991)
"Least-Squares Estimation of Transformation Parameters Between Two Point
Patterns", IEEE TPAMI 13(4):376-380.
"""

from __future__ import annotations

import numpy as np

from src.calibration.constants import CALIBRATION_LOG_DIR, UMEYAMA_RIGID_LOG_FILE
from src.utility.log_cfg import create_logger

__all__ = ["UmeyamaRigid"]

logger = create_logger("UmeyamaRigid", UMEYAMA_RIGID_LOG_FILE, log_dir=CALIBRATION_LOG_DIR)


class UmeyamaRigid:
    """Optimal rigid registration between two corresponding 3-D point sets.

    Solves: ``dst ≈ R @ src + t``   (least-squares sense, no scale).

    Usage::

        solver = UmeyamaRigid()
        R, t, rmse = solver.solve(pts_src, pts_dst)

    Constraints:

    * At least 3 non-collinear point pairs.
    * ``pts_src.shape == pts_dst.shape`` with ``shape[1] == 3``.
    """

    def solve(
        self,
        pts_src: np.ndarray,
        pts_dst: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Find ``R, t`` such that ``pts_dst ≈ R @ pts_src + t``.

        Parameters
        ----------
        pts_src:
            (N, 3) source point cloud.
        pts_dst:
            (N, 3) target point cloud; same ordering as *pts_src*.

        Returns
        -------
        R:
            (3, 3) rotation matrix in SO(3).
        t:
            (3,) translation vector in the same unit as the inputs.
        rmse:
            Root-mean-square residual error (same unit as inputs).

        Raises
        ------
        ValueError
            If shapes are incompatible or fewer than 3 pairs are provided.
        """
        src = np.asarray(pts_src, dtype=np.float64)
        dst = np.asarray(pts_dst, dtype=np.float64)

        if src.ndim != 2 or src.shape[1] != 3:
            raise ValueError(f"pts_src must be (N, 3), got {src.shape}")
        if src.shape != dst.shape:
            raise ValueError(
                f"pts_src and pts_dst must have equal shape; "
                f"got {src.shape} vs {dst.shape}"
            )
        if src.shape[0] < 3:
            raise ValueError(
                f"At least 3 point pairs required, got {src.shape[0]}"
            )

        mu_src = src.mean(axis=0)
        mu_dst = dst.mean(axis=0)
        src_c = src - mu_src
        dst_c = dst - mu_dst

        H = src_c.T @ dst_c / src.shape[0]
        U, _S, Vt = np.linalg.svd(H)

        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            logger.debug("Reflection in the SVD solution; flipping the last singular vector.")
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = mu_dst - R @ mu_src

        residuals = dst - (src @ R.T + t)
        rmse = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))

        logger.info(
            "Registered %d point pairs: rmse=%.4f, |t|=%.4f (input units).",
            src.shape[0],
            rmse,
            float(np.linalg.norm(t)),
        )
        return R, t, rmse
