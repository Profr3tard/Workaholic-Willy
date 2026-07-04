"""SGBM-based disparity computation with optional WLS refinement.

This module wraps OpenCV's ``StereoSGBM`` matcher and adds:

* Validated configuration (``num_disparities`` divisible by 16, odd ``block_size``).
* Optional WLS post-filter (``cv.ximgproc.createDisparityWLSFilter``) that
  smooths disparities in low-texture regions while preserving edges.
  Requires ``opencv-contrib-python``.
* Optional ``MODE_SGBM_3WAY`` for ~2x speedup on 720p frames.
* Optional temporal exponential moving average for realtime mode.

The ``compute`` API contract is stable: ``(disparity_float32, valid_mask_bool)``.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import cv2 as cv
import numpy as np

from backend.src.calibration.exceptions import StereoCalibrationError
from backend.src.calibration.helpers import validate_image_pair_shapes

if TYPE_CHECKING:  # pragma: no cover - typing only
    from backend.config.schema.camera import StereoMatcherConfig

__all__ = ["DisparityComputer"]

# OpenCV's recommended SGBM smoothness penalty heuristics. P1 is the
# small disparity-change penalty (per pixel), P2 is the large-change
# penalty. Factor 3 corresponds to channels — kept as the OpenCV default
# even though we feed grayscale.
_P1_FACTOR = 8 * 3
_P2_FACTOR = 32 * 3


class DisparityComputer:
    """Computes disparity maps from rectified stereo image pairs."""

    def __init__(self, cfg: StereoMatcherConfig) -> None:
        self._cfg = cfg

        block = int(cfg.block_size)
        p1 = int(cfg.p1) if cfg.p1 is not None else _P1_FACTOR * block * block
        p2 = int(cfg.p2) if cfg.p2 is not None else _P2_FACTOR * block * block

        self._left_matcher = cv.StereoSGBM_create(  # type: ignore[attr-defined]
            minDisparity=int(cfg.min_disparity),
            numDisparities=int(cfg.num_disparities),
            blockSize=block,
            P1=p1,
            P2=p2,
            uniquenessRatio=int(cfg.uniqueness_ratio),
            speckleWindowSize=int(cfg.speckle_window_size),
            speckleRange=int(cfg.speckle_range),
            disp12MaxDiff=int(cfg.disp12_max_diff),
        )
        if cfg.mode == "sgbm_3way":
            self._left_matcher.setMode(cv.STEREO_SGBM_MODE_SGBM_3WAY)

        # WLS post-filter — only initialised when contrib is available and
        # the user enabled it. Falls back to plain SGBM otherwise.
        self._right_matcher = None
        self._wls = None
        if cfg.wls.enabled:
            ximgproc = getattr(cv, "ximgproc", None)
            if ximgproc is not None and hasattr(ximgproc, "createDisparityWLSFilter"):
                if cfg.wls.lr_check:
                    self._right_matcher = ximgproc.createRightMatcher(self._left_matcher)
                self._wls = ximgproc.createDisparityWLSFilter(
                    matcher_left=self._left_matcher,
                )
                self._wls.setLambda(float(cfg.wls.lambda_))
                self._wls.setSigmaColor(float(cfg.wls.sigma_color))

        # Reusable grayscale buffers — avoid reallocating one cv.Mat per
        # frame in the realtime loop.
        self._gL: Optional[np.ndarray] = None
        self._gR: Optional[np.ndarray] = None

        # Temporal smoothing state.
        self._alpha = float(cfg.temporal_alpha)
        self._prev_disp: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute(
        self,
        rect_left_bgr: np.ndarray,
        rect_right_bgr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute disparity for a rectified BGR stereo pair.

        Returns:
            disparity (float32, pixels): invalid samples are NaN.
            valid_mask (bool): True where disparity is finite.
        """
        left = np.asarray(rect_left_bgr)
        right = np.asarray(rect_right_bgr)
        try:
            validate_image_pair_shapes(left, right)
        except Exception as exc:
            raise StereoCalibrationError("cannot compute disparity for incompatible image pair") from exc
        gL = self._to_gray(left, slot="left")
        gR = self._to_gray(right, slot="right")

        if self._wls is not None:
            disp_left = self._left_matcher.compute(gL, gR)
            disp_right = (
                self._right_matcher.compute(gR, gL)
                if self._right_matcher is not None
                else None
            )
            raw = self._wls.filter(disp_left, left, None, disp_right)
        else:
            raw = self._left_matcher.compute(gL, gR)

        # OpenCV returns int16 scaled by 16. Divide to get fractional pixels.
        disp = raw.astype(np.float32) * (1.0 / 16.0)

        invalid = disp <= 0.0
        if invalid.any():
            disp[invalid] = np.nan

        if self._alpha > 0.0:
            disp = self._apply_temporal(disp)

        valid_mask = np.isfinite(disp)
        return disp, valid_mask

    def reset_temporal(self) -> None:
        """Clear the temporal smoothing buffer (e.g. on stream restart)."""
        self._prev_disp = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _to_gray(self, bgr: np.ndarray, slot: str) -> np.ndarray:
        if bgr.ndim == 2:
            return bgr
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise StereoCalibrationError(
                f"expected BGR image with shape (H, W, 3), got {bgr.shape}"
            )
        prev = self._gL if slot == "left" else self._gR
        h, w = bgr.shape[:2]
        if prev is None or prev.shape != (h, w):
            prev = np.empty((h, w), dtype=np.uint8)
            if slot == "left":
                self._gL = prev
            else:
                self._gR = prev
        cv.cvtColor(bgr, cv.COLOR_BGR2GRAY, dst=prev)
        return prev

    def _apply_temporal(self, disp: np.ndarray) -> np.ndarray:
        """One-pole EMA on disparity, restricted to mutually-valid pixels."""
        prev = self._prev_disp
        if prev is None or prev.shape != disp.shape:
            self._prev_disp = disp.copy()
            return disp

        alpha = self._alpha
        blended = disp.copy()
        both_valid = np.isfinite(disp) & np.isfinite(prev)
        blended[both_valid] = (
            alpha * disp[both_valid] + (1.0 - alpha) * prev[both_valid]
        )
        # Where the new frame has a hole but the previous one was valid,
        # carry the previous value forward to smooth temporary dropouts.
        only_prev = ~np.isfinite(disp) & np.isfinite(prev)
        blended[only_prev] = prev[only_prev]
        self._prev_disp = blended
        return blended