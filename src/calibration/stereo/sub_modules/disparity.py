"""SGBM-based disparity computation with optional WLS refinement.

Wraps OpenCV's ``StereoSGBM`` matcher and adds validated configuration
(``num_disparities`` divisible by 16, odd ``block_size``). Three parts are
optional: a WLS post-filter (``cv.ximgproc.createDisparityWLSFilter``, which
needs ``opencv-contrib-python``) that smooths low-texture regions while
preserving edges, ``MODE_SGBM_3WAY`` for a ~2x speedup on 720p frames, and a
temporal exponential moving average for realtime mode.

``compute`` returns ``(disparity_float32, valid_mask_bool)``; that shape is
a stable contract.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import cv2 as cv
import numpy as np

from src.calibration.exceptions import StereoCalibrationError
from src.calibration.helpers import validate_image_pair_shapes

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.camera import StereoMatcherConfig

__all__ = ["DisparityComputer"]

# OpenCV's recommended SGBM smoothness penalties: P1 for a disparity change of
# one pixel, P2 for a larger one. The factor 3 is OpenCV's channel count and
# stays although the input is grayscale.
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

        # WLS post-filter, built only when the config asks for it and contrib
        # supplies the filter. Plain SGBM otherwise.
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

        # Grayscale buffers reused across frames, so the realtime loop
        # allocates no cv.Mat per frame.
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

        Raises:
            StereoCalibrationError: the two images have incompatible shapes.
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

        # The matcher output is int16 with disparity scaled by 16, so divide
        # it to get fractional pixels.
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
        """Convert BGR to grayscale in the reusable buffer for ``slot``.

        For a three-channel input the returned array is that buffer, so the
        next call for the same slot overwrites it. A single-channel input is
        returned unchanged, and then the result aliases the caller's array.
        """
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
        """One-pole EMA on disparity, blending only pixels valid in both frames."""
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
        # A hole in the new frame keeps the previous value, so a short dropout
        # does not punch a gap into the output.
        only_prev = ~np.isfinite(disp) & np.isfinite(prev)
        blended[only_prev] = prev[only_prev]
        self._prev_disp = blended
        return blended