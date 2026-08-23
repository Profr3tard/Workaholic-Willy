"""Mask completion utilities for perception."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import numpy as np
import cv2 as cv

__all__ = [
    "MaskCompletion",
    "DEFAULT_MASK_COMPLETION",
    "DEFAULT_MIN_FILL_RATIO",
    "complete_mask",
]

#: A mask covering less than this share of its reference box is treated as incomplete. 0.8 is the
#: shipped value; the measurement above is what it actually selects for.
DEFAULT_MIN_FILL_RATIO = 0.8


class MaskCompletion(StrEnum):
    """What to do with a mask that does not fill its reference box."""

    #: Never replace the mask. SAM2's silhouette is used exactly as segmented.
    NONE = "none"

    #: Replace it with the DETECTOR's axis-aligned box. The shipped behaviour: it can restore extent
    #: the segmenter dropped, and it destroys the object's orientation whenever it fires.
    AXIS_ALIGNED_BOX = "axis_aligned_box"

    #: Replace it with the mask's own minimum-area ORIENTED box. Fills concavity and interior holes
    #: while preserving the principal axis.
    ORIENTED_BOX = "oriented_box"


#: The shipped policy, and the ONE place it is decided. Both perception sources take it, so a real
#: cell and the Isaac cell cannot drift apart on the transform that decides a grasp's closing axis.
DEFAULT_MASK_COMPLETION = MaskCompletion.NONE


def _detector_box(mask: np.ndarray, det: Any) -> tuple[int, int, int, int] | None:
    """The detector's box, clipped to the image. ``None`` when there is nothing usable."""
    raw = getattr(det, "box", None)
    if raw is None:
        return None
    height, width = mask.shape[:2]
    x0, y0, x1, y1 = (int(round(float(c))) for c in raw)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    return None if (x1 <= x0 or y1 <= y0) else (x0, y0, x1, y1)


def _fill_axis_aligned(mask: np.ndarray, box: tuple[int, int, int, int], ratio: float) -> np.ndarray:
    x0, y0, x1, y1 = box
    area = (x1 - x0) * (y1 - y0)
    if int(mask.sum()) >= ratio * area:
        return mask                       # already a near-complete silhouette: keep SAM2's precision
    filled = np.zeros(mask.shape[:2], dtype=bool)
    filled[y0:y1, x0:x1] = True
    return filled


def _fill_oriented(mask: np.ndarray, ratio: float) -> np.ndarray:
    """Fill to ``cv.minAreaRect`` of the mask itself, when the mask underfills it."""
    ys, xs = np.nonzero(mask)
    if xs.size < 3:
        return mask
    points = np.stack([xs, ys], axis=1).astype(np.int32)
    rect = cv.minAreaRect(points)
    (_, (w, h), _) = rect
    area = float(w) * float(h)
    if area <= 0.0 or int(mask.sum()) >= ratio * area:
        return mask
    filled = np.zeros(mask.shape[:2], dtype=np.uint8)
    cv.fillPoly(filled, [np.round(cv.boxPoints(rect)).astype(np.int32)], 1)
    return filled.astype(bool)


def complete_mask(
    mask: np.ndarray,
    det: Any = None,
    *,
    policy: MaskCompletion = DEFAULT_MASK_COMPLETION,
    min_fill_ratio: float = DEFAULT_MIN_FILL_RATIO,
) -> np.ndarray:
    """Apply the configured mask-completion policy. Returns a boolean mask."""
    # copy=False so an already-boolean mask comes back as the SAME object when nothing is replaced.
    # Callers assert that identity to mean "untouched", and the copy would also be pure waste here.
    mask = np.asarray(mask).astype(bool, copy=False)
    if mask.ndim != 2 or policy is MaskCompletion.NONE:
        return mask
    if policy is MaskCompletion.ORIENTED_BOX:
        return _fill_oriented(mask, float(min_fill_ratio))
    box = _detector_box(mask, det)
    return mask if box is None else _fill_axis_aligned(mask, box, float(min_fill_ratio))
