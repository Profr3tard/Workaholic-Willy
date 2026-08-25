"""Topology-risk signals for grasp candidate evaluation.

Derives a deterministic risk score from mask solidity and optionally rejects
structurally risky masks via a configurable threshold. Exposes the score in
telemetry and reports threshold breaches through explicit grasp result reason
codes.
"""

from __future__ import annotations

import cv2 as cv
import numpy as np

__all__ = [
    "depth_continuity_risk",
    "topology_risk_from_mask",
]


def topology_risk_from_mask(mask: np.ndarray) -> float | None:
    """Return ``1 - solidity`` (contour area / convex-hull area) of the largest contour in ``mask``.

    Convex blobs score near ``0.0``; rings, U-shapes, and handles score
    higher. Returns :data:`None` when uncomputable treat :data:`None` as "no signal", not "low
    risk", so a missing score never silently bypasses a threshold gate.
    """
    arr = np.asarray(mask)
    if arr.ndim != 2 or arr.size == 0:
        return None
    binary = (arr > 0).astype(np.uint8) * 255
    if not np.any(binary):
        return None
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv.contourArea)
    area = float(cv.contourArea(contour))
    if area <= 0.0:
        return None
    hull = cv.convexHull(contour)
    hull_area = float(cv.contourArea(hull))
    if hull_area <= 0.0:
        return None
    solidity = float(np.clip(area / hull_area, 0.0, 1.0))
    return 1.0 - solidity


def depth_continuity_risk(
    depth_map: np.ndarray, mask: np.ndarray, *, jump_mm: float = 25.0
) -> float | None:
    """Compute the fraction of mask pixels adjacent to significant depth jumps.

    Returns a value in ``[0.0, 1.0]`` based on finite 4-neighbour depth
    differences above ``jump_mm``, or ``None`` when insufficient valid depth
    data is available.
    """
    depth_arr = np.asarray(depth_map, dtype=np.float64)
    mask_arr = np.asarray(mask)
    if depth_arr.ndim != 2 or mask_arr.shape != depth_arr.shape:
        return None
    mask_bool = mask_arr.astype(bool)
    if not np.any(mask_bool):
        return None
    finite = np.isfinite(depth_arr) & (depth_arr > 0.0)
    valid = mask_bool & finite
    total = int(np.count_nonzero(valid))
    if total < 8:  # too few pixels to make the ratio meaningful
        return None
    jumps = np.zeros_like(valid, dtype=bool)
    for axis in (0, 1):
        d = np.abs(np.diff(depth_arr, axis=axis))
        finite_pair = (
            finite[:-1, :] & finite[1:, :] if axis == 0 else finite[:, :-1] & finite[:, 1:]
        )
        big = (d > float(jump_mm)) & finite_pair
        if axis == 0:
            jumps[:-1, :] |= big & mask_bool[:-1, :]
            jumps[1:, :] |= big & mask_bool[1:, :]
        else:
            jumps[:, :-1] |= big & mask_bool[:, :-1]
            jumps[:, 1:] |= big & mask_bool[:, 1:]
    return float(np.count_nonzero(jumps & valid)) / float(total)
