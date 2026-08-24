"""Provides container wall geometry as point-cloud collision data.

Converts declared container walls into points consumed by the existing
grasp collision checks, allowing candidate grasps to reject finger
collisions with bin walls. Geometry is emitted in the coordinate frames
expected by the downstream collision and grasp-generation APIs.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["container_wall_points_base_mm"]


def _axis_samples(low: float, high: float, step: float) -> np.ndarray:
    """Inclusive samples from ``low`` to ``high``, never fewer than the two endpoints."""

    span = high - low
    count = max(int(np.ceil(span / step)), 1) + 1
    return np.linspace(low, high, count, dtype=np.float64)


def container_wall_points_base_mm(
    interior_min_mm: Sequence[float],
    interior_max_mm: Sequence[float],
    *,
    thickness_mm: float = 5.0,
    sample_mm: float = 8.0,
) -> np.ndarray:
    """Generates the four vertical walls of an interior box as BASE-frame points.

    Walls are sampled as shells of the configured thickness from floor to rim.
    The floor is excluded because it is handled by the support plane, and points
    above the rim are excluded so grasps may pass over the container wall.
    """

    low = np.asarray(interior_min_mm, dtype=np.float64).reshape(3)
    high = np.asarray(interior_max_mm, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        raise ValueError("container interior corners must be finite")
    if np.any(high <= low):
        raise ValueError(
            f"container interior_max_mm {high.tolist()} must exceed interior_min_mm {low.tolist()} "
            "on every axis"
        )
    if thickness_mm <= 0.0 or sample_mm <= 0.0:
        raise ValueError("thickness_mm and sample_mm must be > 0")

    z = _axis_samples(low[2], high[2], sample_mm)
    depth = _axis_samples(0.0, thickness_mm, sample_mm)
    parts: list[np.ndarray] = []

    # Two walls per horizontal axis. For axis 0 the wall spans y and z and is displaced in x; for
    # axis 1 it spans x and z.
    for axis in (0, 1):
        span_axis = 1 - axis
        span = _axis_samples(low[span_axis] - thickness_mm, high[span_axis] + thickness_mm, sample_mm)
        grid_span, grid_z, grid_depth = np.meshgrid(span, z, depth, indexing="ij")
        flat_span = grid_span.reshape(-1)
        flat_z = grid_z.reshape(-1)
        flat_depth = grid_depth.reshape(-1)
        for sign, face in ((-1.0, low[axis]), (1.0, high[axis])):
            points = np.empty((flat_span.size, 3), dtype=np.float64)
            points[:, axis] = face + sign * flat_depth
            points[:, span_axis] = flat_span
            points[:, 2] = flat_z
            parts.append(points)

    return np.vstack(parts)
