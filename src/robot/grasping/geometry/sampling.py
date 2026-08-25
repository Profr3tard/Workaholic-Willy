"""
Deterministic point-cloud sampling utilities.

These helpers all return integer index arrays into the input cloud.
Callers slice their own attribute arrays (points, normals, colours)
with the same index, which keeps every per-point side-channel in sync
and avoids hidden allocations.
"""

from __future__ import annotations

import numpy as np

from ._validation import as_points_nx3

__all__ = [
    "farthest_point_sample_indices",
    "uniform_sample_indices",
    "voxel_downsample_indices",
]


def uniform_sample_indices(n_points: int, max_samples: int) -> np.ndarray:
    """Return up to ``max_samples`` evenly spaced indices in ``[0, n_points)``."""
    if n_points < 0:
        raise ValueError("n_points must be >= 0")
    if max_samples < 1:
        raise ValueError("max_samples must be >= 1")
    if n_points == 0:
        return np.empty((0,), dtype=np.int64)
    if n_points <= max_samples:
        return np.arange(n_points, dtype=np.int64)
    # Round (not truncate) so the picks are as evenly spaced as integer indices allow; endpoints
    # 0 and n_points-1 are exact either way.
    indices = np.round(np.linspace(0, n_points - 1, num=max_samples)).astype(np.int64)
    return np.unique(indices)


def voxel_downsample_indices(
    points_mm: np.ndarray,
    voxel_size_mm: float,
) -> np.ndarray:
    """Return one index per occupied voxel (the first encountered)."""
    if not np.isfinite(voxel_size_mm) or voxel_size_mm <= 0.0:
        raise ValueError("voxel_size_mm must be finite and > 0")
    points = as_points_nx3(points_mm)
    if points.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    voxels = np.floor(points / float(voxel_size_mm)).astype(np.int64)
    _, first_indices = np.unique(voxels, axis=0, return_index=True)
    return np.sort(first_indices.astype(np.int64))


def farthest_point_sample_indices(
    points_mm: np.ndarray,
    num_samples: int,
    *,
    seed_index: int | None = None,
) -> np.ndarray:
    """Greedy farthest-point sampling for diverse contact-pair seeds.

    Selects deterministic, distinct points by iteratively choosing the point
    farthest from the current sample set. Stops when the requested sample count
    is reached or no additional distinct positions remain.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")
    points = as_points_nx3(points_mm)
    n_points = points.shape[0]
    if n_points == 0:
        return np.empty((0,), dtype=np.int64)
    if num_samples >= n_points:
        return np.arange(n_points, dtype=np.int64)

    if seed_index is None:
        seed = int(np.argmax(np.einsum("ij,ij->i", points, points)))
    else:
        if not 0 <= seed_index < n_points:
            raise ValueError(f"seed_index must be in [0, {n_points})")
        seed = int(seed_index)

    chosen = [seed]
    distances = np.einsum("ij,ij->i", points - points[seed], points - points[seed])
    for _ in range(1, num_samples):
        next_idx = int(np.argmax(distances))
        if distances[next_idx] <= 0.0:
            break  # every remaining point coincides with an already-chosen one
        chosen.append(next_idx)
        new_distances = np.einsum(
            "ij,ij->i", points - points[next_idx], points - points[next_idx]
        )
        distances = np.minimum(distances, new_distances)
    return np.asarray(chosen, dtype=np.int64)
