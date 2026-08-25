"""Shared neighbour-query index over an ``(N, 3)`` point cloud."""

from __future__ import annotations

import numpy as np

__all__ = ["RadiusIndex"]


class RadiusIndex:
    """Radius and k-nearest-neighbour queries over a fixed point cloud."""

    def __init__(self, points: np.ndarray) -> None:
        self.points = points
        try:
            from scipy.spatial import cKDTree  # type: ignore
        except Exception:  # pragma: no cover - SciPy is present in tests
            self._tree = None
        else:
            self._tree = cKDTree(points)

    def query_radius(self, point: np.ndarray, radius: float) -> np.ndarray:
        """Indices of every point within ``radius`` of ``point``."""
        if self._tree is not None:
            return np.asarray(self._tree.query_ball_point(point, radius), dtype=np.int64)
        delta = self.points - point
        dist2 = np.einsum("ij,ij->i", delta, delta)
        return np.nonzero(dist2 <= radius * radius)[0].astype(np.int64)

    def query_knn(self, point: np.ndarray, k: int) -> np.ndarray:
        """Indices of the ``k`` nearest points to ``point`` (deterministic ties via mergesort)."""
        if self._tree is not None:
            _, indices = self._tree.query(point, k=k)
            return np.atleast_1d(np.asarray(indices, dtype=np.int64))
        delta = self.points - point
        dist2 = np.einsum("ij,ij->i", delta, delta)
        order = np.argsort(dist2, kind="mergesort")
        return order[:k].astype(np.int64)
