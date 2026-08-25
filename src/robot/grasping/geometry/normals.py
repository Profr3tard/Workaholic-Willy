"""
Surface-normal estimation for masked point clouds.

Normals are estimated by local PCA: for each point, find neighbours in a
metric radius, compute the covariance of the local patch, and use the
eigenvector with the smallest eigenvalue as the surface normal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._spatial import RadiusIndex
from ._validation import as_points_nx3

__all__ = [
    "NormalEstimationConfig",
    "SurfaceNormals",
    "estimate_surface_normals",
]


@dataclass(frozen=True, slots=True)
class NormalEstimationConfig:
    """Parameters controlling local-PCA normal estimation."""

    radius_mm: float = 15.0
    min_neighbors: int = 6
    max_neighbors: int = 64
    orient_towards_camera: bool = True
    camera_position_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    max_curvature: float | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.radius_mm) or self.radius_mm <= 0.0:
            raise ValueError("radius_mm must be finite and > 0")
        if self.min_neighbors < 3:
            raise ValueError("min_neighbors must be >= 3")
        if self.max_neighbors < self.min_neighbors:
            raise ValueError("max_neighbors must be >= min_neighbors")
        camera = tuple(float(v) for v in self.camera_position_mm)
        if len(camera) != 3 or not all(np.isfinite(camera)):
            raise ValueError("camera_position_mm must contain three finite values")
        object.__setattr__(self, "camera_position_mm", camera)
        if self.max_curvature is not None:
            if not np.isfinite(self.max_curvature) or self.max_curvature < 0.0:
                raise ValueError("max_curvature must be finite and >= 0")


@dataclass(frozen=True, slots=True)
class SurfaceNormals:
    """Normal-estimation result for an input point cloud."""

    normals: np.ndarray
    confidence: np.ndarray
    curvature: np.ndarray
    valid_mask: np.ndarray

    def __post_init__(self) -> None:
        normals = np.asarray(self.normals, dtype=np.float32)
        confidence = np.asarray(self.confidence, dtype=np.float32)
        curvature = np.asarray(self.curvature, dtype=np.float32)
        valid = np.asarray(self.valid_mask, dtype=bool)
        if normals.ndim != 2 or normals.shape[1] != 3:
            raise ValueError(f"normals must be shape (N, 3), got {normals.shape}")
        expected = (normals.shape[0],)
        if confidence.shape != expected:
            raise ValueError(f"confidence must be shape {expected}, got {confidence.shape}")
        if curvature.shape != expected:
            raise ValueError(f"curvature must be shape {expected}, got {curvature.shape}")
        if valid.shape != expected:
            raise ValueError(f"valid_mask must be shape {expected}, got {valid.shape}")
        if not np.all(np.isfinite(normals)):
            raise ValueError("normals must contain only finite values")
        if not np.all(np.isfinite(confidence)):
            raise ValueError("confidence must contain only finite values")
        # Invalid curvature entries are +inf by convention.
        if np.any(curvature[valid] < 0.0) or not np.all(np.isfinite(curvature[valid])):
            raise ValueError("valid curvature entries must be finite and >= 0")
        normals.setflags(write=False)
        confidence.setflags(write=False)
        curvature.setflags(write=False)
        valid.setflags(write=False)
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "curvature", curvature)
        object.__setattr__(self, "valid_mask", valid)

    @property
    def size(self) -> int:
        return int(self.normals.shape[0])

    @property
    def valid_count(self) -> int:
        return int(np.count_nonzero(self.valid_mask))


def _sorted_limited_neighbors(
    points: np.ndarray,
    center: np.ndarray,
    indices: np.ndarray,
    max_neighbors: int,
) -> np.ndarray:
    distances = np.linalg.norm(points[indices] - center, axis=1)
    order = np.argsort(distances, kind="mergesort")
    return indices[order[:max_neighbors]]


def _pca_normal(neighbours: np.ndarray) -> tuple[np.ndarray, float]:
    centered = neighbours - neighbours.mean(axis=0)
    covariance = centered.T @ centered / max(float(neighbours.shape[0] - 1), 1.0)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    normal = eigenvectors[:, 0]
    norm = float(np.linalg.norm(normal))
    if norm < 1e-12 or not np.isfinite(norm):
        raise ValueError("degenerate local neighbourhood")
    total = float(np.sum(np.maximum(eigenvalues, 0.0)))
    curvature = float(eigenvalues[0] / total) if total > 1e-12 else 0.0
    return normal / norm, max(0.0, curvature)


def estimate_surface_normals(
    points_mm: np.ndarray,
    config: NormalEstimationConfig | None = None,
    **overrides,
) -> SurfaceNormals:
    """Estimate oriented surface normals for ``points_mm``.

    ``overrides`` may be used for ergonomic one-off calls, e.g.
    ``estimate_surface_normals(points, radius_mm=25.0)``.
    """
    if config is not None and overrides:
        raise ValueError("pass either config or keyword overrides, not both")
    cfg = config or NormalEstimationConfig(**overrides)
    points = as_points_nx3(points_mm)
    n_points = points.shape[0]
    normals = np.zeros((n_points, 3), dtype=np.float32)
    confidence = np.zeros((n_points,), dtype=np.float32)
    curvature = np.full((n_points,), np.inf, dtype=np.float32)
    valid = np.zeros((n_points,), dtype=bool)
    if n_points == 0:
        return SurfaceNormals(normals, confidence, curvature, valid)

    camera = np.asarray(cfg.camera_position_mm, dtype=np.float64)
    index = RadiusIndex(points)

    for i, point in enumerate(points):
        neighbours_idx = index.query_radius(point, cfg.radius_mm)
        if neighbours_idx.size < cfg.min_neighbors:
            continue
        neighbours_idx = _sorted_limited_neighbors(
            points, point, neighbours_idx, cfg.max_neighbors
        )
        if neighbours_idx.size < cfg.min_neighbors:
            continue
        try:
            normal, local_curvature = _pca_normal(points[neighbours_idx])
        except ValueError:
            continue
        if cfg.max_curvature is not None and local_curvature > cfg.max_curvature:
            continue
        if cfg.orient_towards_camera:
            to_camera = camera - point
            if float(np.dot(normal, to_camera)) < 0.0:
                normal = -normal
        planarity = 1.0 - float(np.clip(local_curvature, 0.0, 1.0))
        support = min(1.0, neighbours_idx.size / max(float(cfg.min_neighbors * 2), 1.0))
        normals[i] = normal.astype(np.float32)
        confidence[i] = np.float32(planarity * support)
        curvature[i] = np.float32(local_curvature)
        valid[i] = True

    return SurfaceNormals(normals, confidence, curvature, valid)
