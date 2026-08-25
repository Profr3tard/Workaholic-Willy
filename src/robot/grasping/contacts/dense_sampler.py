"""Dense surface sampling for cluttered-bin grasp planning.

Builds graspability-weighted interior point clouds from segmentation masks
and depth while rejecting boundaries and depth discontinuities. Also
aggregates neighbouring objects into a scene collision cloud. Pure NumPy
with no dependency on the perception model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.robot.grasping.geometry import (
    CameraIntrinsics,
    CloudOutlierConfig,
    SurfaceNormals,
    apply_cloud_outlier_filter,
    as_mask_and_depth,
    depth_unit_to_mm,
    estimate_surface_normals,
    masked_point_cloud,
)
from src.robot.grasping.geometry.sampling import (
    farthest_point_sample_indices,
)

__all__ = [
    "SurfaceSamples",
    "dense_surface_samples",
    "scene_collision_cloud",
]


@dataclass(frozen=True, slots=True)
class SurfaceSamples:
    """Graspability-weighted dense sampling of a segmented object.

    The per-point ``graspability`` ``[0, 1]`` weight doubles as the antipodal
    ``valid_mask``; ``pixels_yx`` maps each point back to the image.
    """

    points_mm: np.ndarray
    normals: SurfaceNormals
    graspability: np.ndarray
    pixels_yx: np.ndarray

    @property
    def size(self) -> int:
        return int(self.points_mm.shape[0])

    @property
    def is_empty(self) -> bool:
        return self.size == 0


def _binary_erode_3x3(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Erode ``mask`` ``iterations`` times with a 3x3 element."""
    if iterations < 0:
        raise ValueError("iterations must be >= 0")
    out = np.asarray(mask, dtype=bool)
    for _ in range(iterations):
        eroded = np.zeros_like(out)
        eroded[1:-1, 1:-1] = (
            out[1:-1, 1:-1]
            & out[:-2, 1:-1]
            & out[2:, 1:-1]
            & out[1:-1, :-2]
            & out[1:-1, 2:]
            & out[:-2, :-2]
            & out[:-2, 2:]
            & out[2:, :-2]
            & out[2:, 2:]
        )
        out = eroded
    return out


def _depth_discontinuity_mask(
    depth_mm: np.ndarray,
    *,
    threshold_mm: float,
) -> np.ndarray:
    """
    Pixels to *exclude*: a depth step above ``threshold_mm`` to a 4-connected neighbour
    marks an occlusion edge where normals are unreliable.
    """
    if threshold_mm <= 0.0:
        return np.zeros_like(depth_mm, dtype=bool)
    depth = np.asarray(depth_mm, dtype=np.float64)
    finite = np.isfinite(depth) & (depth > 0.0)
    safe = np.where(finite, depth, np.nan)
    dy = np.abs(np.diff(safe, axis=0, prepend=safe[:1]))
    dy = np.fmax(dy, np.abs(np.diff(safe, axis=0, append=safe[-1:])))
    dx = np.abs(np.diff(safe, axis=1, prepend=safe[:, :1]))
    dx = np.fmax(dx, np.abs(np.diff(safe, axis=1, append=safe[:, -1:])))
    grad = np.fmax(np.nan_to_num(dy, nan=0.0), np.nan_to_num(dx, nan=0.0))
    return grad > threshold_mm


def dense_surface_samples(
    mask: np.ndarray,
    depth_map: np.ndarray,
    intrinsics: CameraIntrinsics | np.ndarray,
    *,
    unit: str = "mm",
    min_depth_mm: float = 1.0,
    max_depth_mm: float | None = None,
    edge_erosion_px: int = 2,
    depth_discontinuity_mm: float = 8.0,
    voxel_size_mm: float | None = 3.0,
    max_points: int = 4096,
    normal_radius_mm: float = 12.0,
    fps: bool = True,
    curvature_penalty: float = 6.0,
    cloud_outlier_filter: CloudOutlierConfig | None = None,
) -> SurfaceSamples:
    """Build a graspability-weighted dense sampling of one object's surface."""
    if edge_erosion_px < 0:
        raise ValueError("edge_erosion_px must be >= 0")
    if max_points < 1:
        raise ValueError("max_points must be >= 1")
    if normal_radius_mm <= 0.0:
        raise ValueError("normal_radius_mm must be > 0")
    if curvature_penalty < 0.0:
        raise ValueError("curvature_penalty must be >= 0")

    mask_bool, depth = as_mask_and_depth(mask, depth_map)

    # Step 1+2: shrink mask and remove depth discontinuities.
    interior = _binary_erode_3x3(mask_bool, iterations=int(edge_erosion_px))
    discontinuities = _depth_discontinuity_mask(
        depth * depth_unit_to_mm(unit),
        threshold_mm=depth_discontinuity_mm,
    )
    interior &= ~discontinuities
    if not np.any(interior):
        return _empty_samples(intrinsics)

    # Step 3: back-project with optional voxel downsampling.
    cloud = masked_point_cloud(
        interior,
        depth,
        intrinsics,
        unit=unit,
        min_depth_mm=min_depth_mm,
        max_depth_mm=max_depth_mm,
        voxel_size_mm=voxel_size_mm,
    )
    if cloud.is_empty:
        return _empty_samples(intrinsics)

    points = np.asarray(cloud.points_mm, dtype=np.float64)
    pixels = np.asarray(cloud.pixels_yx, dtype=np.int32)

    # Optional outlier filter BEFORE FPS, so the FPS budget is not spent spreading across speckle.
    if cloud_outlier_filter is not None and points.shape[0] > 0:
        keep = apply_cloud_outlier_filter(points, cloud_outlier_filter)
        points = points[keep]
        pixels = pixels[keep]
        if points.shape[0] == 0:
            return _empty_samples(intrinsics)

    # Step 4: FPS downsampling for spatial diversity.
    if fps and points.shape[0] > max_points:
        keep = farthest_point_sample_indices(points, num_samples=max_points)
        points = points[keep]
        pixels = pixels[keep]
    elif points.shape[0] > max_points:
        stride = points.shape[0] // max_points
        points = points[::stride][:max_points]
        pixels = pixels[::stride][:max_points]

    # Step 5: normals + graspability weight.
    normals = estimate_surface_normals(
        points,
        radius_mm=float(normal_radius_mm),
        min_neighbors=6,
        max_neighbors=64,
    )
    confidence = np.asarray(normals.confidence, dtype=np.float32)
    curvature = np.asarray(normals.curvature, dtype=np.float32)
    valid = np.asarray(normals.valid_mask, dtype=bool)
    curvature_clean = np.nan_to_num(
        curvature.astype(np.float64, copy=False),
        nan=1.0,
        posinf=1.0,
        neginf=1.0,
    )
    curvature_factor = np.clip(
        1.0 - float(curvature_penalty) * curvature_clean,
        0.0,
        1.0,
    ).astype(np.float32)
    graspability = np.where(
        valid,
        np.clip(confidence, 0.0, 1.0) * curvature_factor,
        0.0,
    ).astype(np.float32)

    return SurfaceSamples(
        points_mm=points,
        normals=normals,
        graspability=graspability,
        pixels_yx=pixels,
    )


def scene_collision_cloud(
    object_masks: list[np.ndarray],
    depth_map: np.ndarray,
    intrinsics: CameraIntrinsics | np.ndarray,
    *,
    target_index: int | None = None,
    extra_scene_mask: np.ndarray | None = None,
    unit: str = "mm",
    min_depth_mm: float = 1.0,
    max_depth_mm: float | None = None,
    voxel_size_mm: float | None = 4.0,
) -> np.ndarray:
    """Return an Nx3 mm camera-frame collision cloud of every object except ``target_index``."""
    if not object_masks and extra_scene_mask is None:
        return np.empty((0, 3), dtype=np.float64)

    combined = (
        np.zeros_like(depth_map, dtype=bool)
        if not object_masks
        else np.zeros(object_masks[0].shape, dtype=bool)
    )
    for index, mask in enumerate(object_masks):
        if index == target_index:
            continue
        combined |= np.asarray(mask).astype(bool)
    if extra_scene_mask is not None:
        combined |= np.asarray(extra_scene_mask).astype(bool)
    if not np.any(combined):
        return np.empty((0, 3), dtype=np.float64)

    cloud = masked_point_cloud(
        combined,
        depth_map,
        intrinsics,
        unit=unit,
        min_depth_mm=min_depth_mm,
        max_depth_mm=max_depth_mm,
        voxel_size_mm=voxel_size_mm,
    )
    return np.asarray(cloud.points_mm, dtype=np.float64)


def _empty_samples(intrinsics: CameraIntrinsics | np.ndarray) -> SurfaceSamples:
    # ``intrinsics`` is accepted for call-site symmetry but not needed for an empty result.
    return SurfaceSamples(
        points_mm=np.empty((0, 3), dtype=np.float64),
        normals=SurfaceNormals(
            normals=np.empty((0, 3), dtype=np.float32),
            confidence=np.empty((0,), dtype=np.float32),
            curvature=np.empty((0,), dtype=np.float32),
            valid_mask=np.empty((0,), dtype=bool),
        ),
        graspability=np.empty((0,), dtype=np.float32),
        pixels_yx=np.empty((0, 2), dtype=np.int32),
    )
