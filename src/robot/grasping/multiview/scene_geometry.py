"""Fuse associated multi-view object surfaces into BASE-frame candidate clouds.

Consumes cross-camera associations to build one cloud per candidate object,
combining surfaces that a single depth view cannot observe. This improves grasp
coverage by exposing additional contact faces and can optionally return
neighbour geometry for collision filtering.

Scope is deliberately narrow: pure functions over resolved masks, depth,
intrinsics, and CAMERA->BASE transforms. Camera/frame resolution, live inputs,
missing-camera policy, and feature gating remain with the orchestrator.
All geometry uses BASE-frame millimetres.
"""


from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.robot.grasping.constants import (
    SCENE_GEOMETRY_LOG_FILE,
    create_grasping_logger,
)
from src.robot.grasping.geometry.pointcloud import masked_points
from src.robot.grasping.multiview.association import (
    DEFAULT_MIN_SCORE,
    DEFAULT_NEIGHBOUR_MM,
    AssociationMetric,
    SceneAssociation,
    ViewCandidates,
    fuse_scene_clouds,
)

# Logging for this module.
logger = create_grasping_logger("SceneGeometry", SCENE_GEOMETRY_LOG_FILE)

__all__ = [
    "FusedSceneGeometry",
    "ObservedView",
    "fuse_scene_geometry",
    "to_base_mm",
]


@dataclass(frozen=True, slots=True)
class ObservedView:
    """One camera's segmented objects plus everything needed to place them in BASE."""

    name: str
    masks: tuple[np.ndarray, ...]
    depth_map: np.ndarray
    intrinsics: np.ndarray
    camera_to_base: np.ndarray


@dataclass(frozen=True, slots=True)
class FusedSceneGeometry:
    """One fused cloud per primary object, plus what it was built from."""

    clouds_base_mm: tuple[np.ndarray | None, ...]
    views_used: tuple[str, ...]
    associations: tuple[SceneAssociation, ...]
    neighbour_clouds_base_mm: tuple[np.ndarray | None, ...] = ()

    @property
    def objects_fused(self) -> int:
        return sum(1 for cloud in self.clouds_base_mm if cloud is not None)

    @property
    def neighbour_points(self) -> int:
        return sum(len(cloud) for cloud in self.neighbour_clouds_base_mm if cloud is not None)

    def cloud_for(self, index: int) -> np.ndarray | None:
        """Object ``index``'s fused surface, or ``None`` bounds-safe by design."""

        if 0 <= index < len(self.clouds_base_mm):
            return self.clouds_base_mm[index]
        return None

    def neighbour_for(self, index: int) -> np.ndarray | None:
        if 0 <= index < len(self.neighbour_clouds_base_mm):
            return self.neighbour_clouds_base_mm[index]
        return None


def to_base_mm(
    mask: np.ndarray,
    depth_map: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_base: np.ndarray,
) -> np.ndarray:
    """One segmented object's surface in BASE millimetres, ``(N, 3)``."""

    points_cam = np.asarray(masked_points(mask, depth_map, intrinsics), dtype=np.float64)
    if points_cam.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    matrix = np.asarray(camera_to_base, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"camera_to_base must be shape (4, 4), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("camera_to_base must contain only finite values")
    return points_cam @ matrix[:3, :3].T + matrix[:3, 3]


def _neighbour_clouds(
    views: Sequence[ViewCandidates],
    associations: Sequence[SceneAssociation],
    primary_count: int,
    voxel_mm: float,
) -> tuple[np.ndarray | None, ...]:
    """Build per-object neighbour clouds from other-camera observations.

    For each primary object, include surfaces seen by other cameras while excluding
    the assigned match and any candidate with non-zero overlap as the target's best
    match. This asymmetry avoids both contaminating the target cloud and treating
    the target itself as a collision obstacle.

    Unmatched objects remain included as potential hidden colliders. Data from the
    primary view is excluded because the calculator already supplies those
    obstacles from ``other_object_masks``.
    """

    from src.robot.grasping.geometry.sampling import voxel_downsample_indices

    # One flat list over (view, candidate) so the per-object exclusion below is a boolean mask on a
    # single array rather than a vstack per object.
    clouds: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    flat_id: dict[tuple[int, int], int] = {}
    for view_index, view in enumerate(views):
        for candidate_index, cloud in enumerate(view.clouds_base_mm):
            points = np.asarray(cloud, dtype=np.float64).reshape(-1, 3)
            if not points.size:
                continue
            if voxel_mm > 0.0:
                points = points[voxel_downsample_indices(points, voxel_mm)]
            flat_id[(view_index, candidate_index)] = len(clouds)
            owners.append(np.full(len(points), len(clouds), dtype=np.int64))
            clouds.append(points)

    if not clouds:
        return tuple(None for _ in range(primary_count))

    all_points = np.vstack(clouds)
    owner = np.concatenate(owners)

    result: list[np.ndarray | None] = []
    for index in range(primary_count):
        excluded: set[int] = set()
        for view_index, association in enumerate(associations):
            row = association.scores[index] if index < len(association.scores) else ()
            assigned = association.assignment[index] if index < len(association.assignment) else None
            if assigned is not None:
                excluded.add(flat_id.get((view_index, assigned), -1))
            if row:
                best = int(np.argmax(np.asarray(row, dtype=np.float64)))
                if row[best] > 0.0:
                    excluded.add(flat_id.get((view_index, best), -1))
        excluded.discard(-1)
        keep = ~np.isin(owner, sorted(excluded)) if excluded else np.ones(len(owner), dtype=bool)
        points = all_points[keep]
        result.append(points if points.size else None)
    return tuple(result)


def fuse_scene_geometry(
    primary_masks: Sequence[np.ndarray],
    primary_depth_map: np.ndarray,
    primary_intrinsics: np.ndarray,
    primary_camera_to_base: np.ndarray,
    other_views: Sequence[ObservedView],
    *,
    metric: AssociationMetric = AssociationMetric.OVERLAP,
    min_score: float = DEFAULT_MIN_SCORE,
    neighbour_mm: float = DEFAULT_NEIGHBOUR_MM,
    max_centroid_mm: float = 150.0,
    with_neighbours: bool = False,
    neighbour_voxel_mm: float = 8.0,
    score_voxel_mm: float = 0.0,
) -> FusedSceneGeometry:
    """Fuse every primary object with whatever the other cameras can confirm about it."""

    primary_clouds = [
        to_base_mm(mask, primary_depth_map, primary_intrinsics, primary_camera_to_base)
        for mask in primary_masks
    ]
    if not primary_clouds:
        logger.debug("Fusion skipped: the primary view segmented nothing")
        return FusedSceneGeometry((), (), ())

    candidates: list[ViewCandidates] = []
    used: list[str] = []
    for view in other_views:
        clouds = tuple(
            cloud
            for cloud in (
                to_base_mm(mask, view.depth_map, view.intrinsics, view.camera_to_base)
                for mask in view.masks
            )
            if cloud.size
        )
        if not clouds:
            continue
        candidates.append(ViewCandidates(view.name, clouds))
        used.append(view.name)

    if not candidates:
        # Not an error (a one-camera cell is a legitimate deployment), but it means
        # every grasp this pick is synthesised from one side of the object only.
        logger.warning(
            "No other view contributed to fusion (%d offered, %d primary object(s)); "
            "single-view geometry stands",
            len(other_views),
            len(primary_clouds),
        )
        return FusedSceneGeometry(tuple(None for _ in primary_clouds), (), ())

    fused, associations = fuse_scene_clouds(
        primary_clouds,
        candidates,
        metric=metric,
        min_score=min_score,
        neighbour_mm=neighbour_mm,
        max_centroid_mm=max_centroid_mm,
        score_voxel_mm=score_voxel_mm,
    )
    # An object nothing confirmed comes back from fuse_scene_clouds as its own cloud unchanged.
    gained = [
        any(association.assignment[index] is not None for association in associations)
        for index in range(len(primary_clouds))
    ]
    fused_or_none: tuple[np.ndarray | None, ...] = tuple(
        cloud if gained[index] else None for index, cloud in enumerate(fused)
    )
    neighbours = (
        _neighbour_clouds(candidates, associations, len(primary_clouds), neighbour_voxel_mm)
        if with_neighbours
        else ()
    )
    result = FusedSceneGeometry(fused_or_none, tuple(used), associations, neighbours)
    logger.info(
        "Fused %d/%d object(s) from view(s) %s (metric=%s, neighbours=%s, %d neighbour point(s))",
        result.objects_fused,
        len(primary_clouds),
        ", ".join(used),
        metric.value if hasattr(metric, "value") else metric,
        with_neighbours,
        result.neighbour_points,
    )
    return result
