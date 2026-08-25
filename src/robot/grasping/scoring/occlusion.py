"""Occlusion and approach-clearance metrics for clutter-aware grasping.

Computes target-hull occlusion by neighbouring masks and unobstructed
approach distance from a 3D point through the depth map. Both metrics are
pure, deterministic, non-mutating functions and preserve the calculator's
existing candidate-list and reason-code contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.robot.grasping.geometry import CameraIntrinsics

__all__ = ["approach_clearance_mm", "occlusion_ratio"]


def _convex_hull_mask(mask: np.ndarray) -> np.ndarray:
    """
    Return the convex hull of ``mask`` as a same-shape boolean image
    (falls back to the raw mask when cv2 is unavailable or there are too few points).
    """
    if not mask.any():
        return mask.astype(bool, copy=False)
    try:
        import cv2 as cv
    except ImportError:  # pragma: no cover 
        return mask.astype(bool, copy=False)
    bin_mask = (mask.astype(np.uint8) > 0).astype(np.uint8) * 255
    points = np.column_stack(np.nonzero(bin_mask)[::-1])  # (x, y)
    if points.shape[0] < 3:
        return mask.astype(bool, copy=False)
    hull = cv.convexHull(points.reshape(-1, 1, 2))
    out = np.zeros_like(bin_mask)
    cv.fillConvexPoly(out, hull, 255)
    return out.astype(bool)


def occlusion_ratio(
    target_mask: np.ndarray,
    other_masks: list[np.ndarray] | tuple[np.ndarray, ...] | None,
) -> float:
    """
    Return the convex-hull occlusion ratio of ``target_mask`` in ``[0, 1]``.
    Target-hull pixels covered by the union of ``other_masks`` over the target hull area;
    ``0.0`` when the hull is empty or no neighbours are supplied.
    """
    target = np.asarray(target_mask).astype(bool, copy=False)
    hull = _convex_hull_mask(target)
    hull_area = int(hull.sum())
    if hull_area == 0:
        return 0.0
    if not other_masks:
        return 0.0
    H, W = hull.shape
    union = np.zeros((H, W), dtype=bool)
    for other in other_masks:
        arr = np.asarray(other)
        if arr.shape != (H, W):
            raise ValueError(
                "occlusion_ratio: other mask shape "
                f"{arr.shape} does not match target shape {(H, W)}"
            )
        union |= arr.astype(bool, copy=False)
    # Only neighbour pixels that overlap the target hull count as
    # "occluding"; pixels outside the hull are unrelated objects.
    covered = int(np.logical_and(hull, union).sum())
    ratio = covered / float(hull_area)
    return float(np.clip(ratio, 0.0, 1.0))


def approach_clearance_mm(
    point_3d_cam_mm: np.ndarray,
    approach_axis_cam: np.ndarray,
    depth_map: np.ndarray,
    intrinsics: "CameraIntrinsics",
    *,
    max_distance_mm: float = 200.0,
    step_mm: float = 5.0,
    scale_to_mm: float = 1.0,
    clearance_tolerance_mm: float = 5.0,
) -> float:
    """
    Return the unobstructed approach distance in millimetres by ray-marching
    from ``point_3d_cam_mm`` back along ``-approach_axis_cam`` (toward the camera,
    the path the gripper traverses); stops at the first depth surface closer than
    the ray's ``z`` by more than ``clearance_tolerance_mm``
    (which guards single-pixel depth noise), else returns ``max_distance_mm``.
    """
    if not np.isfinite(max_distance_mm) or max_distance_mm <= 0.0:
        raise ValueError("max_distance_mm must be finite and > 0")
    if not np.isfinite(step_mm) or step_mm <= 0.0:
        raise ValueError("step_mm must be finite and > 0")

    start = np.asarray(point_3d_cam_mm, dtype=np.float64).reshape(3)
    axis = np.asarray(approach_axis_cam, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(axis))
    if n < 1e-9:
        raise ValueError("approach_axis_cam cannot be the zero vector")
    axis = axis / n

    depth = np.asarray(depth_map, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("depth_map must be 2D")
    H, W = depth.shape

    travelled = 0.0
    steps = int(np.ceil(max_distance_mm / step_mm))
    for k in range(1, steps + 1):
        distance = min(k * step_mm, max_distance_mm)
        # Step *toward* the camera (negative approach direction).
        sample = start - axis * distance
        z = float(sample[2])
        if z <= 0.0:
            return float(travelled)
        u = sample[0] * intrinsics.fx / z + intrinsics.cx
        v = sample[1] * intrinsics.fy / z + intrinsics.cy
        iu, iv = int(round(u)), int(round(v))
        if not (0 <= iu < W and 0 <= iv < H):
            return float(min(distance, max_distance_mm))
        depth_mm = float(depth[iv, iu]) * float(scale_to_mm)
        if np.isfinite(depth_mm) and depth_mm > 0.0:
            if depth_mm + clearance_tolerance_mm < z:
                # Surface in the depth map sits in front of the ray
                # -> the gripper would collide before reaching here.
                return float(travelled)
        travelled = float(distance)
    return float(min(max_distance_mm, travelled))
