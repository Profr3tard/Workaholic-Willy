"""Small pure bridges the pick loop uses to adapt its inputs."""

from __future__ import annotations

import numpy as np

from src.geometry import Frame
from src.robot.grasping.planning.grasp_pose import GraspPose
from src.robot.grasping.types.feedback import GraspResult
from src.robot.grasping.types.grasp_point import GraspPoint
from src.robot.grasping.types.perception import SegmentationLike
from src.robot.grasping.loop.target_selector import TargetCandidate


def _grasp_point_to_grasp_pose(grasp: GraspPoint) -> GraspPose:
    """Convert a ``GraspPoint`` into a ``GraspPose`` for swept-volume validation.

    Builds the gripper rotation using the ``[closing, binormal, approach]``
    convention and derives the two contacts from the grip width and closing
    axis. Raises ``ValueError`` for degenerate axes.
    """
    closing = np.asarray(grasp.axis, dtype=np.float64)
    approach = np.asarray(grasp.approach, dtype=np.float64)
    binormal = np.cross(approach, closing)
    binormal_norm = float(np.linalg.norm(binormal))
    if binormal_norm < 1e-9:
        raise ValueError("degenerate grasp axes: approach is parallel to the closing axis")
    binormal /= binormal_norm
    rotation = np.column_stack([closing, binormal, approach])
    position = np.asarray(grasp.position, dtype=np.float64)
    half = 0.5 * float(grasp.grip_width_mm)
    score = max(0.0, min(1.0, float(grasp.score)))
    return GraspPose(
        position_mm=position,
        rotation_matrix=rotation,
        grip_width_mm=float(grasp.grip_width_mm),
        score=score,
        confidence=score,
        contacts=(position - closing * half, position + closing * half),
        frame=Frame(grasp.frame.value),
    )


def _build_target_candidates(
    successes: list[tuple[int, GraspResult]],
    depth_map: np.ndarray,
    segmentations: tuple[SegmentationLike, ...],
) -> tuple[TargetCandidate, ...]:
    """Materialise :class:`TargetCandidate` records from per-segmentation successes.

    The centroid depth is the median depth over the segmentation mask (robust to outliers and missing
    ``NaN`` / ``0`` values).
    """

    out: list[TargetCandidate] = []
    for seg_idx, result in successes:
        seg = segmentations[seg_idx]
        mask_raw = getattr(seg, "mask", None)
        if mask_raw is None:
            continue
        mask = np.asarray(mask_raw).astype(bool, copy=False)
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        depth_vals = np.asarray(depth_map, dtype=np.float64)[mask]
        finite = depth_vals[np.isfinite(depth_vals) & (depth_vals > 0.0)]
        centroid_depth = float(np.median(finite)) if finite.size else float("inf")
        out.append(
            TargetCandidate(
                segmentation_index=seg_idx,
                mask=mask,
                centroid_depth_mm=centroid_depth,
                local_score=float(result.top_score),
                bbox_px=bbox,
            )
        )
    return tuple(out)
