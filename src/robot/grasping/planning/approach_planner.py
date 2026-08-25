"""Build straight-line approach and retreat geometry for parallel-jaw grasps.

Provides pre-grasp and retreat poses plus interpolated approach trajectories
from a ``GraspPose``. Pre-grasp offsets along the negative approach axis;
retreat defaults to ``+Z`` in the grasp frame and is caller-configurable.
"""


from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.geometry import Pose
from src.robot.grasping.planning.grasp_pose import GraspPose

__all__ = [
    "approach_waypoints",
    "pre_grasp_pose",
    "retreat_pose",
]


_DEFAULT_WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _validate_offset(offset_mm: float, name: str) -> float:
    value = float(offset_mm)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return value


def _ensure_grasp(pose: GraspPose) -> GraspPose:
    if not isinstance(pose, GraspPose):
        raise TypeError("pose must be a GraspPose")
    return pose


def _shift(position_mm: np.ndarray, axis: np.ndarray, distance_mm: float) -> np.ndarray:
    direction = np.asarray(axis, dtype=np.float64)
    if direction.shape != (3,):
        raise ValueError(f"axis must be shape (3,), got {direction.shape}")
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        raise ValueError("axis cannot be the zero vector")
    return np.asarray(position_mm, dtype=np.float64) + (direction / norm) * float(distance_mm)


def _shifted_grasp(grasp: GraspPose, position_mm: np.ndarray, *, label: str) -> GraspPose:
    """Return a GraspPose at ``position_mm`` keeping orientation/contacts."""
    metadata = {**grasp.metadata, "approach_role": label}
    return GraspPose(
        position_mm=position_mm.astype(np.float64),
        rotation_matrix=grasp.rotation_matrix.copy(),
        grip_width_mm=grasp.grip_width_mm,
        score=grasp.score,
        confidence=grasp.confidence,
        contacts=(grasp.contacts[0].copy(), grasp.contacts[1].copy()),
        frame=grasp.frame,
        metadata=metadata,
    )


def pre_grasp_pose(
    grasp: GraspPose,
    *,
    standoff_mm: float = 80.0,
) -> GraspPose:
    """Return a pose offset along ``-approach_axis`` by ``standoff_mm``."""
    pose = _ensure_grasp(grasp)
    distance = _validate_offset(standoff_mm, "standoff_mm")
    new_position = _shift(pose.position_mm, -pose.approach_axis, distance)
    return _shifted_grasp(pose, new_position, label="pre_grasp")


def retreat_pose(
    grasp: GraspPose,
    *,
    lift_mm: float = 100.0,
    direction: np.ndarray | Sequence[float] = _DEFAULT_WORLD_UP,
) -> GraspPose:
    """Return a pose offset by ``lift_mm`` along ``direction`` (unit vector)."""
    pose = _ensure_grasp(grasp)
    distance = _validate_offset(lift_mm, "lift_mm")
    new_position = _shift(pose.position_mm, np.asarray(direction, dtype=np.float64), distance)
    return _shifted_grasp(pose, new_position, label="retreat")


def approach_waypoints(
    grasp: GraspPose,
    *,
    standoff_mm: float = 80.0,
    num_waypoints: int = 4,
) -> list[Pose]:
    """Return ``num_waypoints`` poses linearly interpolating pre->grasp."""
    pose = _ensure_grasp(grasp)
    if num_waypoints < 2:
        raise ValueError("num_waypoints must be >= 2")
    distance = _validate_offset(standoff_mm, "standoff_mm")
    pre_position = _shift(pose.position_mm, -pose.approach_axis, distance)
    target = np.asarray(pose.position_mm, dtype=np.float64)
    fractions = np.linspace(0.0, 1.0, num=num_waypoints)
    waypoints: list[Pose] = []
    for index, fraction in enumerate(fractions):
        position = pre_position + (target - pre_position) * float(fraction)
        label = f"approach_{index:02d}"
        waypoints.append(
            Pose(
                position_mm=position,
                quaternion_xyzw=pose.quaternion_xyzw.copy(),
                frame=pose.frame,
                label=label,
            )
        )
    return waypoints
