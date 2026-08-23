"""Pure-Python conversion helpers between Willy types and Isaac shapes.

This module is the **explicit translation layer**.
It contains *only* helpers that operate on numeric arrays and Willy
types, it never imports Isaac itself.
The :class:`IsaacRobotArm` will call these helpers after it has
unpacked Isaac articulation / TCP state into raw numpy arrays.

Why a separate module?
----------------------
Keeping the conversion math out of ``arm.py`` makes the surface easy
to unit-test on any host (no Isaac required) and gives the camera /
calibration adapters a single place to reuse the
same pose / joints conventions.

Conventions
-----------
* Positions are millimetres throughout the Willy stack. Isaac is
  metres by default, so every position crossing the boundary passes
  through :func:`metres_to_millimetres` / :func:`millimetres_to_metres`.
* Quaternions use the XYZW convention on the Willy side and may use
  WXYZ on the Isaac side (USD prim attributes). :func:`isaac_wxyz_to_xyzw`
  and :func:`xyzw_to_isaac_wxyz` handle the swap explicitly so call
  sites cannot get the ordering wrong.
* Joint vectors are radians on both sides; the helpers therefore do
  not touch the values, only their dtype / shape.
"""

from __future__ import annotations

import numpy as np

from src.geometry import Frame, Pose
from src.robot.core import JointPositions

__all__ = [
    "isaac_pose_to_willy",
    "willy_pose_to_isaac",
    "isaac_joints_to_willy",
    "willy_joints_to_isaac",
    "metres_to_millimetres",
    "millimetres_to_metres",
    "isaac_wxyz_to_xyzw",
    "xyzw_to_isaac_wxyz",
    "isaac_rotmat_to_wxyz",
]


# ---------------------------------------------------------------------------
# Scalar / array unit conversions
# ---------------------------------------------------------------------------


def metres_to_millimetres(values: np.ndarray) -> np.ndarray:
    """Return ``values * 1000.0`` as a ``float64`` array."""
    return np.asarray(values, dtype=np.float64) * 1000.0


def millimetres_to_metres(values: np.ndarray) -> np.ndarray:
    """Return ``values / 1000.0`` as a ``float64`` array."""
    return np.asarray(values, dtype=np.float64) / 1000.0


# ---------------------------------------------------------------------------
# Quaternion convention swaps
# ---------------------------------------------------------------------------


def isaac_wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    """Reorder a quaternion from Isaac's ``[w, x, y, z]`` to Willy's ``[x, y, z, w]``."""
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def xyzw_to_isaac_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    """Reorder a quaternion from Willy's ``[x, y, z, w]`` to Isaac's ``[w, x, y, z]``."""
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def isaac_rotmat_to_wxyz(rotation_matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a normalised Isaac ``[w, x, y, z]`` quaternion."""
    R = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


# ---------------------------------------------------------------------------
# Pose / joints round-trips
# ---------------------------------------------------------------------------


def isaac_pose_to_willy(
    position_m: np.ndarray,
    orientation_wxyz: np.ndarray,
    *,
    label: str | None = None,
) -> Pose:
    """Build a :class:`Pose` (mm, XYZW, :attr:`Frame.BASE`) from Isaac state.

    Parameters
    ----------
    position_m
        ``(3,)`` array in metres as read from Isaac.
    orientation_wxyz
        ``(4,)`` quaternion in Isaac's ``[w, x, y, z]`` convention.
    label
        Optional human-readable label preserved on the :class:`Pose`.
    """
    return Pose(
        position_mm=metres_to_millimetres(position_m),
        quaternion_xyzw=isaac_wxyz_to_xyzw(orientation_wxyz),
        frame=Frame.BASE,
        label=label,
    )


def willy_pose_to_isaac(pose: Pose) -> tuple[np.ndarray, np.ndarray]:
    """Convert an Willy :class:`Pose` into ``(position_m, orientation_wxyz)``.

    Raises
    ------
    ValueError
        If ``pose.frame`` is not :attr:`Frame.BASE`. Isaac articulation
        commands are expressed in the simulator world / robot base
        frame; passing a tool-frame pose would silently mis-target.
    """
    if pose.frame is not Frame.BASE:
        raise ValueError(
            f"willy_pose_to_isaac requires Frame.BASE; got {pose.frame!r}."
        )
    return (
        millimetres_to_metres(pose.position_mm),
        xyzw_to_isaac_wxyz(pose.quaternion_xyzw),
    )


def isaac_joints_to_willy(joint_radians: np.ndarray) -> JointPositions:
    """Wrap an Isaac joint vector (radians) in a :class:`JointPositions`."""
    return JointPositions(np.asarray(joint_radians, dtype=np.float64))


def willy_joints_to_isaac(joints: JointPositions) -> np.ndarray:
    """Return the raw radian vector backing a :class:`JointPositions`."""
    return np.asarray(joints.values, dtype=np.float64)
