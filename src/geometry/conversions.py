"""
Bridge adapters between :class:`Pose` / :class:`Transform` and raw
numerical representations used elsewhere in Workaholic-Willy.

This module is the only place in the geometry package allowed to
know about cross-cutting numerical types (raw 4x4 ndarrays, axis-angle
tuples). It is **vendor-neutral**: vendor pose types (UR, KUKA, etc.)
live behind their driver packages and provide their own boundary
adapters there (see ``src.robot.drivers.ur.pose_adapter``).

Conversions are intentionally lossy in only one direction (mm <-> m, etc.)
and round-trip exactly otherwise.
"""

from __future__ import annotations

import numpy as np

from .adapters.matrix import transform_to_matrix as _transform_to_matrix
from .frame import Frame
from .quaternion import from_axis_angle, matrix_to_rotation_vector, rotation_vector_to_matrix, to_axis_angle
from .transform import Transform

__all__ = [
    "axis_angle_to_quaternion_xyzw",
    "matrix_to_transform",
    "matrix_to_rotation_vector",
    "quaternion_xyzw_to_axis_angle",
    "rotation_vector_to_matrix",
    "transform_to_matrix",
]


# ---------- Axis-angle <-> quaternion -------------------------------------


def axis_angle_to_quaternion_xyzw(rvec: np.ndarray) -> np.ndarray:
    """Convert a (3,) axis-angle vector (rad) to a canonical XYZW quaternion."""
    return from_axis_angle(rvec)


def quaternion_xyzw_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """Convert a unit XYZW quaternion to a (3,) axis-angle vector (rad)."""
    return to_axis_angle(q)


# ---------- 4x4 ndarray <-> Transform -------------------------------------


def matrix_to_transform(
    T: np.ndarray,
    *,
    from_frame: Frame,
    to_frame: Frame,
) -> Transform:
    """Wrap a validated 4x4 homogeneous matrix (mm) as a :class:`Transform`."""
    return Transform.from_matrix(T, from_frame=from_frame, to_frame=to_frame)


def transform_to_matrix(t: Transform) -> np.ndarray:
    """Return a fresh validated 4x4 homogeneous matrix (mm) for ``t``."""
    return _transform_to_matrix(t)
