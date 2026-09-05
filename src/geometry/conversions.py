"""Bridge adapters between :class:`Pose` or :class:`Transform` and the raw
numerical representations used elsewhere in the stack.

This module is the only place in the geometry package allowed to know about
cross-cutting numerical types such as raw 4x4 ndarrays and axis-angle tuples.
It is vendor neutral: vendor pose types live behind their driver packages and
provide their own boundary adapters there, for example
``src/robot/drivers/ur/pose_adapter.py``.

Conversions round-trip exactly.
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


# Axis-angle and quaternion


def axis_angle_to_quaternion_xyzw(rvec: np.ndarray) -> np.ndarray:
    """Convert a (3,) axis-angle vector (rad) to a canonical XYZW quaternion."""
    return from_axis_angle(rvec)


def quaternion_xyzw_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """Convert a unit XYZW quaternion to a (3,) axis-angle vector (rad)."""
    return to_axis_angle(q)


# 4x4 ndarray and Transform


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
