"""
Workaholic-Willy geometry subsystem vendor-neutral, frame-safe pose / transform
primitives.

This package owns the canonical 3-D representations used across the
robotics stack:

* :class:`Frame`      - coordinate frames (enum).
* :class:`Pose`       - 6-DoF rigid pose (position + orientation + frame).
* :class:`Transform`  - typed rigid transform between two frames.

It explicitly does **not** know anything about robot vendors, OpenCV,
ROS, FastAPI or pipelines. See ``geometry_README.md`` for the full
ownership / non-ownership contract.

Numerics
--------
* Translation: **millimetres** (``float64``).
* Orientation: **unit XYZW quaternion** (``float64``, canonical sign).
* Angles: radians.
* All public ndarrays are read-only after construction.
"""

from __future__ import annotations

from .conversions import (
    axis_angle_to_quaternion_xyzw,
    matrix_to_transform,
    quaternion_xyzw_to_axis_angle,
    transform_to_matrix,
)
from .exceptions import (
    FrameMismatchError,
    GeometryError,
    InvalidMatrixError,
    InvalidPoseError,
    InvalidQuaternionError,
    InvalidTransformError,
)
from .frame import Frame
from .pose import Pose
from .quaternion import (
    IDENTITY_QUAT_XYZW,
    angle_between,
    canonicalise,
    conjugate,
    from_axis_angle,
    from_euler,
    from_rotation_matrix,
    matrix_to_rotation_vector,
    multiply,
    rotate_vector,
    rotation_vector_to_matrix,
    to_axis_angle,
    to_euler,
    to_rotation_matrix,
)
from .serialization import (
    POSE_SCHEMA,
    TRANSFORM_SCHEMA,
    pose_from_dict,
    pose_to_dict,
    transform_from_dict,
    transform_to_dict,
)
from .transform import Transform

__all__ = [
    # Quaternion utilities
    "IDENTITY_QUAT_XYZW",
    # Serialization
    "POSE_SCHEMA",
    "TRANSFORM_SCHEMA",
    # Core types
    "Frame",
    "FrameMismatchError",
    # Exceptions
    "GeometryError",
    "InvalidMatrixError",
    "InvalidPoseError",
    "InvalidQuaternionError",
    "InvalidTransformError",
    "Pose",
    "Transform",
    "angle_between",
    # Conversions
    "axis_angle_to_quaternion_xyzw",
    "canonicalise",
    "conjugate",
    "from_axis_angle",
    "from_euler",
    "from_rotation_matrix",
    "matrix_to_rotation_vector",
    "matrix_to_transform",
    "multiply",
    "pose_from_dict",
    "pose_to_dict",
    "quaternion_xyzw_to_axis_angle",
    "rotate_vector",
    "rotation_vector_to_matrix",
    "to_axis_angle",
    "to_euler",
    "to_rotation_matrix",
    "transform_from_dict",
    "transform_to_dict",
    "transform_to_matrix",
]
