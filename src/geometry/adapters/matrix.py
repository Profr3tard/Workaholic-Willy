"""Canonical matrix conversion adapters.

Matrix conversion is intentionally centralised here so domain logic can
stay on typed ``Transform`` / ``Pose`` / ``Extrinsics`` values while
legacy boundaries keep receiving homogeneous matrices.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.geometry.matrix import position_quaternion_to_matrix
from src.geometry.validation import validate_homogeneous_matrix

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.calibration.extrinsics import Extrinsics
    from src.geometry.pose import Pose
    from src.geometry.transform import Transform

__all__ = [
    "extrinsics_to_matrix",
    "pose_to_matrix",
    "transform_to_matrix",
]


def transform_to_matrix(transform: Transform) -> np.ndarray:
    """Return a fresh validated 4x4 homogeneous matrix (mm)."""
    return validate_homogeneous_matrix(
        position_quaternion_to_matrix(transform.translation_mm, transform.quaternion_xyzw)
    )


def pose_to_matrix(pose: Pose) -> np.ndarray:
    """Return a fresh validated 4x4 homogeneous matrix (mm)."""
    return validate_homogeneous_matrix(
        position_quaternion_to_matrix(pose.position_mm, pose.quaternion_xyzw)
    )


def extrinsics_to_matrix(extrinsics: Extrinsics) -> np.ndarray:
    """Return ``extrinsics.transform`` as a fresh validated 4x4 matrix."""
    return transform_to_matrix(extrinsics.transform)
