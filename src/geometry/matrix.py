"""4x4 homogeneous matrix helpers.

These exist for two reasons:

1. Bridging code that speaks plain ndarrays, such as eye-to-hand calibration
   and OpenCV. See also :mod:`src.geometry.conversions`.
2. Numerical efficiency in tight inner loops, where a single matrix multiply
   is cheaper than allocating a new :class:`Transform`.

For everything else prefer :class:`~src.geometry.transform.Transform`, which
carries explicit frames and is type-checked at every step.
"""

from __future__ import annotations

import numpy as np

from .quaternion import (
    from_rotation_matrix,
    to_rotation_matrix,
)
from .validation import (
    validate_homogeneous_matrix,
    validate_position_mm,
    validate_quaternion_xyzw,
    validate_rotation_matrix,
)

__all__ = [
    "IDENTITY_MATRIX",
    "compose_homogeneous",
    "invert_homogeneous",
    "make_homogeneous",
    "matrix_to_position_quaternion",
    "position_quaternion_to_matrix",
]


IDENTITY_MATRIX: np.ndarray = np.eye(4, dtype=np.float64)
IDENTITY_MATRIX.setflags(write=False)


def make_homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Assemble a 4x4 transform from a 3x3 rotation and a (3,) translation."""
    R = validate_rotation_matrix(R, name="R")
    t = validate_position_mm(t, name="t")
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R
    out[:3, 3] = t
    return out


def matrix_to_position_quaternion(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decompose a validated 4x4 transform into ``(translation_mm, quat_xyzw)``."""
    arr = validate_homogeneous_matrix(T)
    t = arr[:3, 3].astype(np.float64, copy=True)
    q = from_rotation_matrix(arr[:3, :3])
    return t, q


def position_quaternion_to_matrix(t: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Assemble a 4x4 transform from a (3,) translation and an XYZW quaternion."""
    t_v = validate_position_mm(t, name="translation_mm")
    q_v = validate_quaternion_xyzw(q)
    R = to_rotation_matrix(q_v)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R
    out[:3, 3] = t_v
    return out


def invert_homogeneous(T: np.ndarray) -> np.ndarray:
    """Return the inverse of a 4x4 rigid transform, analytically and without a solver."""
    arr = validate_homogeneous_matrix(T)
    R = arr[:3, :3]
    t = arr[:3, 3]
    inv = np.eye(4, dtype=np.float64)
    # For a rigid transform the inverse rotation is the transpose, and the
    # inverse translation is that rotation applied to the negated translation.
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ t
    return inv


def compose_homogeneous(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return ``A @ B`` after validating both as 4x4 rigid transforms."""
    a = validate_homogeneous_matrix(A, name="A")
    b = validate_homogeneous_matrix(B, name="B")
    return a @ b
