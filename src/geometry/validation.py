"""
Low-level validators used across the geometry package.

These helpers raise the typed exceptions from
:mod:`src.geometry.exceptions` so callers can ``except`` on a
single base class. They never silently coerce data, invalid input is
always reported.

Numerics contract
-----------------
* All vectors / matrices are coerced to ``float64``.
* Quaternions are XYZW order. A unit-length tolerance of ``1e-6`` is
  applied; out-of-tolerance vectors are rejected (the caller should
  call :func:`normalise_quaternion` first if normalisation is intended).
"""

from __future__ import annotations

import numpy as np

from .exceptions import (
    InvalidMatrixError,
    InvalidPoseError,
    InvalidQuaternionError,
    InvalidTransformError,
)

# Tolerances are deliberately tight; runtime data should already be clean.
QUATERNION_UNIT_TOL: float = 1e-6
ROTATION_DET_TOL: float = 1e-6
ROTATION_ORTHONORMAL_TOL: float = 1e-6


def as_float64_vector(value: object, expected_len: int, *, name: str) -> np.ndarray:
    """Coerce *value* to a finite ``float64`` ndarray of shape ``(expected_len,)``.

    Raises
    ------
    InvalidPoseError
        If the shape is wrong or any element is non-finite. The exception
        type is intentionally generic at this layer; higher-level
        ``Pose`` / ``Transform`` validators re-raise with their own
        types when needed.
    """
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (expected_len,):
        raise InvalidPoseError(
            f"{name} must have shape ({expected_len},), got shape {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise InvalidPoseError(f"{name} must be finite, got {arr.tolist()!r}")
    return arr


def validate_position_mm(value: object, *, name: str = "position_mm") -> np.ndarray:
    """Validate a (3,) translation in millimetres."""
    return as_float64_vector(value, 3, name=name)


def validate_quaternion_xyzw(
    value: object,
    *,
    name: str = "quaternion_xyzw",
    tol: float = QUATERNION_UNIT_TOL,
) -> np.ndarray:
    """Validate a (4,) XYZW quaternion that is unit-length within ``tol``.

    Use :func:`normalise_quaternion` if you want best-effort
    normalisation instead of strict validation.
    """
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (4,):
        raise InvalidQuaternionError(
            f"{name} must have shape (4,), got shape {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise InvalidQuaternionError(
            f"{name} must be finite, got {arr.tolist()!r}"
        )
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        raise InvalidQuaternionError(f"{name} is the zero vector")
    if abs(norm - 1.0) > tol:
        raise InvalidQuaternionError(
            f"{name} is not unit length (|q|={norm:.6f}, tol={tol:.0e}); "
            "call normalise_quaternion() first if normalisation is intended"
        )
    return arr


def normalise_quaternion(
    value: object,
    *,
    name: str = "quaternion_xyzw",
    canonicalise: bool = True,
) -> np.ndarray:
    """Normalise a (4,) XYZW quaternion to unit length.

    Parameters
    ----------
    canonicalise:
        When ``True`` (the default), flip the sign of the quaternion
        whenever its scalar component (``w`` = index 3) is negative.
        Quaternions ``q`` and ``-q`` represent the same rotation, so
        canonicalising makes equality and hashing well defined.
    """
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (4,):
        raise InvalidQuaternionError(
            f"{name} must have shape (4,), got shape {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise InvalidQuaternionError(
            f"{name} must be finite, got {arr.tolist()!r}"
        )
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        raise InvalidQuaternionError(f"{name} is the zero vector")
    out = arr / norm
    if canonicalise and out[3] < 0.0:
        out = -out
    return out


def validate_homogeneous_matrix(
    value: object,
    *,
    name: str = "matrix",
    det_tol: float = ROTATION_DET_TOL,
) -> np.ndarray:
    """Validate a 4x4 rigid transform matrix.

    Checks performed
    ----------------
    * shape is exactly ``(4, 4)``
    * all entries are finite
    * the rotation block has determinant ``≈ +1``
    * the bottom row is exactly ``[0, 0, 0, 1]`` (within ``det_tol``)
    """
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (4, 4):
        raise InvalidMatrixError(
            f"{name} must have shape (4, 4), got shape {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise InvalidMatrixError(f"{name} must be finite")

    bottom = arr[3, :]
    expected = np.array([0.0, 0.0, 0.0, 1.0])
    if not np.allclose(bottom, expected, atol=det_tol):
        raise InvalidMatrixError(
            f"{name} bottom row must be [0, 0, 0, 1], got {bottom.tolist()!r}"
        )

    validate_rotation_matrix(arr[:3, :3], name=f"{name} rotation", det_tol=det_tol)
    return arr


def validate_rotation_matrix(
    value: object,
    *,
    name: str = "rotation",
    det_tol: float = ROTATION_DET_TOL,
    orthonormal_tol: float = ROTATION_ORTHONORMAL_TOL,
) -> np.ndarray:
    """Validate a finite, orthonormal 3x3 rotation matrix with det +1."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3, 3):
        raise InvalidMatrixError(f"{name} must have shape (3, 3), got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise InvalidMatrixError(f"{name} must be finite")
    identity = np.eye(3, dtype=np.float64)
    if not np.allclose(arr.T @ arr, identity, atol=orthonormal_tol):
        raise InvalidMatrixError(f"{name} must be orthonormal")
    det = float(np.linalg.det(arr))
    if abs(det - 1.0) > det_tol:
        raise InvalidMatrixError(f"{name} determinant must be +1, got {det:.12g}")
    return arr


def validate_frames_compatible(
    a_to: object,
    b_from: object,
    *,
    op: str,
) -> None:
    """Helper used by Transform.compose / apply_pose.

    Raises
    ------
    InvalidTransformError
        Re-raised with a useful message if the chain is broken.
    """
    if a_to != b_from:
        raise InvalidTransformError(
            f"{op}: frame mismatch — left.to_frame={a_to!r} but "
            f"right.from_frame={b_from!r}"
        )


__all__ = [
    "QUATERNION_UNIT_TOL",
    "ROTATION_DET_TOL",
    "ROTATION_ORTHONORMAL_TOL",
    "as_float64_vector",
    "normalise_quaternion",
    "validate_frames_compatible",
    "validate_homogeneous_matrix",
    "validate_position_mm",
    "validate_quaternion_xyzw",
    "validate_rotation_matrix",
]
