"""Rigid 4x4 transform validation for grasp-frame math."""

from __future__ import annotations

import numpy as np

__all__ = ["validate_transform"]

# Deliberately looser than src/geometry (which uses ~1e-6): a real camera->base calibration
# matrix produced by the sim / hand-eye routines carries more numerical slack than a freshly composed
# Pose, so this gate accepts orthonormality within 1e-4 and det within 1e-3 rather than rejecting
# otherwise-good extrinsics. Tighten these only alongside a re-validation of the calibration outputs.
_ORTHO_ATOL = 1e-4
_DET_TOL = 1e-3


def validate_transform(T: np.ndarray, *, name: str = "T") -> np.ndarray:
    """Return a validated 4x4 rigid transform as a float64 copy.

    Checks a proper rigid transform: orthonormal rotation block, ``det(R) ~ 1``, and a
    ``[0, 0, 0, 1]`` bottom row.
    """
    transform = np.asarray(T, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"{name} must be (4, 4), got {transform.shape}")
    if not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} must contain only finite values")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=_ORTHO_ATOL):
        raise ValueError(f"{name} rotation block is not orthonormal")
    determinant = float(np.linalg.det(rotation))
    if abs(determinant - 1.0) > _DET_TOL:
        raise ValueError(f"{name} rotation has det={determinant:.4f}, expected ~1.0")
    if not np.allclose(transform[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-9):
        raise ValueError(f"{name} bottom row must be [0, 0, 0, 1]")
    return transform.copy()
