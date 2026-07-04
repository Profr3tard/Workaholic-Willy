"""Adapter layer for geometry interoperability boundaries."""

from .matrix import extrinsics_to_matrix, pose_to_matrix, transform_to_matrix

__all__ = [
    "extrinsics_to_matrix",
    "pose_to_matrix",
    "transform_to_matrix",
]