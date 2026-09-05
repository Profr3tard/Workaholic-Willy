"""Exceptions raised by the :mod:`src.geometry` package.

All public errors derive from :class:`GeometryError`, so a caller can catch the
whole subsystem with a single ``except GeometryError`` clause.
"""

from __future__ import annotations


class GeometryError(Exception):
    """Base class for every error raised by :mod:`src.geometry`."""


class InvalidQuaternionError(GeometryError):
    """A quaternion is not a unit-length, finite, four-element vector."""


class InvalidPoseError(GeometryError):
    """A :class:`~src.geometry.pose.Pose` failed validation."""


class InvalidTransformError(GeometryError):
    """A :class:`~src.geometry.transform.Transform` failed validation."""


class FrameMismatchError(GeometryError):
    """A geometric operation was attempted across incompatible frames.

    Raised when composing two transforms whose middle frame does not match, or
    when applying a transform to a pose that does not live in the transform's
    ``from_frame``.
    """


class InvalidMatrixError(GeometryError):
    """A 4x4 or 3x3 matrix is not a proper rigid or rotation matrix."""
