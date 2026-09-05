"""Exceptions raised by the :mod:`src.geometry` package.

Every public error derives from :class:`GeometryError`, so one
``except GeometryError`` clause catches the whole subsystem.
"""

from __future__ import annotations


class GeometryError(Exception):
    """Base class for every error raised by :mod:`src.geometry`."""


class InvalidQuaternionError(GeometryError):
    """A quaternion is not a unit-length, finite, four-element vector."""


class InvalidPoseError(GeometryError):
    """A :class:`~src.geometry.pose.Pose` failed validation.

    :mod:`src.geometry.validation` also raises it for a malformed vector at
    the layer below, where no more specific type applies.
    """


class InvalidTransformError(GeometryError):
    """A :class:`~src.geometry.transform.Transform` failed validation.

    ``compose`` and ``apply_pose`` also raise it when the frames do not join.
    """


class FrameMismatchError(GeometryError):
    """A geometric operation was attempted across incompatible frames.

    Comparing two poses that do not share a frame raises it. The transform
    side of the same problem, composing across a mismatched middle frame or
    applying a transform to a pose outside its ``from_frame``, raises
    :class:`InvalidTransformError` instead.
    """


class InvalidMatrixError(GeometryError):
    """A 4x4 or 3x3 matrix is not a proper rigid or rotation matrix."""
