"""JSON-friendly serialization for :class:`Pose` and :class:`Transform`.

Each dictionary carries a schema string, so a breaking change such as moving to
WXYZ ordering or storing covariance is rejected on read instead of being
misread as the current format. In the pose dictionary ``label`` is the only
optional key; every other key of either dictionary is required.

Pose::

    {
        "schema": "willy.geometry.pose/1",
        "frame": "base",
        "label": null,
        "position_mm": [x, y, z],
        "quaternion_xyzw": [x, y, z, w],
    }

Transform::

    {
        "schema": "willy.geometry.transform/1",
        "from_frame": "camera",
        "to_frame": "base",
        "translation_mm": [x, y, z],
        "quaternion_xyzw": [x, y, z, w],
    }
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .exceptions import InvalidPoseError, InvalidTransformError
from .frame import Frame
from .pose import Pose
from .transform import Transform

__all__ = [
    "POSE_SCHEMA",
    "TRANSFORM_SCHEMA",
    "pose_from_dict",
    "pose_to_dict",
    "transform_from_dict",
    "transform_to_dict",
]


POSE_SCHEMA: str = "willy.geometry.pose/1"
TRANSFORM_SCHEMA: str = "willy.geometry.transform/1"


# Pose


def pose_to_dict(pose: Pose) -> dict[str, Any]:
    """Serialise a :class:`Pose` to a JSON-friendly dict tagged ``POSE_SCHEMA``."""
    return {
        "schema": POSE_SCHEMA,
        "frame": pose.frame.value,
        "label": pose.label,
        "position_mm": pose.position_mm.tolist(),
        "quaternion_xyzw": pose.quaternion_xyzw.tolist(),
    }


def pose_from_dict(data: Mapping[str, Any]) -> Pose:
    """Deserialise a :class:`Pose` from :func:`pose_to_dict` output.

    Raises :class:`InvalidPoseError` on a schema mismatch, a missing key or a
    field value :class:`Pose` refuses.
    """
    schema = data.get("schema")
    if schema != POSE_SCHEMA:
        raise InvalidPoseError(
            f"pose_from_dict: expected schema {POSE_SCHEMA!r}, got {schema!r}"
        )
    try:
        return Pose(
            position_mm=np.asarray(data["position_mm"], dtype=np.float64),
            quaternion_xyzw=np.asarray(data["quaternion_xyzw"], dtype=np.float64),
            frame=Frame(data["frame"]),
            label=data.get("label"),
        )
    except KeyError as exc:
        raise InvalidPoseError(f"pose_from_dict: missing required key {exc}") from exc
    except ValueError as exc:
        raise InvalidPoseError(f"pose_from_dict: invalid field value: {exc}") from exc


# Transform


def transform_to_dict(t: Transform) -> dict[str, Any]:
    """Serialise a :class:`Transform` to a dict tagged ``TRANSFORM_SCHEMA``."""
    return {
        "schema": TRANSFORM_SCHEMA,
        "from_frame": t.from_frame.value,
        "to_frame": t.to_frame.value,
        "translation_mm": t.translation_mm.tolist(),
        "quaternion_xyzw": t.quaternion_xyzw.tolist(),
    }


def transform_from_dict(data: Mapping[str, Any]) -> Transform:
    """Deserialise a :class:`Transform` from :func:`transform_to_dict` output.

    Raises :class:`InvalidTransformError` on a schema mismatch, a missing key
    or a field value :class:`Transform` refuses.
    """
    schema = data.get("schema")
    if schema != TRANSFORM_SCHEMA:
        raise InvalidTransformError(
            f"transform_from_dict: expected schema {TRANSFORM_SCHEMA!r}, "
            f"got {schema!r}"
        )
    try:
        return Transform(
            translation_mm=np.asarray(data["translation_mm"], dtype=np.float64),
            quaternion_xyzw=np.asarray(data["quaternion_xyzw"], dtype=np.float64),
            from_frame=Frame(data["from_frame"]),
            to_frame=Frame(data["to_frame"]),
        )
    except KeyError as exc:
        raise InvalidTransformError(f"transform_from_dict: missing required key {exc}") from exc
    except ValueError as exc:
        raise InvalidTransformError(f"transform_from_dict: invalid field value: {exc}") from exc
