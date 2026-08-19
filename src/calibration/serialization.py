"""
JSON serialisation for :class:`Extrinsics`.

Wire format
-----------
::

    {
        "schema": "willy.calibration.extrinsics/1",
        "transform": { ... willy.geometry.transform/1 ... },
        "rmse_mm": 1.234,
        "max_error_mm": 2.5,
        "num_samples": 42,
        "captured_at": "2026-05-08T13:44:18+00:00",
        "rig_id": "rig-0",
        "quality": "excellent",
    }

The schema version is checked on load; mismatches raise
:class:`ExtrinsicsError` rather than silently coercing.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from src.geometry import Frame, GeometryError, Transform, transform_from_dict, transform_to_dict

from .exceptions import ExtrinsicsError
from .extrinsics import EXTRINSICS_SCHEMA, Extrinsics

#: Wire schema for a persisted eye-in-hand CAMERA->TOOL calibration transform. Kept separate from
#: :data:`EXTRINSICS_SCHEMA` (which is CAMERA->BASE-locked) so an EIH wrist camera has a TYPED artifact
#: (the multi-camera registry loads it into an EyeInHandFrameResolver), instead of the old bespoke dict.
CAM_TO_TOOL_SCHEMA: str = "willy.calibration.cam_to_tool/1"

__all__ = [
    "CAM_TO_TOOL_SCHEMA",
    "EXTRINSICS_SCHEMA",
    "extrinsics_from_dict",
    "extrinsics_to_dict",
    "load_cam_to_tool",
    "load_extrinsics",
    "save_cam_to_tool",
    "save_extrinsics",
]


def extrinsics_to_dict(ext: Extrinsics) -> dict[str, Any]:
    """Serialise :class:`Extrinsics` to a JSON-friendly dictionary."""
    return {
        "schema": EXTRINSICS_SCHEMA,
        "transform": transform_to_dict(ext.transform),
        "rmse_mm": float(ext.rmse_mm),
        "max_error_mm": float(ext.max_error_mm),
        "num_samples": int(ext.num_samples),
        "captured_at": ext.captured_at.isoformat(),
        "rig_id": ext.rig_id,
        "quality": ext.quality,
    }


def extrinsics_from_dict(data: Mapping[str, Any]) -> Extrinsics:
    """Deserialise :class:`Extrinsics` from :func:`extrinsics_to_dict` output."""
    schema = data.get("schema")
    if schema != EXTRINSICS_SCHEMA:
        raise ExtrinsicsError(
            f"extrinsics_from_dict: expected schema {EXTRINSICS_SCHEMA!r}, got {schema!r}"
        )

    required = ("transform", "rmse_mm", "max_error_mm", "num_samples", "captured_at", "rig_id")
    missing = [k for k in required if k not in data]
    if missing:
        raise ExtrinsicsError(f"extrinsics_from_dict: missing required keys {missing!r}")

    try:
        transform = transform_from_dict(data["transform"])
    except (GeometryError, TypeError, ValueError, KeyError) as exc:
        raise ExtrinsicsError("extrinsics_from_dict: invalid transform payload") from exc

    captured_raw = data["captured_at"]
    if not isinstance(captured_raw, str):
        raise ExtrinsicsError(
            f"extrinsics_from_dict: captured_at must be ISO-8601 string, got {type(captured_raw).__name__}"
        )
    try:
        captured_at = datetime.fromisoformat(captured_raw)
    except ValueError as exc:
        raise ExtrinsicsError(f"extrinsics_from_dict: invalid captured_at {captured_raw!r}") from exc

    try:
        return Extrinsics(
            transform=transform,
            rmse_mm=float(data["rmse_mm"]),
            max_error_mm=float(data["max_error_mm"]),
            num_samples=int(data["num_samples"]),
            captured_at=captured_at,
            rig_id=str(data["rig_id"]),
            quality=data.get("quality", "unknown"),
        )
    except (TypeError, ValueError, GeometryError) as exc:
        raise ExtrinsicsError("extrinsics_from_dict: invalid extrinsics payload") from exc


def save_extrinsics(path: str | Path, ext: Extrinsics) -> Path:
    """Atomically write :class:`Extrinsics` as schema-versioned JSON.

    Returns the resolved :class:`Path`. Parent directories are created.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(extrinsics_to_dict(ext), indent=2, sort_keys=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(target)
    return target


def load_extrinsics(path: str | Path) -> Extrinsics:
    """Load :class:`Extrinsics` from a JSON file written by :func:`save_extrinsics`."""
    raw = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ExtrinsicsError(f"load_extrinsics: {path!s} is not valid JSON") from exc
    if not isinstance(data, Mapping):
        raise ExtrinsicsError(f"load_extrinsics: top-level JSON must be an object, got {type(data).__name__}")
    return extrinsics_from_dict(data)


# ----------------------------------------------------------------------
# Eye-in-hand CAMERA -> TOOL calibration transform (for a wrist camera)
# ----------------------------------------------------------------------
def save_cam_to_tool(path: str | Path, transform: Transform, *, rig_id: str) -> Path:
    """Atomically persist an eye-in-hand ``CAMERA -> TOOL`` calibration transform as versioned JSON.

    ``transform`` MUST be ``Transform(from_frame=CAMERA, to_frame=TOOL)`` (translation in mm) --
    the static calibration an :class:`EyeInHandFrameResolver` composes with the live TCP. Raises
    :class:`ExtrinsicsError` on the wrong frames.
    """
    if transform.from_frame is not Frame.CAMERA or transform.to_frame is not Frame.TOOL:
        raise ExtrinsicsError(
            "save_cam_to_tool requires Transform(CAMERA -> TOOL); got "
            f"{transform.from_frame.value} -> {transform.to_frame.value}"
        )
    if not str(rig_id).strip():
        raise ExtrinsicsError("save_cam_to_tool: rig_id must be a non-empty string")
    payload = json.dumps(
        {"schema": CAM_TO_TOOL_SCHEMA, "transform": transform_to_dict(transform), "rig_id": str(rig_id)},
        indent=2, sort_keys=True,
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(target)
    return target


def load_cam_to_tool(path: str | Path) -> Transform:
    """Load an eye-in-hand ``CAMERA -> TOOL`` transform written by :func:`save_cam_to_tool`."""
    raw = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ExtrinsicsError(f"load_cam_to_tool: {path!s} is not valid JSON") from exc
    if not isinstance(data, Mapping) or data.get("schema") != CAM_TO_TOOL_SCHEMA:
        raise ExtrinsicsError(
            f"load_cam_to_tool: expected schema {CAM_TO_TOOL_SCHEMA!r}, got {getattr(data, 'get', lambda _: None)('schema')!r}"
        )
    try:
        transform = transform_from_dict(data["transform"])
    except (GeometryError, TypeError, ValueError, KeyError) as exc:
        raise ExtrinsicsError("load_cam_to_tool: invalid transform payload") from exc
    if transform.from_frame is not Frame.CAMERA or transform.to_frame is not Frame.TOOL:
        raise ExtrinsicsError(
            "load_cam_to_tool: stored transform must be CAMERA -> TOOL; got "
            f"{transform.from_frame.value} -> {transform.to_frame.value}"
        )
    return transform
