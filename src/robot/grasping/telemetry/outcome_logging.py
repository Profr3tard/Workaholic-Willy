"""Typed, JSONL-safe recorder for autonomous grasp attempt outcomes.

Serializes structured grasp reports into stable, replayable records without
accessing hardware, perception, or verification. Supports lossless
``to_dict``/``from_dict`` round-trips, keeps arbitrary data in ``extra``,
and stores summaries rather than raw perception payloads.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Iterator, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "GraspAttemptRecord",
    "append_jsonl",
    "execution_metadata_from",
    "frame_metadata_from",
    "grasp_metadata_from",
    "iter_jsonl",
    "json_safe",
    "profile_metadata_from",
    "recovery_metadata_from",
    "refinement_metadata_from",
    "target_metadata_from",
    "verification_metadata_from",
]


# ---------------------------------------------------------------------------
# JSON sanitisation
# ---------------------------------------------------------------------------


def json_safe(value: Any) -> Any:
    """Convert grasp-stack values into strict JSON-serialisable Python types.

    Recursively handles NumPy values and arrays, enums, mappings, sequences,
    objects with ``to_dict()``, and non-finite floats. Unsupported values fall
    back to their string representation.
    """

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return json_safe(float(value))
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Enum):
        return json_safe(value.value)
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(v) for v in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return json_safe(to_dict())
        except Exception:  # pragma: no cover - defensive
            return str(value)
    return str(value)


# ---------------------------------------------------------------------------
# Builders for typed reports produced upstream
# ---------------------------------------------------------------------------


def profile_metadata_from(profile: Any) -> dict[str, Any]:
    """Snapshot a :class:`GraspBehaviorProfile` into a JSON-safe dict."""

    return {
        "mode": json_safe(getattr(profile, "mode", None)),
        "sampling_mode": json_safe(getattr(profile, "sampling_mode", None)),
        "refinement_enabled": bool(getattr(profile, "refinement_enabled", False)),
        "verification_enabled": bool(
            getattr(profile, "verification_enabled", False)
        ),
        "recovery_allowed_actions": [
            json_safe(a)
            for a in tuple(getattr(profile, "recovery_allowed_actions", ()) or ())
        ],
    }


def frame_metadata_from(frame: Any) -> dict[str, Any]:
    """Summarise a :class:`PerceptionFrame` without persisting pixels."""

    depth = getattr(frame, "depth_map", None)
    rgb = getattr(frame, "rgb", None)
    intrinsics = getattr(frame, "intrinsics", None)
    segmentations = tuple(getattr(frame, "segmentations", ()) or ())

    depth_shape: Optional[list[int]] = None
    if depth is not None:
        depth_shape = [int(d) for d in np.asarray(depth).shape]

    rgb_shape: Optional[list[int]] = None
    if rgb is not None:
        rgb_shape = [int(d) for d in np.asarray(rgb).shape]

    intrinsics_list: Optional[list[list[float]]] = None
    if intrinsics is not None:
        intrinsics_list = json_safe(np.asarray(intrinsics).tolist())

    return {
        "timestamp": json_safe(getattr(frame, "timestamp", None)),
        "depth_shape": depth_shape,
        "rgb_shape": rgb_shape,
        "has_rgb": rgb is not None,
        "segmentation_count": len(segmentations),
        "intrinsics": intrinsics_list,
    }


def target_metadata_from(target: Any) -> Optional[dict[str, Any]]:
    """Summarise a :class:`TargetIdentity` (mask shape + centroid)."""

    if target is None:
        return None
    mask = getattr(target, "mask", None)
    mask_shape: Optional[list[int]] = None
    if mask is not None:
        mask_shape = [int(d) for d in np.asarray(mask).shape]
    centroid = getattr(target, "centroid_xy", None)
    return {
        "mask_shape": mask_shape,
        "centroid_xy": json_safe(centroid),
        "area_px": int(getattr(target, "area_px", 0) or 0),
        "label": getattr(target, "label", None),
    }


def grasp_metadata_from(grasp: Any) -> Optional[dict[str, Any]]:
    """Summarise a :class:`GraspPoint`-like object, preferring its own ``to_dict()`` so the canonical schema is reused verbatim."""

    if grasp is None:
        return None
    to_dict = getattr(grasp, "to_dict", None)
    if callable(to_dict):
        try:
            return json_safe(to_dict())
        except Exception:  # pragma: no cover - defensive
            pass
    return {
        "position": json_safe(getattr(grasp, "position", None)),
        "approach": json_safe(getattr(grasp, "approach", None)),
        "axis": json_safe(getattr(grasp, "axis", None)),
        "grip_width_mm": json_safe(getattr(grasp, "grip_width_mm", None)),
        "score": json_safe(getattr(grasp, "score", None)),
        "frame": json_safe(getattr(grasp, "frame", None)),
        "label": getattr(grasp, "label", None),
        "metadata": json_safe(getattr(grasp, "metadata", {}) or {}),
    }


def refinement_metadata_from(report: Any) -> Optional[dict[str, Any]]:
    """Summarise a :class:`RefinementReport`."""

    if report is None:
        return None
    return {
        "outcome": json_safe(getattr(report, "outcome", None)),
        "matched_segmentation_index": json_safe(
            getattr(report, "matched_segmentation_index", None)
        ),
        "match_iou": json_safe(getattr(report, "match_iou", None)),
        "position_delta_mm": json_safe(
            getattr(report, "position_delta_mm", None)
        ),
        "orientation_delta_deg": json_safe(
            getattr(report, "orientation_delta_deg", None)
        ),
        "grip_width_delta_mm": json_safe(
            getattr(report, "grip_width_delta_mm", None)
        ),
        "failure_reason": json_safe(getattr(report, "failure_reason", None)),
        "telemetry": json_safe(getattr(report, "telemetry", {}) or {}),
    }


def verification_metadata_from(report: Any) -> Optional[dict[str, Any]]:
    """Summarise a :class:`GraspVerificationReport`."""

    if report is None:
        return None
    return {
        "outcome": json_safe(getattr(report, "outcome", None)),
        "reason": getattr(report, "reason", ""),
        "telemetry": json_safe(getattr(report, "telemetry", {}) or {}),
    }


def execution_metadata_from(report: Any) -> Optional[dict[str, Any]]:
    """Summarise executed-grasp outcome data for telemetry.

    Extracts execution details from an autonomous or bare pick report using
    duck-typed ``outcome`` and ``executed_grasp`` fields. Returns ``None`` when
    no grasp was executed.
    """

    if report is None:
        return None
    pick = getattr(report, "pick_report", None)
    if pick is None:
        pick = report
    outcome = getattr(pick, "outcome", None)
    executed = getattr(pick, "executed_grasp", None)
    if outcome is None and executed is None:
        return None
    md: dict[str, Any] = {}
    if outcome is not None:
        md["outcome"] = json_safe(getattr(outcome, "value", outcome))
    if executed is not None:
        eg = grasp_metadata_from(executed)
        if eg is not None:
            md["executed_grasp"] = eg
    return md or None


def recovery_metadata_from(report: Any) -> Optional[dict[str, Any]]:
    """Summarise a :class:`SceneRecoveryReport`."""

    if report is None:
        return None
    plan = getattr(report, "plan", None)
    plan_dict: Optional[dict[str, Any]] = None
    if plan is not None:
        plan_dict = {
            "action": json_safe(getattr(plan, "action", None)),
            "reason": getattr(plan, "reason", ""),
            "nudge_offset_mm": json_safe(
                getattr(plan, "nudge_offset_mm", None)
            ),
            "agitate_amplitude_mm": json_safe(
                getattr(plan, "agitate_amplitude_mm", 0.0)
            ),
            "telemetry": json_safe(getattr(plan, "telemetry", {}) or {}),
        }
    return {
        "plan": plan_dict,
        "executed": bool(getattr(report, "executed", False)),
        "outcome": getattr(report, "outcome", ""),
        "telemetry": json_safe(getattr(report, "telemetry", {}) or {}),
    }


# ---------------------------------------------------------------------------
# The record itself
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GraspAttemptRecord:
    """JSONL-safe summary of a single autonomous grasp attempt.

    All record fields use plain Python primitives, dictionaries/lists, or
    ``None``.

    Fields:
        timestamp:
            Epoch timestamp in seconds. ``new()`` defaults to ``time.time()``.
        attempt_id:
            Operator-supplied identifier used to correlate the attempt.
        mode:
            String value of ``GraspMode``.
        final_outcome:
            Stable top-level outcome string.
        profile, frame, target, initial_grasp, initial_telemetry, refined_grasp,
        refinement, selected_grasp, execution, verification:
            Optional per-stage summaries; ``None`` when not executed.
        recovery_actions:
            Ordered recovery summaries; empty if no recovery was attempted.
        extra:
            Free-form data for fields outside the schema, converted with
            ``json_safe`` during serialisation.
    """

    timestamp: float
    attempt_id: str
    mode: str
    final_outcome: str
    profile: Optional[dict[str, Any]] = None
    frame: Optional[dict[str, Any]] = None
    target: Optional[dict[str, Any]] = None
    initial_grasp: Optional[dict[str, Any]] = None
    initial_telemetry: Mapping[str, Any] = field(default_factory=dict)
    refined_grasp: Optional[dict[str, Any]] = None
    refinement: Optional[dict[str, Any]] = None
    selected_grasp: Optional[dict[str, Any]] = None
    execution: Optional[dict[str, Any]] = None
    verification: Optional[dict[str, Any]] = None
    recovery_actions: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    extra: Mapping[str, Any] = field(default_factory=dict)

    SCHEMA_VERSION: ClassVar[int] = 1

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, (int, float)):
            raise TypeError(
                f"timestamp must be a number; got {type(self.timestamp).__name__}"
            )
        if self.timestamp != self.timestamp or self.timestamp < 0.0:
            raise ValueError(
                f"timestamp must be a finite, non-negative epoch second; "
                f"got {self.timestamp!r}"
            )
        if not isinstance(self.attempt_id, str) or not self.attempt_id:
            raise ValueError("attempt_id must be a non-empty string")
        if not isinstance(self.mode, str) or not self.mode:
            raise ValueError("mode must be a non-empty string")
        if not isinstance(self.final_outcome, str) or not self.final_outcome:
            raise ValueError("final_outcome must be a non-empty string")

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    @classmethod
    def new(
        cls,
        *,
        attempt_id: str,
        mode: str,
        final_outcome: str,
        timestamp: Optional[float] = None,
        **kwargs: Any,
    ) -> "GraspAttemptRecord":
        """Construct a record, defaulting ``timestamp`` to wall clock."""

        ts = float(timestamp) if timestamp is not None else float(time.time())
        return cls(
            timestamp=ts,
            attempt_id=attempt_id,
            mode=mode,
            final_outcome=final_outcome,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe ``dict`` representation; stable key order."""

        return {
            "schema_version": int(self.SCHEMA_VERSION),
            "timestamp": json_safe(self.timestamp),
            "attempt_id": self.attempt_id,
            "mode": self.mode,
            "final_outcome": self.final_outcome,
            "profile": json_safe(self.profile),
            "frame": json_safe(self.frame),
            "target": json_safe(self.target),
            "initial_grasp": json_safe(self.initial_grasp),
            "initial_telemetry": json_safe(self.initial_telemetry),
            "refined_grasp": json_safe(self.refined_grasp),
            "refinement": json_safe(self.refinement),
            "selected_grasp": json_safe(self.selected_grasp),
            "execution": json_safe(self.execution),
            "verification": json_safe(self.verification),
            "recovery_actions": [
                json_safe(a) for a in tuple(self.recovery_actions or ())
            ],
            "extra": json_safe(self.extra),
        }

    def to_json(self) -> str:
        """Single-line JSON; safe to append to a ``.jsonl`` file."""

        return json.dumps(
            self.to_dict(), separators=(",", ":"), sort_keys=False
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GraspAttemptRecord":
        """Reverse of :meth:`to_dict`. Ignores unknown top-level keys."""

        try:
            return cls(
                timestamp=float(data["timestamp"]),
                attempt_id=str(data["attempt_id"]),
                mode=str(data["mode"]),
                final_outcome=str(data["final_outcome"]),
                profile=_optional_dict(data.get("profile")),
                frame=_optional_dict(data.get("frame")),
                target=_optional_dict(data.get("target")),
                initial_grasp=_optional_dict(data.get("initial_grasp")),
                initial_telemetry=dict(data.get("initial_telemetry") or {}),
                refined_grasp=_optional_dict(data.get("refined_grasp")),
                refinement=_optional_dict(data.get("refinement")),
                selected_grasp=_optional_dict(data.get("selected_grasp")),
                execution=_optional_dict(data.get("execution")),
                verification=_optional_dict(data.get("verification")),
                recovery_actions=tuple(
                    dict(a) for a in (data.get("recovery_actions") or ())
                ),
                extra=dict(data.get("extra") or {}),
            )
        except KeyError as exc:
            raise ValueError(
                f"GraspAttemptRecord.from_dict missing key {exc}"
            ) from exc

    @classmethod
    def from_json(cls, line: str) -> "GraspAttemptRecord":
        """Reverse of :meth:`to_json`."""

        return cls.from_dict(json.loads(line))


def _optional_dict(value: Any) -> Optional[dict[str, Any]]:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(
            f"expected mapping or None; got {type(value).__name__}"
        )
    return dict(value)


# ---------------------------------------------------------------------------
# JSONL file helpers
# ---------------------------------------------------------------------------


def append_jsonl(
    record: GraspAttemptRecord | Mapping[str, Any],
    path: str | os.PathLike[str],
) -> Path:
    """Atomically append ``record`` as a single line to ``path``."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(record, GraspAttemptRecord):
        payload = record.to_json()
    else:
        payload = json.dumps(
            json_safe(record), separators=(",", ":"), sort_keys=False
        )
    with p.open("a", encoding="utf-8") as fh:
        fh.write(payload)
        fh.write("\n")
    return p


def iter_jsonl(
    path: str | os.PathLike[str],
) -> Iterator[GraspAttemptRecord]:
    """Yield :class:`GraspAttemptRecord` instances from a JSONL file."""

    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield GraspAttemptRecord.from_json(line)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"{p}: malformed record at line {line_no}: {exc}"
                ) from exc
