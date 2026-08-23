"""
SafetyDecision: typed outcome of a single safety-guard evaluation.

Background
----------
The uniform safety-preflight pipeline (joint-limit, IK-quality,
self-collision, payload, motion-continuity, workspace) runs **before**
every commanded motion. Each guard in the pipeline must report its
verdict in a way that is:

* unambiguous (no string parsing in callers),
* structured (carries enough context for logs / UI / events),
* and trivially mappable onto the closed :class:`MotionStatus`
  set so a typed :class:`MotionResult` can be assembled at the driver
  boundary without further classification.

:class:`SafetyDecision` is that wire type. :class:`SafetyReason` is the
closed set of verdict categories, one per guard family plus ``OK`` and
``UNAVAILABLE``.

Numerics contract
-----------------
``SafetyDecision`` carries free-form ``detail: dict[str, str]`` so
guards can attach structured context (e.g. the offending joint index,
the measured singular value) without losing precision. Numeric values
SHOULD be formatted into strings by the guard with enough precision
that diagnostics tooling can re-parse them; the dict is intended for
logging / event payloads, not for cross-guard arithmetic.

When does a guard return ``UNAVAILABLE``?
-----------------------------------------
A guard reports :attr:`SafetyReason.UNAVAILABLE` when it lacks the
data needed to make a decision **and** the operator has not explicitly
opted in to failing closed for that condition. Examples:

* Self-collision guard configured for the ``fcl`` backend, but the
  configured mesh directory is empty.
* Joint-limit guard has no static fallback limits and the connected
  driver does not expose telemetry (e.g. KUKA EKI).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from src.robot.core import MotionStatus

__all__ = [
    "SafetyDecision",
    "SafetyReason",
]


class SafetyReason(str, Enum):
    """Closed set of safety-decision reason categories.

    String values are stable and safe to log / emit on event streams.
    Every non-``OK`` reason MUST map to exactly one
    :class:`MotionStatus` value via :func:`safety_reason_to_motion_status`
    so the driver boundary can synthesise a typed
    :class:`MotionResult` without further classification.
    """

    OK = "ok"
    """The guard accepted the commanded motion."""

    WORKSPACE = "workspace"
    """Cartesian workspace box / orientation diversity rejection."""

    JOINT_LIMIT = "joint_limit"
    """Joint hard limit (or configured margin) would be violated."""

    IK_QUALITY = "ik_quality"
    """IK solution exists but fails a quality check (singularity, jump,
    near limits)."""

    SELF_COLLISION = "self_collision"
    """Commanded configuration would intersect the arm / tool / fixture."""

    PAYLOAD = "payload"
    """Configured payload exceeds the allowed mass / CoG / inertia
    envelope."""

    CONTINUITY = "continuity"
    """Abrupt joint or orientation step between consecutive commands."""

    UNAVAILABLE = "unavailable"
    """Guard could not evaluate (missing telemetry, missing asset)."""


# ---------------------------------------------------------------------
# SafetyReason -> MotionStatus mapping
# ---------------------------------------------------------------------

_REASON_TO_STATUS: dict[SafetyReason, MotionStatus] = {
    SafetyReason.WORKSPACE: MotionStatus.WORKSPACE_REJECTED,
    SafetyReason.JOINT_LIMIT: MotionStatus.JOINT_LIMIT_REJECTED,
    SafetyReason.IK_QUALITY: MotionStatus.IK_QUALITY_REJECTED,
    SafetyReason.SELF_COLLISION: MotionStatus.SELF_COLLISION_REJECTED,
    SafetyReason.PAYLOAD: MotionStatus.PAYLOAD_REJECTED,
    SafetyReason.CONTINUITY: MotionStatus.CONTINUITY_REJECTED,
    SafetyReason.UNAVAILABLE: MotionStatus.CONTROLLER_REJECTED,
}


def safety_reason_to_motion_status(reason: SafetyReason) -> MotionStatus:
    """Translate a non-``OK`` :class:`SafetyReason` into a
    :class:`MotionStatus`.

    Raises
    ------
    ValueError
        If ``reason`` is :attr:`SafetyReason.OK` (which represents
        acceptance and has no failure status).
    """
    if reason is SafetyReason.OK:
        raise ValueError(
            "SafetyReason.OK has no MotionStatus mapping; the motion "
            "is accepted and no MotionResult should be synthesised."
        )
    try:
        return _REASON_TO_STATUS[reason]
    except KeyError as exc:  # pragma: no cover - exhaustive enum
        raise ValueError(
            f"No MotionStatus mapping for SafetyReason {reason!r}."
        ) from exc


@dataclass(frozen=True)
class SafetyDecision:
    """Immutable verdict produced by a single safety guard.

    Parameters
    ----------
    accepted
        ``True`` iff the guard accepts the commanded motion.
    reason
        Closed-set category of the verdict. MUST be
        :attr:`SafetyReason.OK` when ``accepted`` is ``True``.
    guard
        Short identifier of the producing guard
        (e.g. ``"workspace"``, ``"joint_limit"``). Useful for logs /
        UI so operators can tell *which* guard refused.
    message
        Human-readable message, suitable for direct display in
        operator UI / logs. MAY be empty for ``accepted=True``.
    detail
        Free-form structured detail (offending joint index, measured
        singular value, etc.). Stored as ``dict[str, str]`` so it can
        be serialised straight into structured event payloads without
        further coercion.
    motion_status_override
        Optional :class:`MotionStatus` to use at the driver boundary
        in place of the default mapping for ``reason``.

    Construction
    ------------
    Prefer :meth:`accept`, :meth:`reject`, :meth:`unavailable` over
    the bare constructor at call sites.
    """

    accepted: bool
    reason: SafetyReason
    guard: str
    message: str = ""
    detail: dict[str, str] = field(default_factory=dict)
    motion_status_override: MotionStatus | None = None

    def __post_init__(self) -> None:
        if self.accepted and self.reason is not SafetyReason.OK:
            raise ValueError(
                "SafetyDecision.accepted=True requires SafetyReason.OK; "
                f"got reason={self.reason!r}."
            )
        if not self.accepted and self.reason is SafetyReason.OK:
            raise ValueError(
                "SafetyDecision.accepted=False forbids SafetyReason.OK; "
                "use one of the rejection reasons."
            )
        if not isinstance(self.guard, str) or not self.guard:
            raise ValueError(
                "SafetyDecision.guard must be a non-empty string."
            )

    # ------------------------------------------------------------------
    # Convenience predicates
    # ------------------------------------------------------------------

    @property
    def rejected(self) -> bool:
        """``True`` iff this decision refuses the motion."""
        return not self.accepted

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        """Allow ``if decision: ...`` to mean "accepted"."""
        return self.accepted

    @property
    def motion_status(self) -> MotionStatus | None:
        """:class:`MotionStatus` to surface at the driver boundary.

        ``None`` when the decision is accepted (no result needed).
        Returns the explicit override when set, otherwise the default
        mapping for :attr:`reason`.
        """
        if self.accepted:
            return None
        if self.motion_status_override is not None:
            return self.motion_status_override
        return safety_reason_to_motion_status(self.reason)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def accept(cls, guard: str, *, message: str = "") -> "SafetyDecision":
        """Build an accepted decision for ``guard``."""
        return cls(
            accepted=True,
            reason=SafetyReason.OK,
            guard=guard,
            message=message,
        )

    @classmethod
    def reject(
        cls,
        guard: str,
        reason: SafetyReason,
        *,
        message: str = "",
        detail: dict[str, str] | None = None,
        motion_status_override: MotionStatus | None = None,
    ) -> "SafetyDecision":
        """Build a rejection decision.

        ``reason`` must not be :attr:`SafetyReason.OK`.
        """
        if reason is SafetyReason.OK:
            raise ValueError(
                "SafetyDecision.reject requires a non-OK reason; use "
                "SafetyDecision.accept for acceptance."
            )
        return cls(
            accepted=False,
            reason=reason,
            guard=guard,
            message=message,
            detail=dict(detail) if detail else {},
            motion_status_override=motion_status_override,
        )

    @classmethod
    def unavailable(
        cls,
        guard: str,
        *,
        message: str = "",
        detail: dict[str, str] | None = None,
        motion_status_override: MotionStatus | None = None,
    ) -> "SafetyDecision":
        """Build an ``UNAVAILABLE`` decision.

        The preflight orchestrator decides whether this fails open or
        fails closed based on the guard's ``enforce`` flag.
        """
        return cls(
            accepted=False,
            reason=SafetyReason.UNAVAILABLE,
            guard=guard,
            message=message,
            detail=dict(detail) if detail else {},
            motion_status_override=motion_status_override,
        )
