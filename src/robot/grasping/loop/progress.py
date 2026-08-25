"""Typed, default-off progress events for live autonomous picks.

Provides a lightweight listener seam for emitting structured pick progress
during execution without affecting the blocking orchestrator flow. Events
use a typed enum and frozen payload, are only constructed when a listener is
attached, and listener failures are swallowed so they cannot interrupt a pick.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final, Protocol, runtime_checkable

__all__ = [
    "PickProgress",
    "PickProgressListener",
    "PickStage",
    "ShouldCancel",
    "emit",
]

_LOG = logging.getLogger(__name__)


class PickStage(StrEnum):
    """The canonical stages of one pick, in the order they occur."""

    #: One pick has begun. Carries ``attempt_total``.
    PICK_STARTED = "pick_started"
    #: One attempt within that pick has begun. Carries ``attempt``/``attempt_total``.
    ATTEMPT_STARTED = "attempt_started"
    #: A camera frame was acquired. Carries ``segmentation_count``.
    PERCEIVED = "perceived"
    #: Candidates were generated and ranked. Carries ``candidate_count``, ``score``, ``target_index``.
    RANKED = "ranked"
    #: No usable candidate this attempt. Carries ``reasons`` and the chosen ``action``.
    NO_CANDIDATE = "no_candidate"
    #: About to command motion. Carries the executed grasp's ``position_mm`` and ``score``.
    EXECUTING = "executing"
    #: The attempt ended. Carries ``action`` and, on the executing path, whether it succeeded.
    ATTEMPT_FINISHED = "attempt_finished"
    #: The pick ended. Carries the typed ``outcome``.
    PICK_FINISHED = "pick_finished"
    #: The loop stopped because a caller asked it to, between attempts. Carries ``attempt``.
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class PickProgress:
    """One progress event. Every field beyond ``stage`` is optional and stage-dependent."""

    stage: PickStage
    #: 0-based index of the attempt this event belongs to. ``None`` for pick-level events.
    attempt: int | None = None
    attempt_total: int | None = None

    segmentation_count: int | None = None
    candidate_count: int | None = None
    target_index: int | None = None
    score: float | None = None
    #: BASE-frame grasp position, millimetres, on :attr:`PickStage.EXECUTING`.
    position_mm: tuple[float, float, float] | None = None
    #: Typed rejection reasons, as strings. Never free text -- these come from ``GraspFailureReason``.
    reasons: tuple[str, ...] = ()
    #: ``executed`` / ``next_target`` / ``rescan`` / ``relocate`` / ``exhausted``.
    action: str | None = None
    #: The typed ``PickOutcome`` on :attr:`PickStage.PICK_FINISHED`.
    outcome: str | None = None
    #: Which perception route grounded this frame (``simple`` / ``vlm``), and the rule that chose it.
    #: Both ``None`` unless the cell runs a routed pipeline, an un-routed cell is byte-identical.
    route: str | None = None
    route_reason: str | None = None
    #: WHY a motion did not happen, on :attr:`PickStage.ATTEMPT_FINISHED`. All three ``None`` on the
    #: healthy path and on every listener that predates them.
    motion_status: str | None = None
    motion_message: str | None = None
    motion_error: str | None = None
    #: Anything a stage needs that does not deserve a field of its own.
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class PickProgressListener(Protocol):
    """Callable the orchestrator notifies. Must be fast and must not raise."""

    def __call__(self, event: PickProgress) -> None: ...


@runtime_checkable
class ShouldCancel(Protocol):
    """Asked at the top of every attempt: should the loop stop rather than start another one?"""

    def __call__(self) -> bool: ...


#: Sentinel documenting the one thing that makes the default path free.
_NO_LISTENER: Final[None] = None


def emit(listener: "PickProgressListener | None", stage: PickStage, **fields: Any) -> None:
    """Build and deliver one event or, with no listener, do nothing at all."""
    if listener is _NO_LISTENER:
        return
    try:
        listener(PickProgress(stage=stage, **fields))  # type: ignore[misc]
    except Exception:  # noqa: BLE001 - a subscriber must never be able to abort a motion in flight
        _LOG.debug("pick progress listener raised on %s; ignored", stage, exc_info=True)
