"""Recovery-trail serializer: a ``RecoveryTrail`` -> per-step ``recovery_actions`` records.

Pure serializer with no orchestrator state, so it lives in its own leaf and is re-exported from
recovery_orchestrator (kept for the by-name importers).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.robot.grasping.recovery.orchestrator import RecoveryTrail


def recovery_actions_from_trail(trail: RecoveryTrail) -> tuple[dict[str, Any], ...]:
    """Serialize a recovery trail into per-step ``recovery_actions`` records.

    Each step becomes ``{action, outcome, step_result, executed, failure_reason, plan_reason}``. The per-step
    ``outcome`` carries the reward signal: the LAST step's outcome is ``"recovered_success"`` iff the recovery
    ultimately succeeded (``terminal_reason == "recovered_success"``), else the action's own immediate result;
    the raw action result is always preserved in ``step_result``.
    """
    entries = tuple(trail.entries)
    recovered = trail.terminal_reason == "recovered_success"
    last = len(entries) - 1
    rows: list[dict[str, Any]] = []
    for i, e in enumerate(entries):
        action = getattr(e, "plan_action", None)
        failure = getattr(e, "failure_reason", None)
        raw_outcome = str(getattr(e, "outcome", ""))
        rows.append(
            {
                "action": str(getattr(action, "value", action)),
                "outcome": "recovered_success" if (i == last and recovered) else raw_outcome,
                "step_result": raw_outcome,
                "executed": bool(getattr(e, "executed", False)),
                "failure_reason": str(getattr(failure, "value", failure)),
                "plan_reason": str(getattr(e, "plan_reason", "")),
            }
        )
    return tuple(rows)
