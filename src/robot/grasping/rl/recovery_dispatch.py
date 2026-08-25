"""Live recovery-dispatch seam for bounded-canary decisions.

This module is a thin, default-off adapter that closes the loop between the
bounded canary router and a live recovery executor:

* obtain a recovery decision from ``canary_router``,
* apply ``decision.applied_action`` through an injected ``execute_recovery``
  callable, and
* report the observed outcome back to the canary KPI window and optional
  Tier-0 online estimator.

The coordinator never touches robot or motor-control surfaces directly.
Execution is entirely delegated to the injected callable, keeping this seam
pure-stdlib, unit-testable, and usable off-box without Isaac or hardware
dependencies.

Safety and recovery authority remain unchanged: the six-channel action mask
and anti-loop gate inside ``choose_recovery`` continue to outrank every
policy proposal.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

from .action_mask import ActionMaskContext
from .canary_router import ActiveCanaryRouter, CanaryDecision
from .online_router import OnlineUpdateCoordinator
from .recovery_policy import RecoveryRequest

#: The two outcomes a dispatched recovery may report back.
RECOVERY_OUTCOME_SUCCEEDED: str = "succeeded"
RECOVERY_OUTCOME_FAILED: str = "failed"
RECOVERY_OUTCOMES: tuple[str, ...] = (RECOVERY_OUTCOME_SUCCEEDED, RECOVERY_OUTCOME_FAILED)


class RecoveryDispatchError(ValueError):
    """Raised when the injected executor returns an outcome outside :data:`RECOVERY_OUTCOMES`."""


class RecoveryDispatchCoordinator:
    """Closes the canary loop: decide -> execute (injected) -> report outcome.

    ``router`` is the bounded canary. ``online`` (optional) is the Tier-0 estimator: when wired, the
    executed ``(state, applied_action, reward)`` is also folded into the online accumulator so online
    learning sees the SAME ground truth the KPI window does (still shadow-only the online state
    never activates without a fresh promotion). Construction never touches a robot."""

    def __init__(
        self,
        *,
        router: ActiveCanaryRouter,
        online: Optional[OnlineUpdateCoordinator] = None,
    ) -> None:
        self._router = router
        self._online = online

    @property
    def router(self) -> ActiveCanaryRouter:
        return self._router

    def dispatch(
        self,
        *,
        request: RecoveryRequest,
        baseline_action: str,
        mode: str,
        execute_recovery: Callable[[str], str],
        attempt_index: int = 0,
        mask_context: ActionMaskContext | None = None,
    ) -> CanaryDecision:
        """Decide the recovery action, apply it via ``execute_recovery``, and report the outcome.

        ``execute_recovery(applied_action)`` performs the dispatched action on the cell and returns
        ``"succeeded"`` / ``"failed"``. The outcome is reported to the canary (KPI window + auto-
        fallback) and, when an online estimator is wired, folded into it as a reward signal. Returns
        the :class:`CanaryDecision` (the fail-closed mask + anti-loop gate already applied inside)."""

        decision = self._router.choose_recovery(
            request=request,
            baseline_action=baseline_action,
            mode=mode,
            attempt_index=attempt_index,
            mask_context=mask_context,
        )
        outcome = execute_recovery(decision.applied_action)
        if outcome not in RECOVERY_OUTCOMES:
            raise RecoveryDispatchError(
                f"execute_recovery returned {outcome!r}; expected one of {RECOVERY_OUTCOMES!r}"
            )
        self._router.record_outcome(decision.audit_id, outcome)
        if self._online is not None:
            # Fold the ACTUALLY-EXECUTED action + reward into the online estimator (shadow-only), so
            # online learning and the KPI window consume identical ground truth.
            reward = 1.0 if outcome == RECOVERY_OUTCOME_SUCCEEDED else 0.0
            self._online.observe_recovery(
                state_key=request.state_key,
                action=decision.applied_action,
                reward=reward,
            )
        return decision


__all__ = (
    "RECOVERY_OUTCOME_SUCCEEDED",
    "RECOVERY_OUTCOME_FAILED",
    "RECOVERY_OUTCOMES",
    "RecoveryDispatchError",
    "RecoveryDispatchCoordinator",
)
