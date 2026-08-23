"""
SafetyGuard: vendor-neutral safety-guard contract.

Each guard in the safety-preflight pipeline implements the
:class:`SafetyGuard` :class:`~typing.Protocol`. The orchestrator
(``SafetyPreflight``) feeds every guard a :class:`SafetyContext`
describing the commanded motion and any current state, and consumes
the guard's :class:`~src.robot.safety.decision.SafetyDecision`.

Design notes
------------
* Guards MUST be stateless across calls *except* for explicit, well-
  documented memos (e.g. the motion-continuity guard caches the last
  accepted target). Cached state belongs on the guard instance, not
  on global module state, so multiple arms / preflights stay isolated.

* Guards MUST NOT raise on ordinary rejection. Returning
  :meth:`SafetyDecision.reject` is the only sanctioned channel.

* Guards MAY raise on programming-error inputs (e.g. wrong-frame pose
  reaching a guard that should only see :attr:`Frame.BASE`). The
  orchestrator does not catch those. They indicate a bug at the call
  site.

* Guards SHOULD short-circuit when their feature flag (``enforce``)
  is disabled, returning :meth:`SafetyDecision.accept` immediately so
  the pipeline stays predictable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.geometry import Pose
    from src.robot.core import (
        JointPositions,
        MotionCommand,
        RobotArm,
    )

    from .decision import SafetyDecision

__all__ = [
    "SafetyContext",
    "SafetyGuard",
]


@dataclass(frozen=True)
class SafetyContext:
    """Input bundle passed to every guard in the preflight pipeline.

    Parameters
    ----------
    command
        Which :class:`MotionCommand` is being evaluated. Lets guards
        specialise (e.g. ``MOVE_HOME`` may skip continuity).
    target_pose
        The commanded TCP pose. ``None`` when the command is a pure
        joint move.
    target_joints
        The commanded joint positions. ``None`` when the driver has
        not yet resolved IK.
    current_pose
        The arm's *current* TCP pose at evaluation time. ``None``
        when not available (e.g. driver is disconnected).
    current_joints
        The arm's *current* joint positions. ``None`` when not
        available.
    last_target_pose
        Previous accepted TCP target, used by continuity guards.
    last_target_joints
        Previous accepted joint target, used by continuity guards.
    arm
        Optional reference to the :class:`RobotArm` being commanded.
    """

    command: "MotionCommand"
    target_pose: "Pose | None" = None
    target_joints: "JointPositions | None" = None
    current_pose: "Pose | None" = None
    current_joints: "JointPositions | None" = None
    last_target_pose: "Pose | None" = None
    last_target_joints: "JointPositions | None" = None
    arm: "RobotArm | None" = None


@runtime_checkable
class SafetyGuard(Protocol):
    """Vendor-neutral safety-guard interface."""

    @property
    def name(self) -> str:
        """Short, stable identifier (e.g. ``"workspace"``,
        ``"joint_limit"``). Used as the ``guard`` field on every
        :class:`SafetyDecision` produced by this guard."""

    def evaluate(self, ctx: SafetyContext) -> "SafetyDecision":
        """Evaluate ``ctx`` and return a :class:`SafetyDecision`.

        Guards MUST NOT raise on ordinary rejection (use
        :meth:`SafetyDecision.reject`). Guards MAY raise on
        programming-error inputs.
        """
