"""
Typed execution results for the vendor-neutral motion contract.

Background
----------
The bool-returning high-level pipeline surface on :class:`RobotArm`
(``move_to``, ``move_home``, ``wait_until_steady``) collapses every
non-success outcome into ``return False``. That works for happy paths
but makes it impossible to:

* tell a workspace rejection apart from an IK failure or controller
  fault,
* report a precise reason to the operator UI or runtime facade,
* migrate orchestration / calibration code onto a vendor-neutral surface
  that is identical between hardware and simulator drivers.

This module defines the canonical typed result surface returned by
:meth:`RobotArm.move`.

Contract
--------
* :class:`MotionStatus` is the closed set of outcome categories every
  driver MUST classify into.
* :class:`MotionCommand` identifies which high-level command produced
  the result so policy reports / runtime reports can carry context
  without string parsing.
* :class:`MotionResult` is the immutable wire type returned by
  ``RobotArm.move`` (and, optionally, future typed motion methods).

When may a driver raise?
~~~~~~~~~~~~~~~~~~~~~~~~
Ordinary execution outcomes (workspace rejection, IK failure,
controller refusal, timeout, marker not detected, etc.) MUST be
returned as a typed :class:`MotionResult`, **not** raised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.geometry import Pose

    from .joint_positions import JointPositions

__all__ = [
    "MotionCommand",
    "MotionResult",
    "MotionStatus",
]


class MotionStatus(StrEnum):
    """Closed set of motion-result categories.

    The string values are stable and safe to log / emit on event
    streams; downstream consumers may switch on the enum directly.
    """

    EXECUTED = "executed"
    """The command was accepted and completed without fault."""

    WORKSPACE_REJECTED = "workspace_rejected"
    """A pre-flight workspace / safety guard refused the target."""

    IK_FAILED = "ik_failed"
    """The driver could not find a valid joint solution for the target."""

    CONTROLLER_REJECTED = "controller_rejected"
    """The controller refused the command (protective stop, limit, etc.)."""

    TIMEOUT = "timeout"
    """The command did not complete within the configured time budget."""

    CONNECTION_ERROR = "connection_error"
    """The transport / link to the controller is down or faulted."""

    UNSUPPORTED = "unsupported"
    """This driver does not implement the requested command kind."""

    INVALID_TARGET = "invalid_target"
    """The target argument is malformed (wrong frame, NaN, etc.)."""

    CANCELLED = "cancelled"
    """The caller cancelled the move before it completed."""

    UNKNOWN = "unknown"
    """Driver could not classify the failure. Avoid where possible."""

    # ------------------------------------------------------------------
    # Safety-guard rejections
    # ------------------------------------------------------------------
    #
    # These categories are produced by the vendor-neutral
    # :class:`src.robot.safety.SafetyPreflight` pipeline. Drivers
    # MUST surface them verbatim when the preflight short-circuits so
    # operators and orchestration can branch on the precise cause
    # without log scraping.

    JOINT_LIMIT_REJECTED = "joint_limit_rejected"
    """A joint target violates configured hard limits or the configured
    safety margin to those limits."""

    IK_QUALITY_REJECTED = "ik_quality_rejected"
    """The IK solution exists but fails a quality check (NaN, wrong
    DoF, excessive joint jump, near singularity, near joint limits)."""

    SELF_COLLISION_REJECTED = "self_collision_rejected"
    """The commanded configuration would intersect the arm with itself,
    its base, its tool, or a declared fixture."""

    PAYLOAD_REJECTED = "payload_rejected"
    """The currently configured payload is outside the allowed mass /
    CoG / inertia envelope."""

    CONTINUITY_REJECTED = "continuity_rejected"
    """The motion-continuity guard refused an abrupt joint or
    orientation jump between consecutive commands."""


NO_PLAN_FAIL_SAFE_MESSAGE = (
    "cuRobo found no collision-free plan; failing safe (no blind motion)."
)


class MotionCommand(StrEnum):
    """High-level command kind that produced a :class:`MotionResult`."""

    MOVE_TO = "move_to"
    MOVE_HOME = "move_home"
    MOVE_JOINTS = "move_joints"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class MotionResult:
    """Immutable typed outcome of one motion command.

    Parameters
    ----------
    status
        Closed-set outcome category. ``EXECUTED`` is the only success
        value; everything else is some flavour of failure.
    command
        Which high-level command produced this result.
    target_pose
        The :class:`Pose` that was commanded, when applicable.
    target_joints
        The :class:`JointPositions` that were commanded, when
        applicable.
    message
        Optional human-readable detail for logs / UI. MAY be empty.
    exception
        Optional underlying exception for ``CONNECTION_ERROR`` /
        ``UNKNOWN``-style faults where the original traceback is
        useful for diagnostics. MAY be ``None``.

    Notes
    -----
    The class is frozen so it can be passed through layers without
    accidental mutation.
    """

    status: MotionStatus
    command: MotionCommand
    target_pose: "Pose | None" = None
    target_joints: "JointPositions | None" = None
    message: str = ""
    exception: BaseException | None = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------
    # Convenience predicates
    # ------------------------------------------------------------------

    @property
    def ok(self) -> bool:
        """``True`` iff the command executed successfully."""
        return self.status is MotionStatus.EXECUTED

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        """Allow ``if result:`` to test success directly."""
        return self.ok

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def executed(
        cls,
        command: MotionCommand,
        *,
        target_pose: "Pose | None" = None,
        target_joints: "JointPositions | None" = None,
        message: str = "",
    ) -> "MotionResult":
        """Build a successful result."""
        return cls(
            status=MotionStatus.EXECUTED,
            command=command,
            target_pose=target_pose,
            target_joints=target_joints,
            message=message,
        )

    @classmethod
    def failed(
        cls,
        status: MotionStatus,
        command: MotionCommand,
        *,
        target_pose: "Pose | None" = None,
        target_joints: "JointPositions | None" = None,
        message: str = "",
        exception: BaseException | None = None,
    ) -> "MotionResult":
        """Build a failure result with an explicit ``status``."""
        if status is MotionStatus.EXECUTED:
            raise ValueError(
                "MotionResult.failed requires a non-EXECUTED status; "
                "use MotionResult.executed for success."
            )
        return cls(
            status=status,
            command=command,
            target_pose=target_pose,
            target_joints=target_joints,
            message=message,
            exception=exception,
        )

    @classmethod
    def from_bool(
        cls,
        ok: bool,
        command: MotionCommand,
        *,
        target_pose: "Pose | None" = None,
        target_joints: "JointPositions | None" = None,
        failure_status: MotionStatus = MotionStatus.CONTROLLER_REJECTED,
        message: str = "",
    ) -> "MotionResult":
        """Bridge a bool return into a typed result."""
        if ok:
            return cls.executed(
                command,
                target_pose=target_pose,
                target_joints=target_joints,
                message=message,
            )
        return cls.failed(
            failure_status,
            command,
            target_pose=target_pose,
            target_joints=target_joints,
            message=message,
        )
