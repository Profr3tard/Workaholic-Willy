"""
SafetyPreflight: ordered safety-guard pipeline.

This is the single entry point every driver MUST consult before
commanding motion. It runs the configured guards in a deterministic
order and short-circuits on the first rejection:

1. workspace            Cartesian box (and optional margin)
2. joint_limit          per-axis joint hard limits + margin
3. ik_quality           IK solution quality
4. self_collision       link-link / link-fixture collision
5. payload              mass / CoG / inertia envelope
6. motion_continuity    step-size between consecutive commands

The workspace box is the one non-negotiable guard (always wired); every
other family is added by :meth:`SafetyPreflight.from_safety_config` only
when its ``enforce`` flag is set, so callers stay unchanged.

Fail-open vs. fail-closed
-------------------------
Every guard family has an ``enforce: bool`` flag in
:class:`RobotSafetyConfig`. When ``enforce`` is ``False`` the guard is
omitted from the pipeline at construction time it does not even
execute. When ``enforce`` is ``True`` and a guard returns
:attr:`SafetyReason.UNAVAILABLE`, the preflight **fails closed**: the
motion is rejected with the guard's
``motion_status_override`` (defaults to
:attr:`MotionStatus.CONTROLLER_REJECTED`) so the operator sees an
honest refusal rather than a silent acceptance.

Driver boundary
---------------
Drivers build a context, evaluate it, and let :meth:`as_motion_result`
translate any rejection into a typed :class:`MotionResult` (``None`` on
accept), so the translation is written once:

.. code-block:: python

    ctx = self._preflight.context_for_pose(pose, current_joints=self.get_joint_positions())
    decision = self._preflight.evaluate(ctx)
    if (result := SafetyPreflight.as_motion_result(decision, command, target_pose=pose)) is not None:
        return result
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from src.robot.constants import SAFETY_PREFLIGHT_LOG_FILE, create_robot_logger
from src.robot.core import MotionCommand, MotionResult

from .continuity import MotionContinuityGuard
from .decision import SafetyDecision, SafetyReason
from .guard import SafetyContext, SafetyGuard
from .ik_quality import IKQualityGuard
from .joint_limits import JointLimitGuard
from .payload import PayloadGuard
from .self_collision import SelfCollisionGuard
from .workspace import WorkspaceGuard

if TYPE_CHECKING:  # pragma: no cover - typing only
    from config.schema.robot import (
        RobotSafetyConfig,
        WorkspaceLimitsConfig,
    )
    from src.geometry import Pose
    from src.robot.core import JointPositions, RobotArm

__all__ = [
    "SafetyPreflight",
    "WorkspaceSafetyGuard",
]


class WorkspaceSafetyGuard:
    """:class:`SafetyGuard` adapter around the existing
    :class:`WorkspaceGuard` workspace-box check.

    Only the *workspace-box* portion of the legacy guard is consulted
    here. The diversity check stays on :class:`WorkspaceGuard` itself
    because it is a calibration-time concern (pose sampling), not a
    per-move safety question.

    Parameters
    ----------
    guard
        Backing :class:`WorkspaceGuard`. Reused as-is so the existing
        :class:`MotionController` / :class:`PoseProvider` integrations
        keep working.
    """

    name = "workspace"

    def __init__(self, guard: WorkspaceGuard) -> None:
        self._guard = guard

    def evaluate(self, ctx: SafetyContext) -> SafetyDecision:
        if ctx.target_pose is None:
            # Joint-only command: no Cartesian target to bound-check.
            # Real Cartesian bounds for joint moves will be enforced by
            # the joint-limit guard and self-collision guard.
            return SafetyDecision.accept(self.name, message="no Cartesian target")
        pose = ctx.target_pose
        if not self._guard.is_inside_workspace(pose):
            label = pose.label or "<unlabeled>"
            x, y, z = pose.position_mm
            return SafetyDecision.reject(
                self.name,
                SafetyReason.WORKSPACE,
                message=f"pose '{label}' outside workspace box",
                detail={
                    "label": label,
                    "x_mm": f"{float(x):.3f}",
                    "y_mm": f"{float(y):.3f}",
                    "z_mm": f"{float(z):.3f}",
                },
            )
        return SafetyDecision.accept(self.name)


class SafetyPreflight:
    """Ordered :class:`SafetyGuard` pipeline.

    Construction
    ------------
    The straight constructor takes a pre-assembled sequence of guards
    in the order they should run. Prefer :meth:`from_safety_config`
    (or :meth:`from_workspace_only` in tests) over hand-rolling the
    sequence: those factories enforce the canonical guard order and
    the ``enforce``-flag handling.
    """

    # Canonical order of guard names. Used by :meth:`from_safety_config`
    # to sort whatever guard set the operator configured into the
    # deterministic execution order documented in the module docstring.
    _CANONICAL_ORDER: tuple[str, ...] = (
        "workspace",
        "joint_limit",
        "ik_quality",
        "self_collision",
        "payload",
        "motion_continuity",
    )

    def __init__(self, guards: Sequence[SafetyGuard]) -> None:
        # Defensive copy to a tuple so the pipeline is immutable after
        # construction. Guard *instances* remain mutable (they own
        # their own caches), but the *list* cannot be reordered or
        # extended at runtime.
        self._guards: tuple[SafetyGuard, ...] = tuple(guards)
        self._logger = create_robot_logger("SafetyPreflight", SAFETY_PREFLIGHT_LOG_FILE)
        self._last_target_pose: "Pose | None" = None
        self._last_target_joints: "JointPositions | None" = None

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_workspace_only(cls, workspace_guard: WorkspaceGuard) -> "SafetyPreflight":
        """Construct a preflight with **only** the workspace guard."""
        return cls([WorkspaceSafetyGuard(workspace_guard)])

    @classmethod
    def from_safety_config(
        cls,
        safety_cfg: "RobotSafetyConfig",
        workspace_cfg: "WorkspaceLimitsConfig",
        *,
        extra_guards: Iterable[SafetyGuard] = (),
    ) -> "SafetyPreflight":
        """Build the canonical preflight pipeline from configuration.

        Parameters
        ----------
        safety_cfg
            Vendor-neutral safety knobs. Each sub-block carries an
            ``enforce`` flag; disabled blocks are omitted from the
            pipeline entirely (their guards never run).
        workspace_cfg
            Cartesian workspace box. The workspace guard always runs
            because the workspace box is the single non-negotiable
            safety surface for the operator cell.
        extra_guards
            Optional additional guards injected after the canonical
            set. Each must already be sorted into the right canonical
            slot by name; the factory re-sorts the full set into
            :attr:`_CANONICAL_ORDER` before constructing the preflight.
        """
        # Workspace guard is non-negotiable: always wired.
        workspace_margin = float(safety_cfg.limits.workspace_margin_mm)
        narrowed = _shrink_workspace(workspace_cfg, workspace_margin)
        workspace_guard = WorkspaceGuard(limits=narrowed)
        guards: list[SafetyGuard] = [WorkspaceSafetyGuard(workspace_guard)]
        # Wire the per-family guards whose ``enforce`` flag is ``True``.
        if safety_cfg.joint_limits.enforce:
            guards.append(JointLimitGuard(safety_cfg.joint_limits))
        if safety_cfg.ik_quality.enforce:
            guards.append(
                IKQualityGuard(
                    safety_cfg.ik_quality,
                    joint_limits_config=safety_cfg.joint_limits,
                )
            )
        if safety_cfg.self_collision.enforce:
            guards.append(SelfCollisionGuard(safety_cfg.self_collision))
        if safety_cfg.payload.enforce:
            guards.append(PayloadGuard(safety_cfg.payload))
        if safety_cfg.motion_continuity.enforce:
            guards.append(MotionContinuityGuard(safety_cfg.motion_continuity))
        guards.extend(extra_guards)
        guards.sort(key=lambda g: cls._guard_order_key(g.name))
        return cls(guards)

    @staticmethod
    def _guard_order_key(name: str) -> int:
        try:
            return SafetyPreflight._CANONICAL_ORDER.index(name)
        except ValueError:
            # Unknown guards run last in insertion order; this keeps
            # the door open for experimental site-local guards without
            # demanding a contract change.
            return len(SafetyPreflight._CANONICAL_ORDER)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def guards(self) -> tuple[SafetyGuard, ...]:
        """Snapshot of the configured guard pipeline (ordered)."""
        return self._guards

    @property
    def guard_names(self) -> tuple[str, ...]:
        """Tuple of guard ``name`` properties in execution order."""
        return tuple(g.name for g in self._guards)

    # ------------------------------------------------------------------
    # Context builders
    # ------------------------------------------------------------------

    def context_for_pose(
        self,
        pose: "Pose",
        *,
        command: MotionCommand = MotionCommand.MOVE_TO,
        target_joints: "JointPositions | None" = None,
        current_pose: "Pose | None" = None,
        current_joints: "JointPositions | None" = None,
        arm: "RobotArm | None" = None,
    ) -> SafetyContext:
        """Build a :class:`SafetyContext` for a Cartesian move.

        The preflight's memoised ``last_target_*`` are folded in so
        the continuity guard can see the previous accepted target
        without the driver tracking it.
        """
        return SafetyContext(
            command=command,
            target_pose=pose,
            target_joints=target_joints,
            current_pose=current_pose,
            current_joints=current_joints,
            last_target_pose=self._last_target_pose,
            last_target_joints=self._last_target_joints,
            arm=arm,
        )

    def context_for_joints(
        self,
        joints: "JointPositions",
        *,
        command: MotionCommand = MotionCommand.MOVE_JOINTS,
        target_pose: "Pose | None" = None,
        current_pose: "Pose | None" = None,
        current_joints: "JointPositions | None" = None,
        arm: "RobotArm | None" = None,
    ) -> SafetyContext:
        """Build a :class:`SafetyContext` for a joint move."""
        return SafetyContext(
            command=command,
            target_pose=target_pose,
            target_joints=joints,
            current_pose=current_pose,
            current_joints=current_joints,
            last_target_pose=self._last_target_pose,
            last_target_joints=self._last_target_joints,
            arm=arm,
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self, ctx: SafetyContext, *, skip_guards: "frozenset[str]" = frozenset()
    ) -> SafetyDecision:
        """Run the pipeline. Returns the first rejection or accept.

        Side effect: on acceptance, the memoised ``last_target_*``
        are updated from ``ctx`` so the next call's continuity guard
        has them available.
        """
        for guard in self._guards:
            if guard.name in skip_guards:
                continue
            decision = guard.evaluate(ctx)
            if decision.rejected:
                self._logger.warning(
                    "Safety preflight rejected motion: guard=%s reason=%s "
                    "message=%s detail=%s",
                    decision.guard, decision.reason.value,
                    decision.message or "<empty>", decision.detail or {},
                )
                return decision
        # Accepted: remember the target for the next continuity check.
        if ctx.target_pose is not None:
            self._last_target_pose = ctx.target_pose
        if ctx.target_joints is not None:
            self._last_target_joints = ctx.target_joints
        return SafetyDecision.accept("preflight")

    def reset(self) -> None:
        """Clear the continuity memo.

        Drivers SHOULD call this after a controller reset, an
        emergency stop, or any other event that invalidates the
        "previous target" assumption.
        """
        self._last_target_pose = None
        self._last_target_joints = None

    # Guards that gate a Cartesian PATH near a target, NOT a deliberate point-to-point joint command:
    # workspace (box, pose-only), ik_quality (IK-jump step + singularity), motion_continuity (step size).
    # Singularity / large steps only bite during Cartesian or servo control near the config, not an
    # interpolated, in-limits, collision-free JOINT command. (True on real hardware too: a moveJ to a
    # singular-but-reachable config is fine; the singularity only bites under subsequent Cartesian control.)
    # Deliberately a code-level constant, NOT operator YAML: the skip-set is a fail-closed
    # safety-surface selector, an operator typo there could silence a guard on a real joint move.
    _JOINT_MOVE_SKIP_GUARDS = ("workspace", "ik_quality", "motion_continuity")

    def gate_joint_target(
        self,
        joints: "JointPositions",
        *,
        arm: "RobotArm | None" = None,
    ) -> "MotionResult | None":
        """Gate a commanded JOINT-space move; return a typed rejection or ``None`` (accepted)."""
        self.reset()  # restart: do NOT diff continuity into this joint move
        ctx = self.context_for_joints(joints, command=MotionCommand.MOVE_JOINTS, arm=arm)
        rejected: "MotionResult | None" = None
        for guard in self._guards:
            if guard.name in self._JOINT_MOVE_SKIP_GUARDS:
                continue
            decision = guard.evaluate(ctx)
            if decision.rejected:
                self._logger.warning(
                    "Safety preflight rejected JOINT move: guard=%s reason=%s message=%s",
                    decision.guard, decision.reason.value, decision.message or "<empty>",
                )
                rejected = SafetyPreflight.as_motion_result(
                    decision, MotionCommand.MOVE_JOINTS, target_joints=joints,
                )
                break
        self.reset()  # restart: do NOT diff the NEXT Cartesian move across this joint move
        return rejected

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def as_motion_result(
        decision: SafetyDecision,
        command: MotionCommand,
        *,
        target_pose: "Pose | None" = None,
        target_joints: "JointPositions | None" = None,
    ) -> "MotionResult | None":
        """Translate a rejection into a typed :class:`MotionResult`.

        Returns ``None`` for accepted decisions so drivers can write::

            if (result := SafetyPreflight.as_motion_result(decision, cmd, ...)) is not None:
                return result
        """
        if decision.accepted:
            return None
        status = decision.motion_status
        # ``motion_status`` is guaranteed non-None for rejected
        # decisions by :meth:`SafetyDecision.motion_status`.
        assert status is not None  # noqa: S101 - invariant guard
        message = (
            f"[safety:{decision.guard}/{decision.reason.value}] "
            f"{decision.message}"
        ) if decision.message else (
            f"[safety:{decision.guard}/{decision.reason.value}]"
        )
        return MotionResult.failed(
            status,
            command,
            target_pose=target_pose,
            target_joints=target_joints,
            message=message,
        )


def _shrink_workspace(
    cfg: "WorkspaceLimitsConfig",
    margin_mm: float,
) -> "WorkspaceLimitsConfig":
    """Return a :class:`WorkspaceLimitsConfig` shrunk by ``margin_mm``
    on every face.

    A non-positive margin returns ``cfg`` unchanged. Margins that
    would invert any axis raise :class:`ValueError`.
    """
    if margin_mm <= 0.0:
        return cfg
    new = {
        "x_min": cfg.x_min + margin_mm,
        "x_max": cfg.x_max - margin_mm,
        "y_min": cfg.y_min + margin_mm,
        "y_max": cfg.y_max - margin_mm,
        "z_min": cfg.z_min + margin_mm,
        "z_max": cfg.z_max - margin_mm,
    }
    if new["x_min"] >= new["x_max"] or new["y_min"] >= new["y_max"] or new["z_min"] >= new["z_max"]:
        raise ValueError(
            f"workspace_margin_mm={margin_mm} inverts the workspace box "
            f"{(cfg.x_min, cfg.x_max, cfg.y_min, cfg.y_max, cfg.z_min, cfg.z_max)}"
        )
    return cfg.__class__(**new)
