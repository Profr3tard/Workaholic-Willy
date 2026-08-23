"""
IKQualityGuard: IK-solution quality safety guard.

Runs *after* the driver has produced a joint solution
(``ctx.target_joints``) for the commanded TCP pose and rejects:

* NaN / infinite values, or a wrong-DoF vector,
* a joint solution that differs from ``ctx.current_joints`` by more
  than ``max_jump_rad`` on any axis,
* a near-singular configuration via the existing
  :func:`analyze_joint_singularity` helper (Jacobian condition number
  and smallest singular value),
* a configuration inside ``limit_proximity_deg`` of the configured
  joint limits. Configured limits come from
  :func:`src.robot.safety.joint_limits.resolve_joint_limits_deg`
  so this guard and :class:`JointLimitGuard` agree on the envelope.

The guard reuses :func:`analyze_joint_singularity` rather than
duplicating the Jacobian math. The only IK-quality-specific work is
the NaN / jump / proximity checks.

Operating modes
---------------
* ``ctx.target_joints is None``: guard returns
  :attr:`SafetyReason.UNAVAILABLE`. The preflight fails closed when
  ``enforce=True``; the driver is responsible for pre-resolving IK
  on Cartesian commands (see ``URRobotArm.move``).
* ``ctx.arm is None`` or ``arm.capabilities.has_native_fk is False``:
  the singularity Jacobian estimate is skipped; the NaN / jump /
  proximity checks still run.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from src.robot.core.errors import RobotError

from .decision import SafetyDecision, SafetyReason
from .guard import SafetyContext
from .joint_limits import resolve_joint_limits_deg
from .singularity import SingularityThresholds, analyze_joint_singularity

if TYPE_CHECKING:  # pragma: no cover - typing only
    from config.schema.robot import (
        IkQualitySafetyConfig,
        JointLimitSafetyConfig,
    )

__all__ = ["IKQualityGuard"]


class IKQualityGuard:
    """IK-solution quality guard.

    Stateless across calls. Constructed by
    :meth:`SafetyPreflight.from_safety_config` when
    ``safety.ik_quality.enforce`` is ``True``.

    Parameters
    ----------
    config
        :class:`IkQualitySafetyConfig` the per-axis jump cap,
        Jacobian thresholds, and limit-proximity buffer.
    joint_limits_config
        :class:`JointLimitSafetyConfig` used by the limit-proximity
        check.
    """

    name = "ik_quality"

    def __init__(
        self,
        config: "IkQualitySafetyConfig",
        joint_limits_config: "JointLimitSafetyConfig | None" = None,
    ) -> None:
        self._config = config
        self._joint_limits_config = joint_limits_config
        self._thresholds = SingularityThresholds(
            min_singular_value=float(config.min_singular_value),
            max_condition_number=float(config.max_condition_number),
        )
        self._max_jump_rad = float(config.max_jump_rad)
        self._limit_proximity_deg = float(config.limit_proximity_deg)

    def evaluate(self, ctx: SafetyContext) -> SafetyDecision:
        joints = ctx.target_joints
        if joints is None:
            return SafetyDecision.unavailable(
                self.name,
                message="no target_joints available for IK-quality check",
                detail={"reason": "missing_target_joints"},
            )

        # ---- NaN / inf -----------------------------------------------
        # JointPositions itself rejects non-finite values at construction,
        # but the type system can be bypassed (e.g. by a buggy driver
        # adapter that pushes raw arrays). Defence-in-depth: re-check.
        values = joints.values
        if not np.isfinite(values).all():
            return SafetyDecision.reject(
                self.name,
                SafetyReason.IK_QUALITY,
                message="IK solution contains non-finite values",
                detail={"reason": "non_finite"},
            )

        # ---- Wrong DoF ------------------------------------------------
        if ctx.arm is not None:
            expected = ctx.arm.capabilities.dof
            if joints.dof != expected:
                return SafetyDecision.reject(
                    self.name,
                    SafetyReason.IK_QUALITY,
                    message=(
                        f"IK solution has {joints.dof} axes; arm "
                        f"expects {expected}"
                    ),
                    detail={
                        "reason": "wrong_dof",
                        "got": str(joints.dof),
                        "expected": str(expected),
                    },
                )

        # ---- Joint jump from current ---------------------------------
        current = ctx.current_joints
        if current is not None and current.dof == joints.dof:
            delta = np.abs(values - current.values)
            worst = int(np.argmax(delta))
            worst_delta = float(delta[worst])
            if worst_delta > self._max_jump_rad:
                return SafetyDecision.reject(
                    self.name,
                    SafetyReason.IK_QUALITY,
                    message=(
                        f"IK jump on axis {worst}: "
                        f"{worst_delta:.4f} rad > "
                        f"{self._max_jump_rad:.4f} rad"
                    ),
                    detail={
                        "reason": "joint_jump",
                        "axis": str(worst),
                        "delta_rad": f"{worst_delta:.6f}",
                        "max_jump_rad": f"{self._max_jump_rad:.6f}",
                    },
                )

        # ---- Limit-proximity -----------------------------------------
        if self._joint_limits_config is not None:
            vendor = ctx.arm.capabilities.vendor if ctx.arm is not None else None
            model = ctx.arm.capabilities.model if ctx.arm is not None else None
            limits = resolve_joint_limits_deg(
                self._joint_limits_config, vendor=vendor, model=model,
            )
            if limits is not None and len(limits[0]) == joints.dof:
                lo_deg, hi_deg = limits
                buffer_deg = self._limit_proximity_deg
                for axis_idx in range(joints.dof):
                    q_deg = math.degrees(float(values[axis_idx]))
                    near_lower = q_deg - lo_deg[axis_idx]
                    near_upper = hi_deg[axis_idx] - q_deg
                    closest = min(near_lower, near_upper)
                    if closest < buffer_deg:
                        return SafetyDecision.reject(
                            self.name,
                            SafetyReason.IK_QUALITY,
                            message=(
                                f"axis {axis_idx} at {q_deg:.3f} deg is "
                                f"only {closest:.3f} deg from a joint "
                                f"limit (buffer={buffer_deg:.3f})"
                            ),
                            detail={
                                "reason": "limit_proximity",
                                "axis": str(axis_idx),
                                "q_deg": f"{q_deg:.6f}",
                                "min_deg": f"{lo_deg[axis_idx]:.6f}",
                                "max_deg": f"{hi_deg[axis_idx]:.6f}",
                                "proximity_deg": f"{closest:.6f}",
                                "buffer_deg": f"{buffer_deg:.6f}",
                            },
                        )

        # ---- Singularity (Jacobian) ----------------------------------
        if ctx.arm is not None and ctx.arm.capabilities.has_native_fk:
            try:
                report = analyze_joint_singularity(
                    ctx.arm, joints, thresholds=self._thresholds,
                )
            except (RobotError, RuntimeError, OSError, ValueError) as exc:
                # FK probe failed mid-flight (e.g. controller dropped).
                # Surface UNAVAILABLE rather than silently passing so
                # operators know the check did not run.
                return SafetyDecision.unavailable(
                    self.name,
                    message=f"Jacobian probe failed: {exc}",
                    detail={
                        "reason": "jacobian_probe_failure",
                        "exception": type(exc).__name__,
                    },
                )
            if report.is_near_singularity:
                return SafetyDecision.reject(
                    self.name,
                    SafetyReason.IK_QUALITY,
                    message=(
                        "near-singular IK solution: "
                        + "; ".join(report.reasons)
                    ),
                    detail={
                        "reason": "singularity",
                        "min_singular_value": (
                            f"{report.min_singular_value:.6g}"
                        ),
                        "condition_number": (
                            f"{report.condition_number:.3f}"
                        ),
                        "rank": str(report.rank),
                        "expected_rank": str(report.expected_rank),
                    },
                )

        return SafetyDecision.accept(self.name)
