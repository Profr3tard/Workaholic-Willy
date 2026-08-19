"""Vendor-neutral safety-guard config schema."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .._base import StrictModel


class LimitsSafetyConfig(StrictModel):
    """Workspace-margin safety config.

    ``workspace_margin_mm`` is consumed by :class:`SafetyPreflight.from_safety_config` to shrink the
    ``workspace_limits`` box on every face before the workspace guard runs.
    """

    enforce: bool = Field(default=True)

    workspace_margin_mm: float = Field(default=20.0, ge=0.0, le=500.0)


class JointLimitSafetyConfig(StrictModel):
    """Per-axis joint-limit guard configuration.

    The guard prefers driver-side telemetry when available (UR's
    RTDE ``query_joint_limits``); otherwise it falls back to the
    static ``min_deg`` / ``max_deg`` lists below. When neither source
    is available the guard returns :attr:`SafetyReason.UNAVAILABLE` and
    fails closed if ``enforce`` is ``True``.

    ``margin_deg`` is the additional buffer (in degrees) the guard
    keeps between each commanded joint angle and the actual axis limit.
    """

    enforce: bool = Field(default=True)
    margin_deg: float = Field(default=5.0, ge=0.0, le=45.0)
    # Optional per-axis static fallback in degrees. If provided, both
    # lists MUST be the same length as the arm's DoF and ``min_deg[i]
    # < max_deg[i]`` for every axis i.
    min_deg: list[float] | None = Field(default=None)
    max_deg: list[float] | None = Field(default=None)

    @model_validator(mode="after")
    def _check_axis_ordering(self) -> JointLimitSafetyConfig:
        if (self.min_deg is None) != (self.max_deg is None):
            raise ValueError(
                "joint_limit.min_deg and joint_limit.max_deg must be set "
                "together (both or neither)."
            )
        if self.min_deg is not None and self.max_deg is not None:
            if len(self.min_deg) != len(self.max_deg):
                raise ValueError(
                    "joint_limit.min_deg and joint_limit.max_deg must have "
                    "the same length."
                )
            for i, (lo, hi) in enumerate(zip(self.min_deg, self.max_deg)):
                if lo >= hi:
                    raise ValueError(
                        f"joint_limit axis {i}: min_deg ({lo}) must be < "
                        f"max_deg ({hi})."
                    )
        return self


class IkQualitySafetyConfig(StrictModel):
    """IK-solution-quality guard configuration.

    The guard runs *after* the driver has produced a joint solution and
    rejects:

    * NaN / wrong-DoF solutions outright,
    * joint solutions that differ from the current pose by more than
      ``max_jump_rad`` on any axis (large IK jumps signal wrap-around
      ambiguities),
    * solutions whose Jacobian condition number exceeds
      ``max_condition_number`` or whose smallest singular value falls
      below ``min_singular_value`` (near-singular configurations),
    * solutions inside ``limit_proximity_deg`` of the configured joint
      limits (handled by the joint-limit guard *in addition*, but the
      IK-quality guard catches the case where the limit guard is
      disabled).
    """

    enforce: bool = Field(default=True)
    max_jump_rad: float = Field(default=1.0, gt=0.0, le=6.283185307179586)
    min_singular_value: float = Field(default=0.005, gt=0.0, le=1.0)
    max_condition_number: float = Field(default=250.0, gt=0.0, le=1.0e6)
    limit_proximity_deg: float = Field(default=2.0, ge=0.0, le=45.0)


class FixtureBoxConfig(StrictModel):
    """Static fixture obstacle used by the self-collision guard.

    Boxes are axis-aligned in the robot base frame. Defined by
    ``center_mm`` (the geometric centre) and ``half_extents_mm`` (half
    the side length on each axis). Fixtures are checked against
    capsule-approximated arm links in the capsule backend and against
    user-provided mesh files in the ``fcl`` backend.
    """

    name: str = Field(default="fixture", min_length=1)
    center_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    half_extents_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @model_validator(mode="after")
    def _check_positive_extents(self) -> FixtureBoxConfig:
        for axis, extent in zip(("x", "y", "z"), self.half_extents_mm):
            if extent < 0.0:
                raise ValueError(
                    f"fixture {self.name!r}: half_extents_mm.{axis} must be "
                    f">= 0; got {extent}."
                )
        return self


class SelfCollisionSafetyConfig(StrictModel):
    """Self-collision guard configuration.

    Two backends are supported:

    * ``capsule`` (default, always available) approximates each arm
      link as a capsule and runs a closed-form distance check between
      every link pair plus every link/fixture pair.
    * ``fcl`` uses the optional ``python-fcl`` package against the
      meshes in ``mesh_dir``. Selecting ``fcl`` without a populated
      ``mesh_dir`` causes the guard to report
      :attr:`SafetyReason.UNAVAILABLE`; if ``enforce`` is ``True`` the
      motion is rejected (the guard fails closed).

    ``min_distance_mm`` is the minimum signed distance between any two
    monitored shapes; closer than this is treated as a collision.

    ``link_radii_mm`` is the per-link capsule radius in mm. When
    omitted the guard uses a single default radius supplied at runtime.
    Lists are not strict-validated here for length because DoF is a
    driver-side property; the guard validates lengths at evaluation
    time and reports :attr:`SafetyReason.UNAVAILABLE` on mismatch.
    """

    enforce: bool = Field(default=True)
    backend: Literal["capsule", "fcl"] = Field(default="capsule")
    min_distance_mm: float = Field(default=10.0, ge=0.0, le=500.0)
    planner_margin_mm: float = Field(default=0.0, ge=0.0, le=100.0)
    link_radii_mm: list[float] | None = Field(default=None)
    fixtures: list[FixtureBoxConfig] = Field(default_factory=list)
    mesh_dir: str | None = Field(default=None)
    kinematics_model: str | None = Field(default=None)
    collision_mesh_variant: str | None = Field(default=None)
    kinematics_base_yaw_deg: float = Field(default=0.0, ge=-360.0, le=360.0)
    tool_length_mm: float = Field(default=150.0, gt=0.0, le=1000.0)
    tool_radius_mm: float = Field(default=70.0, gt=0.0, le=500.0)
    base_radius_mm: float = Field(default=80.0, gt=0.0, le=1000.0)
    base_height_mm: float = Field(default=150.0, gt=0.0, le=2000.0)
    tool_model: Literal["capsule", "finger"] = Field(default="capsule")
    tool_finger_radius_mm: float = Field(default=16.0, gt=0.0, le=200.0)
    tool_finger_span_mm: float = Field(default=150.0, gt=0.0, le=500.0)


class PayloadSafetyConfig(StrictModel):
    """Tool / payload envelope.

    Validated at the point the operator updates the payload (e.g. by
    calling ``URRobotArm.set_payload`` or by editing this YAML and
    reloading).
    """

    enforce: bool = Field(default=True)
    mass_kg: float = Field(default=0.0, ge=0.0, le=50.0)
    max_mass_kg: float = Field(default=5.0, gt=0.0, le=50.0)
    cog_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    inertia_kgm2: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @model_validator(mode="after")
    def _check_envelope(self) -> PayloadSafetyConfig:
        if self.mass_kg > self.max_mass_kg:
            raise ValueError(
                f"payload.mass_kg ({self.mass_kg}) must be <= max_mass_kg "
                f"({self.max_mass_kg})."
            )
        for axis, val in zip(("Ixx", "Iyy", "Izz"), self.inertia_kgm2):
            if val < 0.0:
                raise ValueError(
                    f"payload.inertia_kgm2.{axis} must be >= 0; got {val}."
                )
        return self


class MotionContinuitySafetyConfig(StrictModel):
    """Step-size guard between consecutive commanded targets.

    Catches accidental ``move(other_pose)`` calls that would slam the
    arm across the workspace.
    """

    enforce: bool = Field(default=True)
    max_joint_step_deg: float = Field(default=45.0, gt=0.0, le=180.0)
    max_orientation_step_deg: float = Field(default=30.0, gt=0.0, le=180.0)
    max_tcp_step_mm: float = Field(default=250.0, gt=0.0, le=2000.0)


class DwellSafetyConfig(StrictModel):
    """Post-Stop dwell + steady-state gating."""

    dwell_after_stop_s: float = Field(default=0.5, ge=0.0, le=10.0)
    require_steady_before_motion: bool = Field(default=True)
    steady_timeout_s: float = Field(default=5.0, gt=0.0, le=60.0)


class RobotSafetyConfig(StrictModel):
    """Vendor-neutral safety surface.

    Every safety knob lives in a dedicated sub-block that owns its own ``enforce`` flag so operators can
    disable individual guards without touching the others.

    Sub-blocks
    ----------
    * :attr:`limits` the ``workspace_margin_mm`` consumed by the workspace guard.
    * :attr:`joint_limits` per-axis joint hard limits + margin.
    * :attr:`ik_quality` IK-solution quality checks.
    * :attr:`motion_continuity` step-size between consecutive commanded targets.
    * :attr:`payload` mass / CoG / inertia envelope.
    * :attr:`self_collision` link-link / link-fixture collision.
    * :attr:`dwell` post-Stop dwell + steady-state gating.

    NOTE: the EMERGENCY STOP. It is a hardware and controller function, outside this software's
    control and deliberately so nothing in this package can enable, disable, route or observe it.
    """

    limits: LimitsSafetyConfig = Field(default_factory=LimitsSafetyConfig)
    joint_limits: JointLimitSafetyConfig = Field(default_factory=JointLimitSafetyConfig)
    ik_quality: IkQualitySafetyConfig = Field(default_factory=IkQualitySafetyConfig)
    self_collision: SelfCollisionSafetyConfig = Field(
        default_factory=SelfCollisionSafetyConfig
    )
    payload: PayloadSafetyConfig = Field(default_factory=PayloadSafetyConfig)
    motion_continuity: MotionContinuitySafetyConfig = Field(
        default_factory=MotionContinuitySafetyConfig
    )
    dwell: DwellSafetyConfig = Field(default_factory=DwellSafetyConfig)

