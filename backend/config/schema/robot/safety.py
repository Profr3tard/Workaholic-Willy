"""Vendor-neutral safety-guard config schema (R7.1 split from robot_schema.py)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .._base import StrictModel


class LimitsSafetyConfig(StrictModel):
    """Workspace-margin safety config.

    ``workspace_margin_mm`` is consumed by :class:`SafetyPreflight.from_safety_config` to shrink the
    ``workspace_limits`` box on every face before the workspace guard runs.

    Deliberately does NOT declare velocity/acceleration ceilings: real per-move vel/acc come from the
    consumed :class:`MotionLimitsConfig` (``config.motion_limits``). Hardware speed-limiting belongs there
    (a ``min(motion_limits, safety)`` clamp at the UR/KUKA boundary), not a second dead surface here.
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
    link_radii_mm: list[float] | None = Field(default=None)
    fixtures: list[FixtureBoxConfig] = Field(default_factory=list)
    mesh_dir: str | None = Field(default=None)

    # Which bundled arm kinematics (DH) table to use for the per-link arm-vs-arm capsules. When
    # ``None`` (default) only a real ``vendor == "ur"`` arm gets arm-vs-arm capsules (keyed by its own
    # model); set e.g. ``"ur5e"`` to let a NON-UR vendor (the Isaac sim, which is physically a UR5e) opt
    # into the UR DH so it gets real arm-vs-arm self-collision instead of only base+tool+fixture checks.
    kinematics_model: str | None = Field(default=None)

    # When a NON-default gripper is MOUNTED (its collision geometry differs from the baked Robotiq 2F-85),
    # this names the per-gripper fcl/Coal mesh bundle the backend loads (``{variant}_collision_meshes.npz``)
    # instead of ``{kinematics_model}_...``. The arm kinematics stay ``kinematics_model`` (the variant bundle
    # copies the arm-link meshes + swaps only the gripper meshes). ``None`` (default) = the kinematics_model
    # bundle (the 2F-85). The sim threads this from ``robot.sim.gripper_mount``.
    collision_mesh_variant: str | None = Field(default=None)

    # Yaw (degrees) of the bundled-DH base frame relative to the robot/system base frame in which
    # poses and fixtures are expressed. The official UR DH (``_ur_kinematics.py``) base is rotated 180 deg
    # about Z from the Isaac UR5e USD ``base_link`` (measured on-box 2026-06-26: ``ur_link_origins_mm`` ==
    # ``[-x, -y, z]`` vs the Lula FK ground truth at every pose). The guard rotates the DH-derived arm-link
    # origins by this yaw so the arm capsules line up with the base/tool/fixture capsules (all already in the
    # system frame). Default 0.0 == no rotation == byte-identical for every existing cell — including the real
    # ``vendor == "ur"`` path, whose own base-yaw MUST be validated on hardware before relying on
    # arm-vs-fixture capsules (the arm self-collision path has never been exercised on real UR). Only the
    # arm-link capsules are affected; arm-vs-arm distance is rotation-invariant, so this knob cannot change it.
    kinematics_base_yaw_deg: float = Field(default=0.0, ge=-360.0, le=360.0)

    # Tool/base capsule geometry (mm). Defaults match the module constants so existing cells are
    # unchanged; override to declare the real mounted tool (e.g. the Robotiq 2F-85) so tool-vs-link and
    # tool-vs-fixture distances are honest.
    tool_length_mm: float = Field(default=150.0, gt=0.0, le=1000.0)
    tool_radius_mm: float = Field(default=70.0, gt=0.0, le=500.0)
    base_radius_mm: float = Field(default=80.0, gt=0.0, le=1000.0)
    base_height_mm: float = Field(default=150.0, gt=0.0, le=2000.0)

    # The tool collision model. ``capsule`` (default) is the legacy single capsule along the tool
    # APPROACH axis (R[:, 2]) with radius ``tool_radius_mm`` -- a rotation-invariant bounding cylinder. For
    # a parallel-jaw gripper that is over-conservative against a bin wall: a 2F-85 reaching into a KLT is
    # ~27 mm wide perpendicular to its closing axis (measured on-box from the Robotiq_2f_85 USD) but the
    # r=70 cylinder claims 140 mm in EVERY direction, so an off-center target is falsely wall-rejected.
    # ``finger`` instead models the descending fingers as a thin capsule ALONG the grasp's CLOSING axis
    # (R[:, 0]; the [closing, binormal, approach] convention of execution_policy._quaternion_from_axes),
    # length ``tool_finger_span_mm`` (fingertip-to-fingertip at full open), radius ``tool_finger_radius_mm``
    # (the measured perpendicular half-width). This is rotation-AWARE: it clears the narrow wall the thin
    # side faces and still rejects when the open-span faces it -- the honest, yaw-dependent footprint.
    # Default ``capsule`` == byte-identical to every existing cell.
    tool_model: Literal["capsule", "finger"] = Field(default="capsule")
    tool_finger_radius_mm: float = Field(default=16.0, gt=0.0, le=200.0)
    tool_finger_span_mm: float = Field(default=150.0, gt=0.0, le=500.0)


class PayloadSafetyConfig(StrictModel):
    """Tool / payload envelope.

    Validated at the point the operator updates the payload (e.g. by
    calling ``URRobotArm.set_payload`` or by editing this YAML and
    reloading). Per-move evaluation is cheap: it just verifies the
    *currently configured* payload still fits the envelope.

    ``cog_mm`` is the centre-of-gravity offset from the flange in mm.
    ``inertia_kgm2`` is the diagonal inertia tensor (Ixx, Iyy, Izz) in
    kg*m^2. ``max_mass_kg`` is the absolute upper bound the guard
    refuses to set, regardless of the configured ``mass_kg``.
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
    arm across the workspace. The guard caches the *last accepted*
    target inside :class:`SafetyPreflight`; the first command after a
    :meth:`SafetyPreflight.reset` is always accepted by this guard
    because there is no previous target to diff against.
    """

    enforce: bool = Field(default=True)
    max_joint_step_deg: float = Field(default=45.0, gt=0.0, le=180.0)
    max_orientation_step_deg: float = Field(default=30.0, gt=0.0, le=180.0)
    max_tcp_step_mm: float = Field(default=250.0, gt=0.0, le=2000.0)


class DwellSafetyConfig(StrictModel):
    """Post-Stop dwell + steady-state gating.

    Distinct from the other safety guards because it is a *temporal* check (does the controller report
    steady state?) rather than a per-target spatial check, so it lives OUTSIDE the per-move
    :class:`SafetyPreflight` pipeline.

    ``require_steady_before_motion`` + ``steady_timeout_s`` ARE consumed by
    :class:`GraspExecutionPolicy` — before every commanded approach/grasp/retreat move the policy blocks
    on ``arm.wait_until_steady(steady_timeout_s)`` and FAILS CLOSED on timeout. ``dwell_after_stop_s`` is
    currently NOT enforced — there is no exercised Stop/E-stop path in the runtime (deferred).
    """

    dwell_after_stop_s: float = Field(default=0.5, ge=0.0, le=10.0)
    require_steady_before_motion: bool = Field(default=True)
    steady_timeout_s: float = Field(default=5.0, gt=0.0, le=60.0)


class RobotSafetyConfig(StrictModel):
    """Vendor-neutral safety surface.

    Every safety knob lives in a dedicated sub-block that owns its own ``enforce`` flag so operators can
    disable individual guards without touching the others.

    Sub-blocks
    ----------
    * :attr:`limits` -- the ``workspace_margin_mm`` consumed by the workspace guard.
    * :attr:`joint_limits` -- per-axis joint hard limits + margin.
    * :attr:`ik_quality` -- IK-solution quality checks.
    * :attr:`motion_continuity` -- step-size between consecutive commanded targets.
    * :attr:`payload` -- mass / CoG / inertia envelope.
    * :attr:`self_collision` -- link-link / link-fixture collision.
    * :attr:`dwell` -- post-Stop dwell + steady-state gating.
    * :attr:`emergency_stop_source` -- which transport delivers the E-stop signal (``"none"`` disables
      external checks; the hardware E-stop is always honoured by the controller).
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

    emergency_stop_source: Literal["none", "controller", "external_io"] = Field(
        default="controller"
    )
