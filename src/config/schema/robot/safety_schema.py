"""Vendor-neutral safety-guard config schema."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .._base import StrictModel


class LimitsSafetyConfig(StrictModel):
    """Workspace-margin safety config.

    ``workspace_margin_mm`` is consumed by :class:`SafetyPreflight.from_safety_config` to shrink the
    ``workspace_limits`` box on every face before the workspace guard runs.

    No velocity or acceleration ceiling lives here. Per-move vel/acc come from
    :class:`MotionLimitsConfig` (``config.motion_limits``), and hardware speed-limiting belongs there
    too, as a ``min(motion_limits, safety)`` clamp at the UR/KUKA boundary.
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
    # lists must be the same length as the arm's DoF and ``min_deg[i]
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

    The guard runs after the driver has produced a joint solution and
    rejects:

    * NaN / wrong-DoF solutions outright,
    * joint solutions that differ from the current pose by more than
      ``max_jump_rad`` on any axis (large IK jumps signal wrap-around
      ambiguities),
    * solutions whose Jacobian condition number exceeds
      ``max_condition_number`` or whose smallest singular value falls
      below ``min_singular_value`` (near-singular configurations),
    * solutions inside ``limit_proximity_deg`` of the configured joint
      limits, which the joint-limit guard also covers when enabled, so
      this one still catches them when it is disabled.
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

    ``link_radii_mm`` is the per-link capsule radius in mm; when
    omitted the guard uses a single default radius supplied at runtime.
    Its length is not validated here because DoF is a driver-side
    property: the guard checks it at evaluation time and reports
    :attr:`SafetyReason.UNAVAILABLE` on mismatch.
    """

    enforce: bool = Field(default=True)
    backend: Literal["capsule", "fcl"] = Field(default="fcl")
    min_distance_mm: float = Field(default=10.0, ge=0.0, le=500.0)
    #: Clearance (mm) handed to the trajectory planner so it stops proposing configurations this guard
    #: will reject: cuRobo plans against a sphere model, the guard re-checks the exact meshes, and the
    #: two disagree. cuRobo returned UR5e plans at 9.44-9.47 mm against this guard's 10.000 mm, losing
    #: 3 of 10 picks to what looks like bad grasping. Not to be derived from ``min_distance_mm``: the
    #: margin a planner can absorb depends on how tightly its spheres fit that robot. A UR5e plans fine
    #: at 10 mm; a UR3e plans fine to 6 mm and finds no plan at all at 10 mm, its thinner links reading
    #: as permanent self-collision, which takes a UR3e cell from 10/10 to 0/10. 0.0 (default) leaves
    #: the planner's own config untouched.
    planner_margin_mm: float = Field(default=0.0, ge=0.0, le=100.0)
    link_radii_mm: list[float] | None = Field(default=None)
    fixtures: list[FixtureBoxConfig] = Field(default_factory=list)
    mesh_dir: str | None = Field(default=None)

    # Which bundled arm kinematics (DH) table to use for the per-link arm-vs-arm capsules. When
    # ``None`` (default) only a real ``vendor == "ur"`` arm gets arm-vs-arm capsules, keyed by its own
    # model. Setting e.g. ``"ur5e"`` lets a non-UR vendor (the Isaac sim, physically a UR5e) opt into the
    # UR DH and so get real arm-vs-arm self-collision instead of only base+tool+fixture checks.
    kinematics_model: str | None = Field(default=None)

    # Names the per-gripper fcl/Coal mesh bundle the backend loads (``{variant}_collision_meshes.npz``)
    # instead of ``{kinematics_model}_...``, for a mounted gripper whose collision geometry differs from
    # the baked Robotiq 2F-85. The arm kinematics stay ``kinematics_model``: the variant bundle copies the
    # arm-link meshes and swaps only the gripper meshes. ``None`` (default) is the kinematics_model bundle,
    # the 2F-85. The sim threads this from ``robot.sim.gripper_mount``.
    collision_mesh_variant: str | None = Field(default=None)

    # Yaw (degrees) of the bundled-DH base frame relative to the robot/system base frame that poses and
    # fixtures are expressed in. The official UR DH (``_ur_kinematics.py``) base is rotated 180 deg about
    # Z from the Isaac UR5e USD ``base_link``: ``ur_link_origins_mm`` == ``[-x, -y, z]`` against the Lula
    # FK ground truth at every pose. The guard rotates the DH-derived arm-link origins by this yaw so the
    # arm capsules line up with the base/tool/fixture capsules, which are already in the system frame.
    # Only the arm-link capsules move, and arm-vs-arm distance is rotation-invariant, so this knob cannot
    # change it. Default 0.0 is no rotation. The arm self-collision path has never run on real UR, so a
    # real ``vendor == "ur"`` cell must have its base yaw validated on hardware before arm-vs-fixture
    # capsules are relied on.
    kinematics_base_yaw_deg: float = Field(default=0.0, ge=-360.0, le=360.0)

    # Tool/base capsule geometry (mm). Defaults match the module constants, so a cell that sets none of
    # them is unchanged. Override them to declare the real mounted tool (e.g. the Robotiq 2F-85) and get
    # honest tool-vs-link and tool-vs-fixture distances.
    tool_length_mm: float = Field(default=150.0, gt=0.0, le=1000.0)
    tool_radius_mm: float = Field(default=70.0, gt=0.0, le=500.0)
    base_radius_mm: float = Field(default=80.0, gt=0.0, le=1000.0)
    base_height_mm: float = Field(default=150.0, gt=0.0, le=2000.0)

    # The tool collision model. ``capsule`` (default) is one capsule along the tool approach axis
    # (R[:, 2]) with radius ``tool_radius_mm``, a rotation-invariant bounding cylinder. For a parallel-jaw
    # gripper that is over-conservative against a bin wall: a 2F-85 reaching into a KLT is ~27 mm wide
    # perpendicular to its closing axis (from the Robotiq_2f_85 USD) while the r=70 cylinder claims 140 mm
    # in every direction, so an off-center target is falsely wall-rejected. ``finger`` instead models the
    # descending fingers as a thin capsule along the grasp's closing axis (R[:, 0]; the
    # [closing, binormal, approach] convention of execution_policy._quaternion_from_axes), of length
    # ``tool_finger_span_mm`` (fingertip-to-fingertip at full open) and radius ``tool_finger_radius_mm``
    # (the perpendicular half-width). Being rotation-aware, it clears the narrow wall the thin side faces
    # and still rejects when the open span faces it. Default ``capsule`` is byte-identical to every cell.
    tool_model: Literal["capsule", "finger"] = Field(default="capsule")
    tool_finger_radius_mm: float = Field(default=16.0, gt=0.0, le=200.0)
    tool_finger_span_mm: float = Field(default=150.0, gt=0.0, le=500.0)


class PayloadSafetyConfig(StrictModel):
    """Tool / payload envelope.

    Validated at the point the operator updates the payload (e.g. by
    calling ``URRobotArm.set_payload`` or by editing this YAML and
    reloading). Per-move evaluation only verifies that the currently
    configured payload still fits the envelope.

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
    arm across the workspace. The guard caches the last accepted target
    inside :class:`SafetyPreflight`, so the first command after a
    :meth:`SafetyPreflight.reset` is always accepted: there is no
    previous target to diff against.
    """

    enforce: bool = Field(default=True)
    max_joint_step_deg: float = Field(default=45.0, gt=0.0, le=180.0)
    max_orientation_step_deg: float = Field(default=30.0, gt=0.0, le=180.0)
    max_tcp_step_mm: float = Field(default=250.0, gt=0.0, le=2000.0)


class DwellSafetyConfig(StrictModel):
    """Post-Stop dwell + steady-state gating.

    A temporal check (does the controller report steady state?) rather than a per-target spatial one,
    so it lives outside the per-move :class:`SafetyPreflight` pipeline.

    ``require_steady_before_motion`` + ``steady_timeout_s`` are consumed by
    :class:`GraspExecutionPolicy`: before every commanded approach/grasp/retreat move the policy blocks
    on ``arm.wait_until_steady(steady_timeout_s)`` and fails closed on timeout. ``dwell_after_stop_s``
    is not enforced; there is no exercised Stop/E-stop path in the runtime.
    """

    require_steady_before_motion: bool = Field(default=True)
    steady_timeout_s: float = Field(default=5.0, gt=0.0, le=60.0)


class SupportPlaneConfig(StrictModel):
    """The bench, table or floor the cell stands on, as one axis-aligned slab in the base frame.

    ``height_mm`` is the top surface, the number an operator can measure: put a rule on the bench and
    read the height above the robot's base plate. The slab is built downwards from there by
    ``thickness_mm``, so raising the thickness never moves the surface the arm must stay above.

    A slab at the bench surface makes the planner refuse low top-down reaches, which is why the sim
    runners closest to a production pick sink their floor to a top of ``-50 mm`` and carry no walls.
    """

    height_mm: float = Field(default=0.0)
    extent_mm: tuple[float, float] = (2000.0, 2000.0)
    thickness_mm: float = Field(default=50.0, gt=0.0, le=2000.0)

    @model_validator(mode="after")
    def _check_extent(self) -> SupportPlaneConfig:
        for axis, extent in zip(("x", "y"), self.extent_mm):
            if extent <= 0.0:
                raise ValueError(
                    f"support_plane.extent_mm.{axis} must be > 0; got {extent}. A zero-extent slab is "
                    "not a table, it is a plane the planner cannot collide with."
                )
        return self


class TrajectoryCheckConfig(StrictModel):
    """Whether a planned path is checked configuration by configuration before any of it is commanded.

    The one-shot guards judge where a move ends, so a plan that grazes a fixture in the middle and
    lands clear passes them. Nothing checks the middle while this is disabled: the sim applies each
    waypoint straight to the articulation and the real UR moveJ's them in turn, so the planner is asked
    for a collision-free path and then trusted to have produced one.

    The exact-mesh backend costs about 9.6 ms per configuration, so a 30 to 100 waypoint plan adds
    roughly 0.3 to 1.0 s before the arm starts moving. That is paid once per move, before motion, not
    interleaved with control.
    """

    enabled: bool = Field(default=False)
    stride: int = Field(default=1, ge=1, le=64)


class AttachedPayloadConfig(StrictModel):
    """Whether the planner is told that the gripper is carrying something.

    The planner's collision model ends at the gripper, so every transit, lift, place and retreat after
    a successful close is planned as if the hand were empty. On a cell carrying a part out of a bin
    that part is the geometry most likely to meet a wall.

    It blocks on geometry, not on principle: against the real planner (ur5e, a wall at x = 250 mm) a
    300 x 300 x 50 mm plate turns a 61-waypoint plan into no plan, while a 20 mm cube still plans.

    A box is not the part. The lateral extents come from the jaw opening at the grasp, which measures
    the part at the grasp line and says nothing about the rest of it, and ``length_mm`` is a declared
    worst case rather than a measurement.
    """

    enabled: bool = Field(default=False)
    sphere_slots: int = Field(default=16, ge=4, le=128)
    length_mm: float = Field(default=120.0, gt=0.0, le=2000.0)
    lateral_margin_mm: float = Field(default=10.0, ge=0.0, le=500.0)


class PlanningWorldConfig(StrictModel):
    """What the trajectory planner is told about the cell, as axis-aligned boxes in the base frame.

    Separate from the guard's own fixture list only in what consumes it: the guard checks a single
    commanded configuration, the planner shapes the whole path. ``include_fixtures`` keeps them one
    declaration, so a bin wall added for the guard is a bin wall the planner routes around.

    The planner's world model is boxes and nothing else. It has no mesh, point-cloud or voxel channel,
    so perceived geometry cannot reach it and anything that is not box-shaped has to be enclosed by one.
    """

    enabled: bool = Field(default=False)
    support_plane: SupportPlaneConfig | None = Field(default=None)
    payload: AttachedPayloadConfig = Field(default_factory=AttachedPayloadConfig)
    include_fixtures: bool = Field(default=True)
    require_registration: bool = Field(default=True)

    @model_validator(mode="after")
    def _check_support_declared(self) -> PlanningWorldConfig:
        if self.enabled and self.support_plane is None:
            raise ValueError(
                "safety.planning_world.enabled is true but no support_plane is declared. Registering a "
                "world replaces the planner's own, boot table included, so enabling this without a "
                "bench would remove the only surface the planner currently knows about. Declare the "
                "bench you measured, or leave the block disabled."
            )
        return self


class RobotSafetyConfig(StrictModel):
    """Vendor-neutral safety surface.

    Every safety knob lives in a dedicated sub-block that owns its own ``enforce`` flag so operators can
    disable individual guards without touching the others.

    Sub-blocks
    ----------
    * :attr:`limits`: the ``workspace_margin_mm`` consumed by the workspace guard.
    * :attr:`joint_limits`: per-axis joint hard limits + margin.
    * :attr:`ik_quality`: IK-solution quality checks.
    * :attr:`motion_continuity`: step-size between consecutive commanded targets.
    * :attr:`payload`: mass / CoG / inertia envelope.
    * :attr:`self_collision`: link-link / link-fixture collision.
    * :attr:`dwell`: post-Stop dwell + steady-state gating.
    * :attr:`planning_world`: the boxes the trajectory planner routes around.
    * :attr:`trajectory_check`: gate every configuration of a planned path, not only its end.

    Not here: the emergency stop. It is a hardware and controller function, and nothing in this package
    can enable, disable, route or observe it.
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
    planning_world: PlanningWorldConfig = Field(default_factory=PlanningWorldConfig)
    trajectory_check: TrajectoryCheckConfig = Field(default_factory=TrajectoryCheckConfig)

