"""Schemas for the vendor-selected robot configuration tree."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from .._base import StrictModel

from .ur_schema import URConfig
from .kuka_schema import KukaConfig, KukaEkiConfig
from .dummy_schema import DummyConfig
from .sim_schema import (
    SimCameraSchema,
    SimConfig,
    SimGateConfig,
    SimMarkerConfig,
    SimObjectConfig,
    SimSceneConfig,
    SimTableConfig,
    SimViewpointConfig,
)
from .calibration_schema import RobotCalibrationConfig, RobotCalibrationQualityBandsMm

from .safety_schema import (
    DwellSafetyConfig,
    FixtureBoxConfig,
    IkQualitySafetyConfig,
    JointLimitSafetyConfig,
    LimitsSafetyConfig,
    MotionContinuitySafetyConfig,
    PayloadSafetyConfig,
    RobotSafetyConfig,
    SelfCollisionSafetyConfig,
)
from .tool_frame_schema import ToolFrameConfig

from .grasping_schema import (
    BlockerGraphSchemaConfig,
    GraspingClosedLoopConfig,
    GraspingDecisionConfig,
    GraspingDenseRecoveryConfig,
    GraspingFeasibilityConfig,
    GraspingOcclusionConfig,
    GraspingOrderingConfig,
    GraspingPerformanceConfig,
    GraspingRecoveryConfig,
    GraspingRecoveryFixtureConfig,
    GraspingSuccessModelConfig,
    GraspingUncertaintyConfig,
    GraspingVerificationConfig,
    GraspingWatchdogConfig,
    RobotGraspingApproachValidationConfig,
    RobotGraspingCommitPolicyConfig,
    RobotGraspingConfig,
    RobotGraspingFusionConfig,
    UncertaintyChannelWeightsConfig,
)

from .rl_schema import (
    RLExperimentalConfig,
    RLRollbackTriggersConfig,
    RL_ACTIVE_MODES,
    RL_MODE_GEOMETRY_ONLY,
    RL_MODE_HYBRID_ML,
    RL_MODE_RL_ACTIVE,
    RL_MODE_RL_EXPERIMENTAL,
    RL_MODE_RL_SHADOW,
    RL_MODE_VALUES,
    RobotRLConfig,
)

__all__ = [
    "BlockerGraphSchemaConfig",
    "DummyConfig",
    "DwellSafetyConfig",
    "FixtureBoxConfig",
    "GraspingClosedLoopConfig",
    "GraspingDecisionConfig",
    "GraspingDenseRecoveryConfig",
    "GraspingFeasibilityConfig",
    "GraspingOcclusionConfig",
    "GraspingOrderingConfig",
    "GraspingPerformanceConfig",
    "GraspingRecoveryConfig",
    "GraspingRecoveryFixtureConfig",
    "GraspingSuccessModelConfig",
    "GraspingUncertaintyConfig",
    "GraspingVerificationConfig",
    "GraspingWatchdogConfig",
    "GripperConfig",
    "IkQualitySafetyConfig",
    "JointLimitSafetyConfig",
    "KukaConfig",
    "KukaEkiConfig",
    "LimitsSafetyConfig",
    "MotionContinuitySafetyConfig",
    "MotionLimitsConfig",
    "PayloadSafetyConfig",
    "RLExperimentalConfig",
    "RLRollbackTriggersConfig",
    "RL_ACTIVE_MODES",
    "RL_MODE_GEOMETRY_ONLY",
    "RL_MODE_HYBRID_ML",
    "RL_MODE_RL_ACTIVE",
    "RL_MODE_RL_EXPERIMENTAL",
    "RL_MODE_RL_SHADOW",
    "RL_MODE_VALUES",
    "RobotCalibrationConfig",
    "RobotCalibrationQualityBandsMm",
    "RobotConfig",
    "RobotGraspingApproachValidationConfig",
    "RobotGraspingCommitPolicyConfig",
    "RobotGraspingConfig",
    "RobotGraspingFusionConfig",
    "RobotRLConfig",
    "RobotSafetyConfig",
    "SafePoseConfig",
    "SelfCollisionSafetyConfig",
    "SimCameraSchema",
    "SimConfig",
    "SimGateConfig",
    "SimMarkerConfig",
    "SimObjectConfig",
    "SimSceneConfig",
    "SimTableConfig",
    "SimViewpointConfig",
    "URConfig",
    "VacuumGripperConfig",
    "UncertaintyChannelWeightsConfig",
    "WorkspaceLimitsConfig",
]


class MotionLimitsConfig(StrictModel):
    """Hard upper bounds on commanded velocity / acceleration."""

    max_velocity: float = Field(default=1.0, gt=0.0, le=3.14)
    max_acceleration: float = Field(default=0.5, gt=0.0, le=5.0)


class SafePoseConfig(StrictModel):
    """Cartesian pose the robot can retreat to at operator request."""

    x: float = 0.0
    y: float = -300.0
    z: float = 400.0
    rx: float = 0.0
    ry: float = 3.14159
    rz: float = 0.0
    label: str = "safe"


class WorkspaceLimitsConfig(StrictModel):
    """Cartesian workspace bounding box in millimetres."""

    x_min: float = -500.0
    x_max: float = 500.0
    y_min: float = -500.0
    y_max: float = 500.0
    z_min: float = 0.0
    z_max: float = 500.0

    @model_validator(mode="after")
    def _check_ordering(self) -> WorkspaceLimitsConfig:
        if self.x_min >= self.x_max:
            raise ValueError(f"x_min ({self.x_min}) must be < x_max ({self.x_max})")
        if self.y_min >= self.y_max:
            raise ValueError(f"y_min ({self.y_min}) must be < y_max ({self.y_max})")
        if self.z_min >= self.z_max:
            raise ValueError(f"z_min ({self.z_min}) must be < z_max ({self.z_max})")
        return self


class VacuumGripperConfig(StrictModel):
    """Wiring + timing for a suction end-effector on the controller's digital I/O.

    Only consulted when ``gripper.vendor == "vacuum"``. Every field here is a number someone has to
    MEASURE on the actual cell -- which pin the ejector is on, which bank it lives in, whether a vacuum
    switch is wired and how long that ejector takes to build a seal. Stating them as config is what lets
    the suction path be built and tested before the hardware exists: bring-up becomes measuring, not
    coding. Defaults describe the common case (tool I/O, pin 0, no switch) and are NOT a claim about any
    particular cell.
    """

    #: Output pin that switches the ejector / pump on.
    vacuum_output_pin: int = Field(default=0, ge=0, le=7)
    #: Optional output pulsed on RELEASE. Residual vacuum keeps a light part stuck to the cup after the
    #: ejector stops, so it lets go somewhere unintended; a blow-off pulse pushes it off deliberately.
    blow_off_output_pin: int | None = Field(default=None, ge=0, le=7)
    #: Optional input from a vacuum switch. WITH it the gripper can be ASKED whether it is holding a
    #: part -- real post-close verification, which the jaw path does not have. Without it the driver can
    #: only report what it commanded.
    vacuum_ok_input_pin: int | None = Field(default=None, ge=0, le=7)
    #: Which I/O bank the pins live on. A tool-mounted ejector is usually on the TOOL block.
    io_port: Literal["standard", "configurable", "tool"] = "tool"
    #: How long to wait for the switch to confirm a seal. A timeout is a MISSED GRASP, not a fault.
    engage_timeout_s: float = Field(default=1.0, gt=0.0, le=30.0)
    #: Blow-off pulse length on release.
    blow_off_s: float = Field(default=0.15, ge=0.0, le=5.0)
    #: Commanded width at/below which the driver engages vacuum. Mirrors the simulated cup so the two
    #: interpret the width-based Gripper Protocol identically.
    vacuum_on_below_mm: float = Field(default=5.0, gt=0.0)


class GripperConfig(StrictModel):
    """Physical gripper opening limits.

    The ``vendor`` field selects which driver the robot runtime will
    instantiate. It is validated at config-load against the ``GripperVendor``
    enum: unknown vendors are rejected early with a clear message rather than
    failing late at ``create_gripper``.
    """

    vendor: str = Field(default="robotiq", min_length=1)
    #: Physical opening of the mounted gripper, in mm. Default is the Robotiq 2F-85 (85 mm), the
    #: end-effector this project ships. The Robotiq driver anchors its count map on this value, so it
    #: must be the real physical open width, not a policy ceiling.
    max_width_mm: float = Field(default=85.0, gt=0.0)
    #: Smallest MEANINGFUL grip, a policy floor -- NOT the physical closed width (the 2F-85 closes to
    #: 0 mm). The driver's count map is anchored on 0, so this only clamps commanded widths.
    min_width_mm: float = Field(default=5.0, ge=0.0)
    #: The PHYSICAL closed width: what ``get_width_mm()`` reads when the jaws are shut on nothing.
    #: Default 0.0 = the Robotiq 2F-85 (fingers touching).
    #:
    #: Separate from ``min_width_mm`` on purpose, and the separation is load-bearing. MEASURED
    #: 2026-08-09: ``WidthDeltaGripperVerifier`` -- the only grasp verifier a jaw cell can use, since
    #: the Robotiq driver exposes no object-detection capability -- decides "the jaws collapsed on
    #: nothing" as ``post_close <= min_width + width_delta_min_mm``. Reading the POLICY floor (5.0)
    #: there put that threshold at 7 mm, so a genuinely held 6 mm part was reported as an EMPTY grasp.
    #: Reading the physical 0.0 makes the same case pass. This repo already untangled the identical
    #: confusion once for the driver's count map (robotiq.py) -- the verifier was reading the other one.
    closed_width_mm: float = Field(default=0.0, ge=0.0)
    #: Wiring for a suction end-effector; inert unless ``vendor == "vacuum"``.
    vacuum: VacuumGripperConfig = Field(default_factory=VacuumGripperConfig)
    #: Flange -> grasp-centre transform for THIS end-effector. Hangs off the gripper, not the robot,
    #: because it is a property of what is bolted on -- and this repo already carries four different
    #: ones and swaps between two of them mid-run. Default ``source: undeclared`` is inert, so every
    #: existing config keeps loading unchanged; the real-arm driver is what refuses it.
    tool_frame: ToolFrameConfig = Field(default_factory=ToolFrameConfig)

    @field_validator("vendor")
    @classmethod
    def _validate_gripper_vendor(cls, v: str) -> str:
        # Lazy import keeps the config layer free of robot-runtime imports at module load (the enum is
        # a pure StrEnum, no SDKs). Coerces case-insensitively to the canonical value; rejects unknowns.
        from backend.src.robot.core.gripper_vendor import GripperVendor

        try:
            return GripperVendor.from_string(v).value
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

    @model_validator(mode="after")
    def _check_ordering(self) -> GripperConfig:
        if self.min_width_mm >= self.max_width_mm:
            raise ValueError(
                f"min_width_mm ({self.min_width_mm}) must be < "
                f"max_width_mm ({self.max_width_mm})"
            )
        return self


class RobotConfig(StrictModel):
    """Top-level robot configuration tree.

    The ``vendor`` field selects which driver the robot runtime will
    instantiate. It defaults to ``"ur"`` for backwards compatibility
    with existing deployments. The config layer intentionally avoids
    importing robot runtime modules so config validation works on hosts
    that only need to edit or lint YAML.

    Vendor-specific transport settings live in named sibling blocks
    (``ur``, ``sim``, ``dummy``, ``kuka``) so each driver owns its own
    typed surface. Only the block matching ``vendor`` is consulted at
    runtime; the others remain valid but unused. The old flat
    ``connection:`` block was removed; migrate to ``ur:`` for UR
    deployments and to ``kuka.controller_ip`` for KUKA.
    """

    vendor: str = Field(default="ur", min_length=1)
    ur: URConfig = Field(default_factory=URConfig)
    sim: SimConfig = Field(default_factory=SimConfig)
    dummy: DummyConfig = Field(default_factory=DummyConfig)
    kuka: KukaConfig = Field(default_factory=KukaConfig)
    motion_limits: MotionLimitsConfig = Field(default_factory=MotionLimitsConfig)
    workspace_limits: WorkspaceLimitsConfig = Field(default_factory=WorkspaceLimitsConfig)
    gripper: GripperConfig = Field(default_factory=GripperConfig)
    #: Joint configuration the arm returns to on ``move_home()``, in radians. ``None`` uses
    #: :data:`HOME_JOINTS_DEFAULT`, which was authored for a UR5e.
    #:
    #: MEASURED 2026-08-09 why a real cell needs its own: that default puts a UR3e's grasp centre at
    #: z = 561.9 mm and r = 466.9 mm (93.4% of its 500 mm reach -- past the 85% this project treats as
    #: near-singular), while the UR3e cell's own ``workspace_limits`` stop at z = 320. So the natural
    #: FIRST motion on a new cell left the declared workspace. The sim has had
    #: ``sim.home_joint_positions`` all along; the real path had no field at all.
    home_joint_positions: tuple[float, ...] | None = None

    safe_pose: SafePoseConfig = Field(default_factory=SafePoseConfig)
    calibration: RobotCalibrationConfig = Field(default_factory=RobotCalibrationConfig)
    safety: RobotSafetyConfig = Field(default_factory=RobotSafetyConfig)
    grasping: RobotGraspingConfig = Field(default_factory=RobotGraspingConfig)
    # RL optimisation extension layer. Defaults preserve production
    # behaviour byte-identically (mode='hybrid_ml', no artifact loaded).
    # RL-active modes are schema-gated *and* runtime-rejected; see
    # backend/src/robot/grasping/rl.
    rl: RobotRLConfig = Field(default_factory=RobotRLConfig)

    @model_validator(mode="after")
    def _check_self_collision_model_matches_sim_robot(self) -> "RobotConfig":
        """On an ENABLED sim cell, ``safety.self_collision.kinematics_model`` must equal ``sim.robot_model``.

        They are independent hand-edited keys, but the self-collision guard feeds ``kinematics_model`` into the
        UR DH table (``safety/_ur_kinematics.py``) to derive per-link transforms. A half-migrated cell (e.g.
        ``sim.robot_model: ur3e`` with ``kinematics_model: "ur5e"`` still in place) would evaluate EVERY motion
        against the wrong link lengths (ur5e a2/a3 = -425/-392.2 mm vs ur3e -243.55/-213.2 mm) and silently
        return wrong self-collision verdicts. Fail loudly at config load instead. ur5e == ur5e today, so this
        changes nothing for existing cells; it is inert when the sim block is disabled (a real-robot config).
        """
        sc = self.safety.self_collision
        if self.sim.enabled and sc.kinematics_model and sc.kinematics_model != self.sim.robot_model:
            raise ValueError(
                f"safety.self_collision.kinematics_model={sc.kinematics_model!r} does not match "
                f"sim.robot_model={self.sim.robot_model!r}. The self-collision guard would derive link "
                "transforms from the WRONG robot's DH chain. Set both to the same model."
            )
        return self

    @model_validator(mode="after")
    def _check_self_collision_model_matches_ur_robot(self) -> "RobotConfig":
        """The same coupling for a REAL UR cell: ``kinematics_model`` must equal ``ur.model``.

        Identical mechanism to the sim check above and a strictly worse consequence, because there is a
        physical arm on the other end: the guard would derive link transforms from another robot's DH
        chain and return wrong self-collision verdicts for real motion. Gated on ``vendor == "ur"`` so a
        sim or dummy cell (whose ``ur`` block is inert boilerplate) is unaffected; ur5e == ur5e today, so
        no existing config changes.
        """
        sc = self.safety.self_collision
        if self.vendor == "ur" and sc.kinematics_model and sc.kinematics_model != self.ur.model:
            raise ValueError(
                f"safety.self_collision.kinematics_model={sc.kinematics_model!r} does not match "
                f"ur.model={self.ur.model!r}. On real hardware the self-collision guard would derive link "
                "transforms from the WRONG robot's DH chain. Set both to the same model."
            )
        return self

    @model_validator(mode="after")
    def _check_self_collision_model_is_meaningful_for_this_vendor(self) -> "RobotConfig":
        """``kinematics_model`` selects a UR DH table, so it means nothing on a non-UR, non-sim cell.

        The two rules above couple the key to ``sim.robot_model`` and to ``ur.model``. Neither fires when
        the cell is neither -- and profiles COMPOSE, so that gap is reachable: ``WILLY_PROFILE=sim,web``
        loads cleanly today with ``vendor='kuka'`` and ``kinematics_model='ur5e'``, and the guard then
        evaluates a KUKA arm against UR5e link lengths. That is precisely the failure the sibling rules
        exist to prevent, arriving through the one door they do not watch.

        Only the bundled UR tables exist, so there is no correct value here for another vendor: the
        honest configuration is to leave it unset and let the guard use the capsule path (or fail closed
        if the mesh backend was demanded). Unset is always accepted, so this changes nothing for any
        existing cell.
        """
        model = self.safety.self_collision.kinematics_model
        if model and self.vendor not in ("ur", "sim"):
            raise ValueError(
                f"safety.self_collision.kinematics_model={model!r} is set on a vendor={self.vendor!r} "
                f"cell. That key selects a UNIVERSAL ROBOTS DH table; there is none for this vendor, so "
                f"the self-collision guard would evaluate this arm against another robot's link lengths. "
                f"Leave it unset."
            )
        return self

    @model_validator(mode="after")
    def _check_safe_pose_in_workspace(self) -> "RobotConfig":
        """The safe (retreat) pose must lie inside ``workspace_limits``.

        The ``SafetyPreflight`` WorkspaceGuard rejects ANY motion whose target is outside
        ``workspace_limits``, so a safe_pose outside the box would make the retreat-to-safe motion
        itself fail closed. (NB: safe_pose carries rx/ry/rz euler while poses elsewhere are XYZW
        quaternions — a representation mismatch the driver boundary converts; only position is gated here.)
        """
        sp, wl = self.safe_pose, self.workspace_limits
        if not (wl.x_min <= sp.x <= wl.x_max
                and wl.y_min <= sp.y <= wl.y_max
                and wl.z_min <= sp.z <= wl.z_max):
            raise ValueError(
                f"safe_pose ({sp.x}, {sp.y}, {sp.z}) mm is outside workspace_limits "
                f"x[{wl.x_min}, {wl.x_max}] y[{wl.y_min}, {wl.y_max}] z[{wl.z_min}, {wl.z_max}] — "
                "the WorkspaceGuard would reject the retreat-to-safe motion."
            )
        return self

    @field_validator("vendor")
    @classmethod
    def _validate_vendor(cls, v: str) -> str:
        """Validate ``vendor`` against the ``RobotVendor`` enum at config-load (reject typos
        like 'ur5' early instead of failing late at ``create_arm``)."""
        from backend.src.robot.core.vendor import RobotVendor

        try:
            return RobotVendor.from_string(v).value
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

