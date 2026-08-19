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
    MEASURE on the actual cell which pin the ejector is on, which bank it lives in, whether a vacuum
    switch is wired and how long that ejector takes to build a seal.
    """

    #: Output pin that switches the ejector / pump on.
    vacuum_output_pin: int = Field(default=0, ge=0, le=7)
    #: Optional output pulsed on RELEASE. Residual vacuum keeps a light part stuck to the cup after the
    #: ejector stops, so it lets go somewhere unintended; a blow-off pulse pushes it off deliberately.
    blow_off_output_pin: int | None = Field(default=None, ge=0, le=7)
    #: Optional input from a vacuum switch. WITH it the gripper can be ASKED whether it is holding a
    #: part, real post-close verification, which the jaw path does not have. Without it the driver can
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


class JawIOGripperConfig(StrictModel):
    """Wiring + timing for a parallel-jaw gripper on the controller's digital I/O.

    Only consulted when ``gripper.vendor == "jaw_io"``. Same design rule as
    :class:`VacuumGripperConfig`, and for the same reason: every field here is a number somebody has
    to MEASURE on the actual cell, which pin closes the jaws, whether the reed switch is
    active-high, how long the cylinder takes to travel.

    A digital-I/O jaw is BINARY. It cannot travel to 40 mm, so the width-based ``Gripper`` Protocol
    is reinterpreted exactly as the suction driver reinterprets it see ``closed_below_mm``.
    """

    #: How the jaws are driven.
    #:
    #: ``single_solenoid`` -- ONE output: high = close, low = open (spring return). The common
    #: pneumatic case. On power loss the spring opens and a held part FALLS.
    #:
    #: ``double_solenoid`` -- TWO outputs, pulsed: one closes, the other opens, and the valve is
    #: bistable. On power loss it HOLDS the part, which is the safer failure, but the jaw state is
    #: then not inferable from the outputs, so feedback pins matter more here.
    actuation: Literal["single_solenoid", "double_solenoid"] = "single_solenoid"
    #: Output that closes the jaws. Held high while closed (single) or pulsed (double).
    close_output_pin: int = Field(default=0, ge=0, le=7)
    #: Output that opens the jaws. REQUIRED for ``double_solenoid``; unused (and must stay unset) for
    #: ``single_solenoid``, where "open" is simply dropping ``close_output_pin``.
    open_output_pin: int | None = Field(default=None, ge=0, le=7)
    #: Pulse length for ``double_solenoid``.
    pulse_s: float = Field(default=0.2, gt=0.0, le=5.0)
    #: Optional input from a dedicated part-present sensor. Simplest feedback: one pin, read directly.
    part_present_input_pin: int | None = Field(default=None, ge=0, le=7)
    #: Optional reed switch that reads TRUE when the jaws are FULLY CLOSED.
    closed_confirm_input_pin: int | None = Field(default=None, ge=0, le=7)
    #: Optional reed switch that reads TRUE when the jaws are FULLY OPEN. With both switches wired the
    #: driver can tell "closed on nothing" from "closed on a part" (neither switch active = the jaws
    #: stopped in between = something is between them), which is a better post-grasp signal than the
    #: jaw path has ever had.
    open_confirm_input_pin: int | None = Field(default=None, ge=0, le=7)
    #: Which I/O bank the pins live on. A tool-mounted gripper is usually on the TOOL block.
    io_port: Literal["standard", "configurable", "tool"] = "tool"
    #: How long to wait for the jaws to reach a settled state after a close.
    close_timeout_s: float = Field(default=1.0, gt=0.0, le=30.0)
    #: Fixed travel wait used when NO feedback pin is wired (there is nothing to poll, so the driver
    #: can only wait). Also the settle pause after the jaws leave the open switch.
    close_settle_s: float = Field(default=0.3, ge=0.0, le=10.0)
    #: Commanded width at/below which the driver CLOSES.
    closed_below_mm: float = Field(default=5.0, gt=0.0)
    open_on_connect_without_feedback: bool = False

    @model_validator(mode="after")
    def _check_actuation_pins(self) -> "JawIOGripperConfig":
        if self.actuation == "double_solenoid" and self.open_output_pin is None:
            raise ValueError(
                "gripper.jaw_io: actuation 'double_solenoid' needs `open_output_pin`, a bistable "
                "valve has no spring to open it, so without that pin the jaws could never release."
            )
        if self.actuation == "single_solenoid" and self.open_output_pin is not None:
            raise ValueError(
                "gripper.jaw_io: actuation 'single_solenoid' opens by DROPPING `close_output_pin`, "
                "so `open_output_pin` is never driven. Remove it, or set actuation to "
                "'double_solenoid' if the valve really has two coils."
            )
        pins = [self.close_output_pin, self.open_output_pin]
        if self.open_output_pin is not None and self.close_output_pin == self.open_output_pin:
            raise ValueError(
                f"gripper.jaw_io: close_output_pin and open_output_pin are both {pins[0]}, one pin "
                "cannot drive both coils."
            )
        return self


class GripperConfig(StrictModel):
    """Physical gripper opening limits.

    The ``vendor`` field selects which driver the robot runtime will
    instantiate. It is validated at config-load against the ``GripperVendor``
    enum: unknown vendors are rejected early with a clear message rather than
    failing late at ``create_gripper``.
    """

    vendor: str = Field(default="robotiq", min_length=1)
    max_width_mm: float = Field(default=85.0, gt=0.0)
    #: Smallest MEANINGFUL grip, a policy floor
    min_width_mm: float = Field(default=5.0, ge=0.0)
    closed_width_mm: float = Field(default=0.0, ge=0.0)
    vacuum: VacuumGripperConfig = Field(default_factory=VacuumGripperConfig)
    jaw_io: JawIOGripperConfig = Field(default_factory=JawIOGripperConfig)
    tool_frame: ToolFrameConfig = Field(default_factory=ToolFrameConfig)

    @field_validator("vendor")
    @classmethod
    def _validate_gripper_vendor(cls, v: str) -> str:
        # Lazy import keeps the config layer free of robot-runtime imports at module load (the enum is
        # a pure StrEnum, no SDKs). Coerces case-insensitively to the canonical value; rejects unknowns.
        from src.robot.core.gripper_vendor import GripperVendor

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
    runtime; the others remain valid but unused.
    """

    vendor: str = Field(default="ur", min_length=1)
    ur: URConfig = Field(default_factory=URConfig)
    sim: SimConfig = Field(default_factory=SimConfig)
    dummy: DummyConfig = Field(default_factory=DummyConfig)
    kuka: KukaConfig = Field(default_factory=KukaConfig)
    motion_limits: MotionLimitsConfig = Field(default_factory=MotionLimitsConfig)
    workspace_limits: WorkspaceLimitsConfig = Field(default_factory=WorkspaceLimitsConfig)
    gripper: GripperConfig = Field(default_factory=GripperConfig)
    home_joint_positions: tuple[float, ...] | None = None

    safe_pose: SafePoseConfig = Field(default_factory=SafePoseConfig)
    calibration: RobotCalibrationConfig = Field(default_factory=RobotCalibrationConfig)
    safety: RobotSafetyConfig = Field(default_factory=RobotSafetyConfig)
    grasping: RobotGraspingConfig = Field(default_factory=RobotGraspingConfig)
    # RL optimisation extension layer. Defaults preserve production
    # behaviour byte-identically (mode='hybrid_ml', no artifact loaded).
    # RL-active modes are schema-gated *and* runtime-rejected; see
    # src/robot/grasping/rl.
    rl: RobotRLConfig = Field(default_factory=RobotRLConfig)

    @model_validator(mode="after")
    def _check_self_collision_model_matches_sim_robot(self) -> "RobotConfig":
        """On an ENABLED sim cell, ``safety.self_collision.kinematics_model`` must equal ``sim.robot_model``.
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
        """The same coupling for a REAL UR cell: ``kinematics_model`` must equal ``ur.model``."""
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
        """``kinematics_model`` selects a UR DH table, so it means nothing on a non-UR, non-sim cell."""
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
        """The safe (retreat) pose must lie inside ``workspace_limits``."""
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
        from src.robot.core.vendor import RobotVendor

        try:
            return RobotVendor.from_string(v).value
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

