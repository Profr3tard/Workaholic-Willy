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
    AttachedPayloadConfig,
    PlanningWorldConfig,
    SelfCollisionSafetyConfig,
    SupportPlaneConfig,
    TrajectoryCheckConfig,
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
    "AttachedPayloadConfig",
    "PlanningWorldConfig",
    "SelfCollisionSafetyConfig",
    "SupportPlaneConfig",
    "TrajectoryCheckConfig",
    "SimCameraSchema",
    "SimConfig",
    "SimGateConfig",
    "SimMarkerConfig",
    "SimObjectConfig",
    "SimSceneConfig",
    "SimTableConfig",
    "SimViewpointConfig",
    "URConfig",
    "OnRobotGripperConfig",
    "VacuumGripperConfig",
    "UncertaintyChannelWeightsConfig",
    "WorkspaceLimitsConfig",
]


class MotionLimitsConfig(StrictModel):
    """Hard upper bounds on commanded velocity / acceleration."""

    max_velocity: float = Field(default=1.0, gt=0.0, le=3.14)
    max_acceleration: float = Field(default=0.5, gt=0.0, le=5.0)


class SafePoseConfig(StrictModel):
    """Cartesian position the robot can retreat to at operator request.

    Position only, deliberately. The single reader is the willy_sim reach doctor
    (`harness/reach.py`), which asks whether the point is inside the arm's envelope. Orientation
    does not enter into that question. If this block is ever made an actual motion target,
    orientation comes back with the code that moves to it.
    """

    x: float = 0.0
    y: float = -300.0
    z: float = 400.0


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
    Measure on the actual cell: which pin the ejector is on, which bank it is in, whether a vacuum
    switch is wired and how long that ejector takes to build a seal. Stating them as config is what lets
    the suction path be built and tested before the hardware exists: bring-up becomes measuring, not
    coding. Defaults describe the common case (tool I/O, pin 0, no switch) and are not a claim about any
    particular cell.
    """

    #: Output pin that switches the ejector / pump on.
    vacuum_output_pin: int = Field(default=0, ge=0, le=7)
    #: Optional output pulsed on release. Residual vacuum keeps a light part stuck to the cup after the
    #: ejector stops, so it lets go somewhere unintended; a blow-off pulse pushes it off deliberately.
    blow_off_output_pin: int | None = Field(default=None, ge=0, le=7)
    #: Optional input from a vacuum switch. With it the gripper can be asked whether it is holding a
    #: part: real post-close verification, which the jaw path lacks. Without it the driver can
    #: only report what it commanded.
    vacuum_ok_input_pin: int | None = Field(default=None, ge=0, le=7)
    #: Which I/O bank the pins live on. A tool-mounted ejector is usually on the TOOL block.
    io_port: Literal["standard", "configurable", "tool"] = "tool"
    #: How long to wait for the switch to confirm a seal. A timeout is a missed GRASP, not a fault.
    engage_timeout_s: float = Field(default=1.0, gt=0.0, le=30.0)
    #: Blow-off pulse length on release.
    blow_off_s: float = Field(default=0.15, ge=0.0, le=5.0)
    #: Commanded width at/below which the driver engages vacuum. Mirrors the simulated cup so the two
    #: interpret the width-based Gripper Protocol identically.
    vacuum_on_below_mm: float = Field(default=5.0, gt=0.0)


class JawIOGripperConfig(StrictModel):
    """Wiring + timing for a parallel-jaw gripper on the controller's digital I/O.

    Only consulted when ``gripper.vendor == "jaw_io"``. Same design rule as
    :class:`VacuumGripperConfig`, and for the same reason. Every field here is a number somebody has
    to measure on the actual cell: which pin closes the jaws, whether the reed switch is
    active-high, how long the cylinder takes to travel. Stating them as config is what lets the jaw
    path be built and tested before the gripper has been bought, so bring-up is measuring rather than
    coding. Defaults describe the common pneumatic case (tool I/O, single solenoid, no feedback) and
    are not a claim about any particular cell.

    A digital-I/O jaw is binary. It cannot travel to 40 mm, so the width-based ``Gripper`` Protocol
    is reinterpreted exactly as the suction driver reinterprets it; see ``closed_below_mm``.
    """

    #: How the jaws are driven.
    #:
    #: ``single_solenoid``: one output, high = close, low = open (spring return). The common
    #: pneumatic case. On power loss the spring opens and a held part falls.
    #:
    #: ``double_solenoid``: two outputs, pulsed. One closes, the other opens, and the valve is
    #: bistable. On power loss it holds the part, which is the safer failure, but the jaw state is
    #: then not inferable from the outputs, so feedback pins matter more here.
    actuation: Literal["single_solenoid", "double_solenoid"] = "single_solenoid"
    #: Output that closes the jaws. Held high while closed (single) or pulsed (double).
    close_output_pin: int = Field(default=0, ge=0, le=7)
    #: Output that opens the jaws. Required for ``double_solenoid``; unused (and must stay unset) for
    #: ``single_solenoid``, where "open" is simply dropping ``close_output_pin``.
    open_output_pin: int | None = Field(default=None, ge=0, le=7)
    #: Pulse length for ``double_solenoid``. A bistable valve latches, so the coil is energised only
    #: long enough to throw it; holding it high is what cooks the coil.
    pulse_s: float = Field(default=0.2, gt=0.0, le=5.0)
    #: Optional input from a dedicated part-present sensor. Simplest feedback: one pin, read directly.
    part_present_input_pin: int | None = Field(default=None, ge=0, le=7)
    #: Optional reed switch that reads TRUE when the jaws are fully closed. This is the interesting
    #: one: fully closed after a close command means the jaws met each other, i.e. an empty grasp.
    closed_confirm_input_pin: int | None = Field(default=None, ge=0, le=7)
    #: Optional reed switch that reads TRUE when the jaws are fully open. With both switches wired the
    #: driver can tell "closed on nothing" from "closed on a part" (neither switch active = the jaws
    #: stopped in between = something is between them), which is a better post-grasp signal than the
    #: jaw path has ever had.
    open_confirm_input_pin: int | None = Field(default=None, ge=0, le=7)
    #: Which I/O bank the pins live on. A tool-mounted gripper is usually on the TOOL block.
    io_port: Literal["standard", "configurable", "tool"] = "tool"
    #: How long to wait for the jaws to reach a settled state after a close. A timeout is a missed
    #: GRASP, not a fault; the verification stage decides, exactly as for suction.
    close_timeout_s: float = Field(default=1.0, gt=0.0, le=30.0)
    #: Fixed travel wait used when NO feedback pin is wired (there is nothing to poll, so the driver
    #: can only wait). Also the settle pause after the jaws leave the open switch.
    close_settle_s: float = Field(default=0.3, ge=0.0, le=10.0)
    #: Commanded width at/below which the driver closes. Mirrors ``vacuum_on_below_mm`` so both I/O
    #: end-effectors interpret the width-based Protocol identically.
    closed_below_mm: float = Field(default=5.0, gt=0.0)
    #: What ``connect()`` does when NO feedback is wired, and it defaults to the cautious answer.
    #:
    #: With feedback the rule is the operator's (2026-08-17): open only when the sensor proves the
    #: jaws are empty, and hold + warn when they are not, rather than dropping an unknown workpiece
    #: wherever the arm happens to be. Without feedback "proven empty" is unreachable, so the same
    #: rule means: do not actuate. Set this True for the suction driver's behaviour instead (assert a
    #: known state on connect and accept that a held part is released).
    open_on_connect_without_feedback: bool = False

    @model_validator(mode="after")
    def _check_actuation_pins(self) -> "JawIOGripperConfig":
        # A double solenoid with no open pin cannot open; it would latch closed on the first grasp
        # and never let go. Refuse at config load rather than at the first release.
        if self.actuation == "double_solenoid" and self.open_output_pin is None:
            raise ValueError(
                "gripper.jaw_io: actuation 'double_solenoid' needs `open_output_pin`; a bistable "
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
                f"gripper.jaw_io: close_output_pin and open_output_pin are both {pins[0]}; one pin "
                "cannot drive both coils."
            )
        return self


class OnRobotGripperConfig(StrictModel):
    """An OnRobot RG2 / RG6 reached over Modbus TCP through the OnRobot Compute Box.

    **The compute box is a separate device with its own address.** The Robotiq branch reuses
    ``robot.ur.ip`` because the URCap daemon runs on the arm's controller; an OnRobot box does not,
    so :attr:`host` is config of its own here. Defaulting it to the arm would point every command at
    a machine that has never heard of it.

    **There is no speed field, and that is not an oversight.** RG2/RG6 have no speed register
    anywhere in the writable map (force, width, control), and OnRobot's own library exposes only a
    read-only ``rg_get_speed``. A key that reached nothing would be worse than its absence.

    **Rg2 and rg6 only.** The 2FG7 shares the family name and not the register map, and no public
    map for it could be sourced; vg10/vgc10 are vacuum and 3FG15 is three-fingered. Configuring one
    of those here would drive an address nobody has verified.
    """

    #: The Compute Box's own IP. Not the robot's, and not to be assumed: the factory default is
    #: 192.168.1.1, but the documented Dynamic IP mode only falls back to it after a 60-second DHCP
    #: timeout, so a box on a DHCP network can be anywhere. Read it from the Web Client.
    host: str = Field(default="192.168.1.1", min_length=1)
    #: Modbus TCP port. OnRobot documents 502 and one concurrent connection.
    port: int = Field(default=502, ge=1, le=65535)
    #: Modbus unit id, which selects the TOOL. It is chosen by the mounting, not by the gripper:
    #: 65 through a Quick Changer or a HEX-E/H QC, 66 for the primary side of a Dual Quick Changer
    #: and 67 for the secondary. An rg2-ft answers on 65 too, with an incompatible map; there is
    #: no way to tell from Modbus alone, so confirm the model in the Web Client before commanding.
    unit_id: int = Field(default=65, ge=1, le=247)
    #: Gripping force in newtons, used when a caller passes none. Native units here: the register is
    #: tenths of a newton, unlike Robotiq's opaque 0-255 count whose physical span differs per model.
    #: 20 N is deliberately gentle; the RG2's range is roughly 3-40 N and the RG6's 25-120 N.
    default_force_n: float = Field(default=20.0, gt=0.0, le=200.0)
    #: Whether commanded widths are interpreted with the configured fingertip offset (control value
    #: 16) or without it (control value 1). Only meaningful when non-standard fingertips are fitted
    #: and their offset has been written to the gripper.
    use_fingertip_offset: bool = False


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
    #: Smallest meaningful grip, a policy floor, not the physical closed width (the 2F-85 closes to
    #: 0 mm). The driver's count map is anchored on 0, so this only clamps commanded widths.
    min_width_mm: float = Field(default=5.0, ge=0.0)
    #: The physical closed width: what ``get_width_mm()`` reads when the jaws are shut on nothing.
    #: Default 0.0 = the Robotiq 2F-85 (fingers touching).
    #:
    #: Separate from ``min_width_mm`` on purpose, and the separation is load-bearing. Measured
    #: 2026-08-09: ``WidthDeltaGripperVerifier``, the only grasp verifier a jaw cell can use since
    #: the Robotiq driver exposes no object-detection capability, decides "the jaws collapsed on
    #: nothing" as ``post_close <= min_width + width_delta_min_mm``. Reading the policy floor (5.0)
    #: there put that threshold at 7 mm, so a genuinely held 6 mm part was reported as an empty grasp.
    #: Reading the physical 0.0 makes the same case pass. This repo already untangled the identical
    #: confusion for the driver's count map (robotiq.py); the verifier was reading the other one.
    closed_width_mm: float = Field(default=0.0, ge=0.0)
    #: Wiring for a suction end-effector; inert unless ``vendor == "vacuum"``.
    vacuum: VacuumGripperConfig = Field(default_factory=VacuumGripperConfig)
    #: Wiring for a parallel-jaw end-effector on the controller's digital I/O; inert unless
    #: ``vendor == "jaw_io"``. Separate from ``vacuum`` so a cell can declare both and swap the
    #: vendor string, which is exactly how a jaw/suction cell is commissioned.
    jaw_io: JawIOGripperConfig = Field(default_factory=JawIOGripperConfig)
    #: An OnRobot RG2/RG6 on a Compute Box; inert unless ``vendor == "onrobot"``. Separate from the
    #: two I/O blocks above for the same reason those are separate from each other: a cell can
    #: declare several and commission by swapping one string.
    onrobot: OnRobotGripperConfig = Field(default_factory=OnRobotGripperConfig)
    #: Flange -> grasp-centre transform for this end-effector. Hangs off the gripper, not the robot,
    #: because it is a property of what is bolted on, and this repo already carries four different
    #: ones and swaps between two of them mid-run. Default ``source: undeclared`` is inert, so every
    #: existing config keeps loading unchanged; the real-arm driver is what refuses it.
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
    #: Measured 2026-08-09 why a real cell needs its own: that default puts a UR3e's grasp centre at
    #: z = 561.9 mm and r = 466.9 mm (93.4% of its 500 mm reach, past the 85% this project treats as
    #: near-singular), while the UR3e cell's own ``workspace_limits`` stop at z = 320. So the natural
    #: First motion on a new cell left the declared workspace. The sim has had
    #: ``sim.home_joint_positions`` all along; the real path had no field at all.
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
        """On an enabled sim cell, ``safety.self_collision.kinematics_model`` must equal ``sim.robot_model``.

        They are independent hand-edited keys, but the self-collision guard feeds ``kinematics_model`` into the
        UR DH table (``safety/_ur_kinematics.py``) to derive per-link transforms. A half-migrated cell (e.g.
        ``sim.robot_model: ur3e`` with ``kinematics_model: "ur5e"`` still in place) would evaluate every motion
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
        """The same coupling for a real ur cell: ``kinematics_model`` must equal ``ur.model``.

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
        the cell is neither; profiles compose, so that gap is reachable: ``WILLY_PROFILE=sim,web``
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

        The ``SafetyPreflight`` WorkspaceGuard rejects any motion whose target is outside
        ``workspace_limits``, so a safe_pose outside the box would make the retreat-to-safe motion
        itself fail closed. (NB: safe_pose carries rx/ry/rz euler while poses elsewhere are XYZW
        quaternions, a representation mismatch the driver boundary converts; only position is
        gated here.)
        """
        sp, wl = self.safe_pose, self.workspace_limits
        if not (wl.x_min <= sp.x <= wl.x_max
                and wl.y_min <= sp.y <= wl.y_max
                and wl.z_min <= sp.z <= wl.z_max):
            raise ValueError(
                f"safe_pose ({sp.x}, {sp.y}, {sp.z}) mm is outside workspace_limits "
                f"x[{wl.x_min}, {wl.x_max}] y[{wl.y_min}, {wl.y_max}] z[{wl.z_min}, {wl.z_max}]: "
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

