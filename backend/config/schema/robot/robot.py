"""Schemas for the vendor-selected robot configuration tree."""

from __future__ import annotations


from pydantic import Field, field_validator, model_validator

from .._base import StrictModel

from .ur import URConfig
from .kuka import KukaConfig, KukaEkiConfig
from .dummy import DummyConfig
from .sim import (
    SimCameraSchema,
    SimConfig,
    SimGateConfig,
    SimMarkerConfig,
    SimObjectConfig,
    SimSceneConfig,
    SimTableConfig,
    SimViewpointConfig,
)
from .calibration import RobotCalibrationConfig, RobotCalibrationQualityBandsMm

from .safety import (
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

from .grasping import (
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

from .rl import (
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


class GripperConfig(StrictModel):
    """Physical gripper opening limits.

    The ``vendor`` field selects which driver the robot runtime will
    instantiate. It is validated at config-load against the ``GripperVendor``
    enum: unknown vendors are rejected early with a clear message rather than
    failing late at ``create_gripper``.
    """

    vendor: str = Field(default="robotiq", min_length=1)
    max_width_mm: float = Field(default=150.0, gt=0.0)
    min_width_mm: float = Field(default=5.0, ge=0.0)

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

