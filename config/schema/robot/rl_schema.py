"""RL optimisation-layer config schema."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .._base import StrictModel


#: Stable string identifiers for every product mode.
RL_MODE_GEOMETRY_ONLY: str = "geometry_only"
RL_MODE_HYBRID_ML: str = "hybrid_ml"
RL_MODE_RL_SHADOW: str = "rl_shadow"
RL_MODE_RL_ACTIVE: str = "rl_active"
RL_MODE_RL_EXPERIMENTAL: str = "rl_experimental"

#: All five product modes accepted by the schema. The runtime supports only
#: the deterministic subset; see
#: :data:`src.robot.grasping.rl.RL_SUPPORTED_MODES_DETERMINISTIC`.
RL_MODE_VALUES: tuple[str, ...] = (
    RL_MODE_GEOMETRY_ONLY,
    RL_MODE_HYBRID_ML,
    RL_MODE_RL_SHADOW,
    RL_MODE_RL_ACTIVE,
    RL_MODE_RL_EXPERIMENTAL,
)

#: RL-active modes (i.e. those that, once implemented, may influence
#: ranking/sequencing/recovery decisions).
RL_ACTIVE_MODES: frozenset[str] = frozenset(
    {RL_MODE_RL_SHADOW, RL_MODE_RL_ACTIVE, RL_MODE_RL_EXPERIMENTAL}
)


class RLRollbackTriggersConfig(StrictModel):
    """Auto-rollback triggers for RL-active modes (carrier defaults).

    The default thresholds are placeholders that install the schema surface; the
    final numeric values are re-locked (and written into the promoted artifact
    manifest) before any RL artifact reaches ``rl_active``.

    All triggers are non-negative. ``0`` means "any single occurrence triggers
    rollback" for count fields and "must stay strictly at baseline" for rate
    fields.
    """

    decision_latency_breach_consecutive_windows: int = Field(
        default=3, ge=1
    )
    override_regret_breach_consecutive_windows: int = Field(
        default=2, ge=1
    )
    override_regret_rate_cap: float = Field(
        default=0.10, ge=0.0, le=1.0
    )
    untyped_outcome_cap: int = Field(default=0, ge=0)
    dead_loop_rate_tolerance_abs: float = Field(
        default=0.005, ge=0.0, le=1.0
    )


class RLExperimentalConfig(StrictModel):
    """``rl_experimental`` lane carrier."""

    enabled: bool = Field(default=False)


class RobotRLConfig(StrictModel):
    """RL optimisation extension layer.

    The default ``mode = "hybrid_ml"`` is the current production stack
    (deterministic pipeline + promoted ML scorers, no RL artifact loaded), so
    defaults are byte-identical to the deterministic production behaviour.
    """

    mode: Literal[
        "geometry_only",
        "hybrid_ml",
        "rl_shadow",
        "rl_active",
        "rl_experimental",
    ] = Field(default="hybrid_ml")
    policy_id: str | None = Field(default=None, min_length=1)
    artifact_path: str | None = Field(default=None, min_length=1)
    # Optional per-policy artifact overrides for the shadow router (else committed baselines are used).
    ranking_artifact_path: str | None = Field(default=None, min_length=1)
    sequencing_artifact_path: str | None = Field(default=None, min_length=1)
    perception_artifact_path: str | None = Field(default=None, min_length=1)
    recovery_artifact_path: str | None = Field(default=None, min_length=1)
    rollback_triggers: RLRollbackTriggersConfig = Field(
        default_factory=RLRollbackTriggersConfig
    )
    experimental: RLExperimentalConfig = Field(
        default_factory=RLExperimentalConfig
    )
    # Learner runtime tier.
    runtime_tier: Literal["stdlib", "numpy", "torch"] = Field(default="stdlib")
    online_update_enabled: bool = Field(default=False)

    @model_validator(mode="after")
    def _validate_rl_active_requirements(self) -> "RobotRLConfig":
        if self.online_update_enabled and self.mode not in RL_ACTIVE_MODES:
            raise ValueError(
                "robot.rl.online_update_enabled=true requires an RL mode "
                "(rl_shadow / rl_active / rl_experimental)"
            )
        if self.mode in RL_ACTIVE_MODES:
            missing: list[str] = []
            if self.policy_id is None:
                missing.append("policy_id")
            if self.artifact_path is None:
                missing.append("artifact_path")
            if missing:
                raise ValueError(
                    f"robot.rl.mode={self.mode!r} requires "
                    f"{', '.join(missing)} to be set"
                )
            if (
                self.mode == RL_MODE_RL_EXPERIMENTAL
                and not self.experimental.enabled
            ):
                raise ValueError(
                    "robot.rl.mode='rl_experimental' requires "
                    "robot.rl.experimental.enabled=true"
                )
        return self
