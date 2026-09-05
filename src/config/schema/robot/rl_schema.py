"""RL optimisation-layer config schema."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .._base import StrictModel


# RL optimisation extension layer (contract surface). The schema accepts five
# product modes; the runtime supports only {geometry_only, hybrid_ml} and
# rejects the rest. Any RL-active mode requires both policy_id and artifact_path
# so an operator cannot enable a mode whose producer was never promoted.
#
# Operator authority hierarchy (hard lock): robot.safety > runtime/hardware
# constraints > deterministic geometry > deterministic recovery > ML/RL. Nothing
# here may alter safety. Action masks are NOT YAML knobs: they are immutable
# code-side mappings per policy class, so the runtime safety boundary cannot be
# widened via config.

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
#: ranking/sequencing/recovery decisions). All three require a typed
#: artifact and trigger telemetry-completeness requirements.
RL_ACTIVE_MODES: frozenset[str] = frozenset(
    {RL_MODE_RL_SHADOW, RL_MODE_RL_ACTIVE, RL_MODE_RL_EXPERIMENTAL}
)




class RLExperimentalConfig(StrictModel):
    """``rl_experimental`` lane carrier.

    The runtime rejects this lane; the block exists so operators see the full
    surface in YAML. ``enabled`` must be ``True`` for the schema to accept
    ``mode = "rl_experimental"``.
    """

    enabled: bool = Field(default=False)


class RobotRLConfig(StrictModel):
    """RL optimisation extension layer.

    The default ``mode = "hybrid_ml"`` is the current production stack
    (deterministic pipeline + promoted ML scorers, no RL artifact loaded), so
    defaults are byte-identical to the deterministic production behaviour.

    To enable an RL-active mode the operator must set ``mode`` to one of
    ``rl_shadow`` / ``rl_active`` / ``rl_experimental``, provide both
    ``policy_id`` and ``artifact_path``, and (for ``rl_experimental``) set
    ``experimental.enabled``. The runtime still rejects every RL-active mode
    with a typed :class:`RLModeNotImplementedError`; the schema gate is
    defence-in-depth for when the producers land.
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
    # StrictModel forbids extras, so the shadow.py `getattr(rl_cfg, "<x>_artifact_path", None)` overrides
    # were unreachable without these fields.
    ranking_artifact_path: str | None = Field(default=None, min_length=1)
    sequencing_artifact_path: str | None = Field(default=None, min_length=1)
    perception_artifact_path: str | None = Field(default=None, min_length=1)
    recovery_artifact_path: str | None = Field(default=None, min_length=1)
    experimental: RLExperimentalConfig = Field(
        default_factory=RLExperimentalConfig
    )
    # Learner runtime tier. ``stdlib`` (default) is the pure-stdlib shipping path:
    # byte-deterministic, portable, the safety-auditable hot path. ``numpy`` and
    # ``torch`` are OPT-IN, lazily-imported heavier tiers for exploring what stronger
    # models achieve; they never become the ship default implicitly and stay fully
    # subordinate to the safety action mask. Heavy-tier artifacts are experimental-stamped
    # and never overwrite the committed stdlib goldens.
    runtime_tier: Literal["stdlib", "numpy", "torch"] = Field(default="stdlib")
    # Tier-0 online learning (shadow-only, zero grasp influence). When True the shadow
    # router accumulates online updates into a mutable side-state and stamps the artifact
    # root ``online_state`` (still-shadow); the promotion gate keeps refusing online-mutated
    # artifacts. Requires an RL mode (defence-in-depth against a silent no-op).
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
