"""Grasping (deterministic + RL-shadow)."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field, model_validator

from .._base import StrictModel


class GraspingClosedLoopConfig(StrictModel):
    """Pre-grasp refinement (closed-loop) knobs.

    Consumed by :class:`backend.src.robot.grasping.closed_loop.refinement.RefinementPolicy`
    when the operator opts into closed-loop modes. ``enabled=False`` disables the
    second-scan refinement step even when the active mode profile would permit it.
    """

    enabled: bool = Field(default=False)
    pregrasp_rescan: bool = Field(default=True)
    max_position_correction_mm: float = Field(default=20.0, gt=0.0, le=200.0)
    max_orientation_correction_deg: float = Field(
        default=15.0, gt=0.0, le=90.0
    )
    max_grip_width_correction_mm: float = Field(
        default=20.0, gt=0.0, le=200.0
    )
    target_match_iou_threshold: float = Field(default=0.3, ge=0.0, le=1.0)


class GraspingVerificationConfig(StrictModel):
    """Post-grasp verification knobs.

    Mirrors the operator-tunable fields on
    :class:`backend.src.robot.grasping.closed_loop.verification.GraspVerificationPolicy`.
    Verification stays off unless ``enabled: true`` is set explicitly.
    """

    enabled: bool = Field(default=False)
    require_object_detected: bool = Field(default=False)
    width_delta_min_mm: float = Field(default=2.0, ge=0.0, le=50.0)
    #: How far ABOVE the commanded close width the jaws may sit and still count as engaged, mm.
    #:
    #: MEASURED 2026-08-09, why this is no longer ``None``: without an upper bound the verifier cannot
    #: see the case it exists for. A gripper whose close never executed sits at its pre-open width
    #: (~80 mm on a 2F-85), and with no ceiling that was reported PASSED -- the cell carries air to the
    #: drop-off and logs a success. 10 mm separates cleanly (a commanded 45 mm close reads 45-ish when
    #: it holds, ~80 mm when it did not) while leaving room for a compliant object that lands a few mm
    #: proud of the command. Re-measure on the real jaw before trusting the exact number.
    width_delta_max_mm: float | None = Field(default=10.0, ge=0.0, le=50.0)
    post_lift_vision_check: bool = Field(default=False)
    vision_displacement_iou_max: float = Field(default=0.2, ge=0.0, le=1.0)
    fail_closed: bool = Field(default=True)


class GraspingDenseRecoveryConfig(StrictModel):
    """Scene-recovery knobs (dense-clutter behaviour).

    ``allowed_actions`` is a free-form list of recovery action names, matched
    against :class:`backend.src.robot.grasping.recovery.SceneRecoveryAction` at
    runtime, so unknown names produce a typed error rather than a silent allow.
    """

    enabled: bool = Field(default=False)
    allowed_actions: tuple[str, ...] = Field(default=("next_viewpoint",))
    max_recovery_actions: int = Field(default=2, ge=0, le=10)
    nudge_max_offset_mm: float = Field(default=15.0, gt=0.0, le=100.0)
    container_agitate_amplitude_mm: float = Field(
        default=10.0, ge=0.0, le=100.0
    )


class GraspingDecisionConfig(StrictModel):
    """Pre-execution decision-layer knobs.

    Consumed by :class:`backend.src.robot.grasping.decision.DecisionPolicy` when
    the operator opts the AUTO fail-closed decision layer on. Defaults to
    ``enabled=False`` so an existing ``robot.yaml`` keeps its previous behaviour.

    Attributes
    ----------
    enabled
        Master switch. When :data:`False`, the runtime does not build a
        :class:`DecisionEngine` and the legacy pick path is used.
    auto_uncertainty_threshold
        Maximum uncertainty (in ``[0, 1]``) accepted as a confident grasp.
        ``uncertainty = (1 - top_score)`` plus an optional ``reasons_penalty``
        when the grasp result carries failure reasons. Default ``0.4`` ⇒ require
        ``top_score >= 0.6``.
    max_reobservations
        Bounded camera re-observation budget per
        :meth:`AutonomousGraspService.pick` call. Default ``2``.
    reasons_penalty
        Added to the base uncertainty when :class:`GraspResult.reasons` is
        non-empty. Default ``0.2``.
    fail_closed_on_real_hardware
        When :data:`True` and the running arm reports
        ``capabilities.is_simulated == False``, an exhausted re-observation budget
        or absent viewpoint planner is mapped to
        :class:`DecisionAction.FAIL_CLOSED`. Simulated runs stay permissive.
    """

    enabled: bool = Field(default=False)
    auto_uncertainty_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    max_reobservations: int = Field(default=2, ge=0, le=10)
    reasons_penalty: float = Field(default=0.2, ge=0.0, le=1.0)
    fail_closed_on_real_hardware: bool = Field(default=True)


_KNOWN_GRASP_MODES: frozenset[str] = frozenset(
    {
        "easy",
        "auto",
        "closed_loop",
        "dense_clutter",
        "dense_autonomous",
    }
)


class GraspingFeasibilityConfig(StrictModel):
    """Feasibility-aware ranking knobs.

    Carries three pre-commit feasibility signals that demote (never reject)
    candidates with poor execution prospects: IK reachability quality, joint-limit
    margin, and swept approach feasibility. All three are disabled by default
    (byte-identical). The operator opts in per-signal by flipping the
    corresponding ``*_enabled`` flag *and* setting a non-zero top-level
    :attr:`weight` — without both, the signal carries no weight. ``apply_modes``
    excludes ``easy`` so EASY's cycle-time budget is preserved.

    Attributes
    ----------
    enabled
        Master switch. When :data:`False`, the calculator is built without a
        feasibility scorer and the legacy ranking is used.
    weight
        Top-level feasibility weight in the
        :class:`~backend.src.robot.grasping.scoring.GraspScoreWeights` normalised
        average. Default ``0.0`` keeps ``total_score`` byte-identical.
    apply_modes
        Modes not listed here skip feasibility ranking regardless of any other
        flag (EASY is permanently locked out).
    ik_quality_enabled, joint_margin_enabled, swept_approach_enabled
        Per-signal switches. Disabled signals contribute a neutral ``0.5`` floor
        so they never demote a candidate below an enabled-but-unknown peer.
    ik_quality_weight, joint_margin_weight, swept_approach_weight
        Internal sub-weights inside the feasibility score; only relevant when
        their corresponding signal is enabled.
    swept_approach_top_k
        Maximum number of top-ranked candidates evaluated with swept-approach
        validation. Caps O(N) cost so the affected modes still meet cycle time.
    """

    enabled: bool = Field(default=False)
    weight: float = Field(default=0.0, ge=0.0, le=1.0)
    apply_modes: tuple[str, ...] = Field(
        default=("auto", "dense_clutter", "dense_autonomous")
    )
    ik_quality_enabled: bool = Field(default=False)
    ik_quality_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    joint_margin_enabled: bool = Field(default=False)
    joint_margin_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    swept_approach_enabled: bool = Field(default=False)
    swept_approach_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    swept_approach_top_k: int = Field(default=5, ge=1, le=64)

    @model_validator(mode="after")
    def _validate_apply_modes(self) -> "GraspingFeasibilityConfig":
        unknown = [m for m in self.apply_modes if m not in _KNOWN_GRASP_MODES]
        if unknown:
            raise ValueError(
                "feasibility.apply_modes contains unknown grasp mode(s): "
                f"{unknown!r}; valid: {sorted(_KNOWN_GRASP_MODES)!r}"
            )
        return self


class GraspingOcclusionConfig(StrictModel):
    """Occlusion and hidden-geometry knobs.

    Carries two directional-occlusion behaviours on top of the feasibility
    carrier: a corridor analyzer (marches the approach — and optional retreat —
    axis in camera-frame to estimate per-direction clearance and a fused
    depth+mask blockage confidence) and a hard reject (drops candidates whose
    corridor confidence exceeds :attr:`hard_reject_confidence_threshold` instead
    of merely demoting them). All flags default off (byte-identical); EASY is
    excluded from :attr:`apply_modes`.

    Attributes
    ----------
    directional_enabled
        Master switch for the corridor analyzer. When False, no corridor reports
        are produced and ``hard_reject_enabled`` has no effect.
    hard_reject_enabled
        When True (and the analyzer is enabled and the mode is in
        ``apply_modes``), candidates whose blockage confidence exceeds
        :attr:`hard_reject_confidence_threshold` are rejected outright rather than
        only demoted by feasibility.
    apply_modes
        Modes not listed here skip the analyzer entirely regardless of other flags.
    corridor_radius_mm, corridor_step_mm, corridor_max_distance_mm
        Sampling geometry passed to the analyzer.
    mask_fusion_weight
        Convex-combination weight of mask blockage vs depth blockage in the
        analyzer's fused confidence.
    partial_confidence_threshold, hard_reject_confidence_threshold
        Confidence cut-offs separating CLEAR ↔ PARTIAL ↔ BLOCKED and the
        hard-reject trigger respectively.
    top_k
        Cap on number of candidates analyzed per ``pick()`` cycle.
    """

    directional_enabled: bool = Field(default=False)
    hard_reject_enabled: bool = Field(default=False)
    apply_modes: tuple[str, ...] = Field(
        default=("auto", "dense_clutter", "dense_autonomous")
    )
    corridor_radius_mm: float = Field(default=20.0, gt=0.0)
    corridor_step_mm: float = Field(default=5.0, gt=0.0)
    corridor_max_distance_mm: float = Field(default=200.0, gt=0.0)
    mask_fusion_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    partial_confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    hard_reject_confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    top_k: int = Field(default=5, ge=1, le=64)

    @model_validator(mode="after")
    def _validate(self) -> "GraspingOcclusionConfig":
        unknown = [m for m in self.apply_modes if m not in _KNOWN_GRASP_MODES]
        if unknown:
            raise ValueError(
                "occlusion.apply_modes contains unknown grasp mode(s): "
                f"{unknown!r}; valid: {sorted(_KNOWN_GRASP_MODES)!r}"
            )
        if self.partial_confidence_threshold >= self.hard_reject_confidence_threshold:
            raise ValueError(
                "occlusion.partial_confidence_threshold "
                f"({self.partial_confidence_threshold}) must be < "
                "hard_reject_confidence_threshold "
                f"({self.hard_reject_confidence_threshold})"
            )
        return self


class BlockerGraphSchemaConfig(StrictModel):
    """Per-signal blocker-graph flags.

    Every signal defaults off so :class:`GraspingOrderingConfig` is behaviourally
    inert until the operator opts in. ``adjacency_radius_px`` is integer pixels in
    the perception mask; ``depth_tolerance_mm`` is millimetres of camera-space
    depth difference required before one candidate is considered "in front of"
    another.
    """

    mask_adjacency_enabled: bool = Field(default=False)
    depth_only_enabled: bool = Field(default=False)
    corridor_overlap_enabled: bool = Field(default=False)
    adjacency_radius_px: int = Field(default=5, ge=0, le=128)
    depth_tolerance_mm: float = Field(default=10.0, ge=0.0)


class GraspingOrderingConfig(StrictModel):
    """Clutter-aware target-ordering knobs.

    When :attr:`enabled` the runtime computes an *unlock score* per candidate via
    the configured :class:`BlockerGraphSchemaConfig` signals and a *priority*
    score ``local + unlock_weight * unlock``. The ``argmax(local_score)`` winner
    is only displaced when the priority winner is within
    :attr:`max_local_score_drop` of it — the operator decides how much local score
    they will give up to unblock other picks. Every flag defaults off
    (byte-identical); EASY is excluded from the default apply-modes.
    """

    enabled: bool = Field(default=False)
    unlock_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    max_local_score_drop: float = Field(default=0.1, ge=0.0, le=1.0)
    apply_modes: tuple[str, ...] = Field(
        default=("auto", "dense_clutter", "dense_autonomous")
    )
    blocker_graph: BlockerGraphSchemaConfig = Field(
        default_factory=BlockerGraphSchemaConfig
    )

    @model_validator(mode="after")
    def _validate_apply_modes(self) -> "GraspingOrderingConfig":
        unknown = [m for m in self.apply_modes if m not in _KNOWN_GRASP_MODES]
        if unknown:
            raise ValueError(
                "ordering.apply_modes contains unknown grasp mode(s): "
                f"{unknown!r}; valid: {sorted(_KNOWN_GRASP_MODES)!r}"
            )
        return self


class GraspingRecoveryConfig(StrictModel):
    """Recovery-orchestration knobs.

    Mirrors the surface in
    :class:`backend.src.robot.grasping.recovery.SceneRecoveryPolicy` but holds
    string action names (decoupled from the runtime enum import). Every flag
    defaults off (byte-identical); EASY is excluded from the default apply-modes.
    When :attr:`allowed_actions` contains a physical action (``nudge_target`` /
    ``container_agitate``), :attr:`fixture` must be set — the cross-field check
    prevents arming a physical recovery without an envelope.
    """

    enabled: bool = Field(default=False)
    max_recovery_actions: int = Field(default=2, ge=0, le=20)
    allowed_actions: tuple[str, ...] = Field(default=())
    per_action_budget: tuple[tuple[str, int], ...] = Field(default=())
    apply_modes: tuple[str, ...] = Field(
        default=("auto", "dense_clutter", "dense_autonomous")
    )
    fixture: "GraspingRecoveryFixtureConfig | None" = Field(default=None)

    @model_validator(mode="after")
    def _validate(self) -> "GraspingRecoveryConfig":
        valid_actions = frozenset(
            {
                "rescan",
                "next_viewpoint",
                "next_target",
                "nudge_target",
                "container_agitate",
            }
        )
        physical = frozenset({"nudge_target", "container_agitate"})
        for action in self.allowed_actions:
            if action not in valid_actions:
                raise ValueError(
                    "recovery.allowed_actions contains unknown action: "
                    f"{action!r}; valid: {sorted(valid_actions)!r}"
                )
        for entry in self.per_action_budget:
            if (
                not isinstance(entry, tuple)
                or len(entry) != 2
            ):
                raise ValueError(
                    "recovery.per_action_budget entries must be "
                    f"(action, count) pairs; got {entry!r}"
                )
            action, count = entry
            if action not in valid_actions:
                raise ValueError(
                    "recovery.per_action_budget unknown action: "
                    f"{action!r}; valid: {sorted(valid_actions)!r}"
                )
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError(
                    "recovery.per_action_budget counts must be non-negative "
                    f"ints; got {action}={count!r}"
                )
        unknown = [m for m in self.apply_modes if m not in _KNOWN_GRASP_MODES]
        if unknown:
            raise ValueError(
                "recovery.apply_modes contains unknown grasp mode(s): "
                f"{unknown!r}; valid: {sorted(_KNOWN_GRASP_MODES)!r}"
            )
        physical_in_use = [a for a in self.allowed_actions if a in physical]
        if physical_in_use and self.fixture is None:
            raise ValueError(
                "recovery physical actions require a fixture envelope; "
                f"got {physical_in_use!r} without fixture"
            )
        return self


class GraspingRecoveryFixtureConfig(StrictModel):
    """Operator-bounded envelope for physical recovery actions.

    A recovery action is the robot DELIBERATELY pushing something -- nudging a part that will not
    separate, agitating a container. That is motion aimed at the scene rather than at a grasp, so its
    reach is bounded by the operator rather than inferred: these three numbers are where a person says
    how far the cell is allowed to shove things, and the recovery planner may not exceed them.
    """

    #: Centre of the axis-aligned box, in the robot BASE frame (mm), inside which a physical recovery
    #: action may act. Typically the bin or tray the cell is clearing.
    center_mm: tuple[float, float, float] = Field(...)
    #: HALF the side length on each axis (mm) -- so the box spans ``center +/- half_extents``. Half
    #: extents rather than corners because the recovery planner reasons about distance from the centre.
    half_extents_mm: tuple[float, float, float] = Field(...)
    #: Longest single push (mm) any one recovery action may command. Bounds the DISPLACEMENT even when
    #: the box above would permit more: a large shove can eject a part from the bin or bury the target
    #: deeper, so the ceiling is deliberately small and separate from the envelope.
    max_nudge_mm: float = Field(default=5.0, ge=0.0, le=50.0)


class UncertaintyChannelWeightsConfig(StrictModel):
    """Per-channel weights for the uncertainty fusion layer.

    The five always-produced channels default to ``1.0``. The two optional
    channels (``topology_risk`` and ``semantic_confidence``) default to ``0.0``
    because the runtime does not always produce them; operators opt them in.
    """

    depth_confidence: float = Field(default=1.0, ge=0.0)
    mask_confidence: float = Field(default=1.0, ge=0.0)
    occlusion_corridor_risk: float = Field(default=1.0, ge=0.0)
    feasibility_margin: float = Field(default=1.0, ge=0.0)
    verification_residual: float = Field(default=1.0, ge=0.0)
    topology_risk: float = Field(default=0.0, ge=0.0)
    semantic_confidence: float = Field(default=0.0, ge=0.0)

    def sum(self) -> float:
        return (
            self.depth_confidence
            + self.mask_confidence
            + self.occlusion_corridor_risk
            + self.feasibility_margin
            + self.verification_residual
            + self.topology_risk
            + self.semantic_confidence
        )


class GraspingUncertaintyConfig(StrictModel):
    """Uncertainty fusion + calibration layer.

    Fusion is a weighted convex combination of seven typed per-channel
    monotone-remapped values (exposed via :class:`UncertaintyChannelWeightsConfig`;
    runtime carrier in :mod:`backend.src.robot.grasping.uncertainty`). When
    :attr:`enabled`, :attr:`fail_closed_threshold` replaces
    :attr:`GraspingDecisionConfig.auto_uncertainty_threshold`; when disabled the
    legacy decision threshold remains (byte-identical). The layer always emits a
    typed snapshot on every :class:`AutonomousGraspReport` (``fused=0.0,
    fused_available=False`` when disabled). ``"easy"`` is excluded from
    :attr:`apply_modes`.

    Attributes
    ----------
    enabled
        Master switch. Defaults to :data:`False` (byte-identical).
    fail_closed_threshold
        When ``fused >= fail_closed_threshold`` the decision layer fail-closes the
        attempt (replacing ``GraspingDecisionConfig.auto_uncertainty_threshold``).
        Default ``0.4`` mirrors the legacy value.
    apply_modes
        Stable mode names the layer applies to. ``"easy"`` is rejected.
    weights
        Per-channel non-negative weights. When :attr:`enabled` at least one weight
        must be strictly positive (otherwise the fused signal is unconditionally
        zero --- a footgun).
    ranking_penalty_weight
        Ranking carrier. Multiplies the fused uncertainty when subtracting from a
        candidate score. Default ``0.0`` ⇒ ranking byte-identical.
    recovery_aggressive_threshold
        Recovery carrier. When fused exceeds this threshold the orchestrator may
        prefer perception-recovery actions (``NEXT_VIEWPOINT``/``RESCAN``). Default
        ``1.01`` ⇒ never trips (fused is clamped to ``[0, 1]``).
    calibration_artifact_path
        Optional path (absolute or relative to the config root) to a JSON
        calibration artifact produced by
        :mod:`backend.src.robot.grasping.calibration.uncertainty_calibration`.
        When :data:`None`, the identity calibration is used.
    """

    enabled: bool = Field(default=False)
    fail_closed_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    apply_modes: tuple[str, ...] = Field(
        default=("auto", "dense_clutter", "dense_autonomous")
    )
    weights: UncertaintyChannelWeightsConfig = Field(
        default_factory=UncertaintyChannelWeightsConfig
    )
    ranking_penalty_weight: float = Field(
        default=0.0,
        ge=0.0,
        json_schema_extra={
            "runtime_mutable": True,
            "min_value": 0.0,
            "max_value": 1.0,
            "max_abs_step": 0.20,
            "max_rel_step": 0.50,
        },
    )
    recovery_aggressive_threshold: float = Field(
        default=1.01,
        ge=0.0,
        json_schema_extra={
            "runtime_mutable": True,
            "min_value": 0.0,
            "max_value": 1.01,
            "max_abs_step": 0.10,
            "max_rel_step": 0.30,
        },
    )
    # 1.01 = off-sentinel: the channel spread is bounded above by 1.0, so the gate never trips by default.
    channel_disagreement_threshold: float = Field(default=1.01, ge=0.0)
    calibration_artifact_path: str | None = Field(default=None)
    # Per-candidate uncertainty re-rank (the consumer of the corridor-risk producer). rerank_modes is
    # dense-only and SEPARATE from apply_modes (which includes "auto" and would crash the dense-only
    # carrier); default off + weight 0.0 is byte-identical, so enabling is observe-only until a positive
    # weight is set. Enabling here is NOT sufficient: the calculator owner must also build it with
    # ``corridor_risk_per_candidate=True`` (the producer) or the re-rank silently no-ops. Costs N corridor
    # depth-marches per dense pick.
    rerank_enabled: bool = Field(default=False)
    # Deliberately not runtime-mutable: weight 0.0 is an explicit, human-logged tuning step -- auto-tuning
    # it up would silently enable reordering, defeating the observe-only intent.
    rerank_weight: float = Field(default=0.0, ge=0.0, le=0.5)
    rerank_modes: tuple[str, ...] = Field(
        default=("dense_clutter", "dense_autonomous")
    )

    @model_validator(mode="after")
    def _validate(self) -> "GraspingUncertaintyConfig":
        if "easy" in self.apply_modes:
            raise ValueError(
                "uncertainty.apply_modes must not include 'easy'; "
                "EASY mode is permanently excluded from T6."
            )
        unknown = [m for m in self.apply_modes if m not in _KNOWN_GRASP_MODES]
        if unknown:
            raise ValueError(
                "uncertainty.apply_modes contains unknown grasp mode(s): "
                f"{unknown!r}; valid: {sorted(_KNOWN_GRASP_MODES)!r}"
            )
        if self.enabled and self.weights.sum() <= 0.0:
            raise ValueError(
                "uncertainty.enabled=True requires at least one "
                "weight to be strictly positive; got all zeros."
            )
        _dense_rerank_modes = frozenset({"dense_clutter", "dense_autonomous"})
        bad_rerank = [m for m in self.rerank_modes if m not in _dense_rerank_modes]
        if bad_rerank:
            raise ValueError(
                "uncertainty.rerank_modes is LOCKED to dense modes only "
                f"(mirrors the U4 ranking-blend contract); got disallowed mode(s) {bad_rerank!r}; "
                f"valid: {sorted(_dense_rerank_modes)!r}"
            )
        if len(set(self.rerank_modes)) != len(self.rerank_modes):
            raise ValueError("uncertainty.rerank_modes must be unique")
        return self


class GraspingSuccessModelConfig(StrictModel):
    """Success-probability overlay (default off).

    Ships the offline trainer + artifact + runtime-pure predictor. Defaults are
    picked so adding this block (or omitting it) is byte-identical:
    ``enabled=False`` ⇒ nothing in the pipeline loads or runs the model.

    Attributes
    ----------
    enabled
        Master switch. Defaults to :data:`False` so the model is opaque to the
        runtime.
    artifact_dir
        Path (absolute or relative to the config root) of the directory containing
        ``model.json`` and ``manifest.json``. Defaults to the committed v1 artifact
        under ``assets/models/success_probability/v1``.
    feature_schema_version
        The feature-schema version the runtime expects the artifact to declare.
        Bumping it opts future feature-set extensions out of the v1 artifact
        without breaking old deployments.
    apply_modes
        Stable mode names the model may score in. Defaults to the four canonical
        modes; ``"easy"`` is included because EASY pick rate is the
        strict-improvement target the model is calibrated against.
    ranking_blend_enabled
        Master switch for the bounded probability-blend reranker. Defaults to
        :data:`False` (byte-identical). When :data:`True` (and ``enabled``, and a
        non-shadow lifecycle artifact is loaded), the runtime reranks
        ``best_result.candidates`` in DENSE modes only using the convex blend
        ``(1 - w) * geometric + w * predicted_p``.
    ranking_blend_weight
        Blend weight ``w``. Clamped to ``[0.0, 0.5]`` so the geometric score always
        retains ≥50 % influence and a miscalibrated model cannot dominate ranking.
        Default ``0.20``.
    ranking_blend_modes
        Locked DENSE-only subset of grasp-modes the blend reranker may operate in:
        a subset of ``{"dense_clutter", "dense_autonomous"}`` (``"easy"`` and
        ``"auto"`` stay geometry-only so EASY pick-rate cannot regress). Must also
        be a subset of ``apply_modes`` (else the shadow probability is never
        annotated and the blend has nothing to read).
    """

    enabled: bool = Field(default=False)
    artifact_dir: str = Field(
        default="assets/models/success_probability/v1", min_length=1
    )
    feature_schema_version: int = Field(default=1, ge=1)
    apply_modes: tuple[str, ...] = Field(
        default=("easy", "auto", "dense_clutter", "dense_autonomous")
    )
    ranking_blend_enabled: bool = Field(default=False)
    ranking_blend_weight: float = Field(
        default=0.20,
        ge=0.0,
        le=0.5,
        json_schema_extra={
            "runtime_mutable": True,
            "min_value": 0.0,
            "max_value": 0.5,
            "max_abs_step": 0.05,
            "max_rel_step": 0.25,
        },
    )
    ranking_blend_modes: tuple[str, ...] = Field(
        default=("dense_clutter", "dense_autonomous")
    )
    lifecycle_phase: Literal["shadow", "canary", "active"] = Field(
        default="shadow",
        description=(
            "U4 lifecycle gate (G1). 'shadow' (default) = the learned success predictor runs but its "
            "output is discarded (byte-identical to the no-blend path). 'canary'/'active' let the "
            "bounded blend REORDER candidates in the dense modes; both REQUIRE a valid promotion.json "
            "(verify_promotion: SHA-chain + verdict='pass' + thresholds) or the model load fails CLOSED "
            "to None (no blend). Previously this was hardcoded to 'shadow' in the builder, so the "
            "promoted predictor could never influence which grasp executes."
        ),
    )

    @model_validator(mode="after")
    def _validate(self) -> "GraspingSuccessModelConfig":
        unknown = [m for m in self.apply_modes if m not in _KNOWN_GRASP_MODES]
        if unknown:
            raise ValueError(
                "success_model.apply_modes contains unknown grasp mode(s): "
                f"{unknown!r}; valid: {sorted(_KNOWN_GRASP_MODES)!r}"
            )
        if len(set(self.apply_modes)) != len(self.apply_modes):
            raise ValueError("success_model.apply_modes must be unique")
        # Ranking-blend is LOCKED to dense-only so EASY/AUTO/CLOSED_LOOP never blend and EASY pick-rate
        # cannot regress regardless of operator config.
        _DENSE_ONLY: frozenset[str] = frozenset(
            {"dense_clutter", "dense_autonomous"}
        )
        forbidden = [
            m for m in self.ranking_blend_modes if m not in _DENSE_ONLY
        ]
        if forbidden:
            raise ValueError(
                "success_model.ranking_blend_modes is LOCKED to "
                "dense modes only per the U4 design contract; "
                f"got disallowed mode(s) {forbidden!r}; valid: "
                f"{sorted(_DENSE_ONLY)!r}"
            )
        if len(set(self.ranking_blend_modes)) != len(self.ranking_blend_modes):
            raise ValueError(
                "success_model.ranking_blend_modes must be unique"
            )
        # Blend modes must be a subset of apply_modes: otherwise the
        # shadow probability is never annotated for that mode and the
        # blend would have nothing to read.
        missing = [
            m for m in self.ranking_blend_modes if m not in self.apply_modes
        ]
        if missing:
            raise ValueError(
                "success_model.ranking_blend_modes must be a subset of "
                f"apply_modes; got {missing!r} not in {self.apply_modes!r}"
            )
        return self


class GraspingWatchdogConfig(StrictModel):
    """Drift + OOD watchdog runtime knobs.

    Evaluates five drift monitors (predicted-vs-observed calibration delta,
    depth-confidence shift, fail-closed spike, hand-eye residual trend,
    verification residual trend) plus an OOD score over a rolling window of recent
    attempts, and surfaces a typed severity ladder
    (``none``/``low``/``moderate``/``high``/``severe``). Evaluators are pure
    functions; the :class:`AutonomousGraspService` owns the rolling history. The
    default ``shadow`` mode computes + emits telemetry but never alters behaviour;
    HIGH/SEVERE block AUTO on real hardware only (a telemetry flag in sim).
    Defaults are conservative (a healthy scene stays ``none``); thresholds form
    non-decreasing ladders and a misconfigured YAML is rejected at construction.

    Attributes
    ----------
    mode
        One of ``"disabled"``, ``"shadow"``, ``"canary"``, ``"active"``. Defaults
        to ``"shadow"``. ``"disabled"`` omits watchdog telemetry entirely.
    window_size
        Rolling history length, in attempts. Must be ``>= 1``.
    min_samples_for_trend
        Minimum samples before any trend monitor emits anything other than
        ``none``. Must be ``>= 2``.
    block_modes
        Stable grasp-mode names (matching
        :class:`backend.src.robot.grasping.types.modes.GraspMode`) that fall under the
        HIGH/SEVERE block on real hardware. Defaults to ``("auto",)``. ``"easy"``
        is permanently rejected.
    """

    mode: str = Field(default="shadow", min_length=1)
    window_size: int = Field(
        default=20,
        ge=1,
        le=1024,
        json_schema_extra={
            "runtime_mutable": True,
            "min_value": 1,
            "max_value": 1024,
            "max_abs_step": 10,
            "max_rel_step": 0.50,
        },
    )
    min_samples_for_trend: int = Field(default=6, ge=2, le=1024)

    calibration_delta_moderate_mm: float = Field(default=2.0, ge=0.0)
    calibration_delta_high_mm: float = Field(default=5.0, ge=0.0)
    calibration_delta_severe_mm: float = Field(default=10.0, ge=0.0)

    depth_confidence_shift_moderate: float = Field(default=0.05, ge=0.0, le=1.0)
    depth_confidence_shift_high: float = Field(default=0.10, ge=0.0, le=1.0)
    depth_confidence_shift_severe: float = Field(default=0.20, ge=0.0, le=1.0)

    fail_closed_rate_moderate: float = Field(default=0.30, ge=0.0, le=1.0)
    fail_closed_rate_high: float = Field(default=0.50, ge=0.0, le=1.0)
    fail_closed_rate_severe: float = Field(default=0.75, ge=0.0, le=1.0)

    hand_eye_residual_trend_moderate_mm: float = Field(default=1.0, ge=0.0)
    hand_eye_residual_trend_high_mm: float = Field(default=3.0, ge=0.0)
    hand_eye_residual_trend_severe_mm: float = Field(default=7.0, ge=0.0)

    verification_residual_trend_moderate_mm: float = Field(default=0.5, ge=0.0)
    verification_residual_trend_high_mm: float = Field(default=2.0, ge=0.0)
    verification_residual_trend_severe_mm: float = Field(default=5.0, ge=0.0)

    ood_score_moderate: float = Field(default=0.5, ge=0.0, le=1.0)
    ood_score_high: float = Field(default=0.3, ge=0.0, le=1.0)
    ood_score_severe: float = Field(default=0.1, ge=0.0, le=1.0)

    block_modes: tuple[str, ...] = Field(default=("auto",))

    _ALLOWED_MODES: ClassVar[frozenset[str]] = frozenset(
        {"disabled", "shadow", "canary", "active"}
    )

    @model_validator(mode="after")
    def _validate(self) -> "GraspingWatchdogConfig":
        if self.mode not in self._ALLOWED_MODES:
            raise ValueError(
                "watchdog.mode must be one of "
                f"{sorted(self._ALLOWED_MODES)!r}; got {self.mode!r}"
            )
        ladders: tuple[tuple[str, tuple[float, float, float]], ...] = (
            (
                "calibration_delta",
                (
                    self.calibration_delta_moderate_mm,
                    self.calibration_delta_high_mm,
                    self.calibration_delta_severe_mm,
                ),
            ),
            (
                "depth_confidence_shift",
                (
                    self.depth_confidence_shift_moderate,
                    self.depth_confidence_shift_high,
                    self.depth_confidence_shift_severe,
                ),
            ),
            (
                "fail_closed_rate",
                (
                    self.fail_closed_rate_moderate,
                    self.fail_closed_rate_high,
                    self.fail_closed_rate_severe,
                ),
            ),
            (
                "hand_eye_residual_trend",
                (
                    self.hand_eye_residual_trend_moderate_mm,
                    self.hand_eye_residual_trend_high_mm,
                    self.hand_eye_residual_trend_severe_mm,
                ),
            ),
            (
                "verification_residual_trend",
                (
                    self.verification_residual_trend_moderate_mm,
                    self.verification_residual_trend_high_mm,
                    self.verification_residual_trend_severe_mm,
                ),
            ),
        )
        for name, (m, h, s) in ladders:
            if not (m <= h <= s):
                raise ValueError(
                    f"watchdog.{name}_* thresholds must satisfy "
                    f"moderate <= high <= severe; got ({m}, {h}, {s})"
                )
        # OOD ladder is inverted (smaller score => higher severity).
        if not (
            self.ood_score_moderate
            >= self.ood_score_high
            >= self.ood_score_severe
        ):
            raise ValueError(
                "watchdog.ood_score_* thresholds must satisfy "
                "moderate >= high >= severe; got "
                f"({self.ood_score_moderate}, "
                f"{self.ood_score_high}, "
                f"{self.ood_score_severe})"
            )
        if "easy" in self.block_modes:
            raise ValueError(
                "watchdog.block_modes must not include 'easy'; "
                "EASY mode is permanently excluded from U+ "
                "behavioural layers."
            )
        unknown = [m for m in self.block_modes if m not in _KNOWN_GRASP_MODES]
        if unknown:
            raise ValueError(
                "watchdog.block_modes contains unknown grasp mode(s): "
                f"{unknown!r}; valid: {sorted(_KNOWN_GRASP_MODES)!r}"
            )
        if len(set(self.block_modes)) != len(self.block_modes):
            raise ValueError(
                "watchdog.block_modes must be unique; got "
                f"{self.block_modes!r}"
            )
        return self


class GraspingPerformanceConfig(StrictModel):
    """Runtime SLO budgets and bounded-compute knobs.

    Carries the per-stage p95 latency budgets the runtime measures against and
    placeholders for bounded-compute knobs (timeouts + fallback policies) that are
    plumbed end-to-end but not yet enforced — real enforcement waits for the live
    success-probability model and fusion path, so today the timeouts are advisory.
    Per-stage latency is captured by a typed
    :class:`backend.src.robot.grasping.telemetry.latency_tracker.LatencyTracker` seam owned
    by :class:`AutonomousGraspService`; a stage that does not run has an absent
    (``None``) latency and the SLO gate skips nulls. A breach emits a typed
    ``RobotWatchdogEvent.SLO_BREACH`` and lets the listener decide — the runtime
    never fail-closes on its own SLO.

    Attributes
    ----------
    enabled
        Master switch. When ``False`` the runtime still records latency telemetry
        but does not emit breach events or apply bounded-compute fallbacks. Default
        ``False`` is byte-identical.
    decision_latency_slo_ms / ranking_latency_slo_ms / fusion_latency_slo_ms
        Per-stage p95 budgets enforced by the SLO gate.
    decision_timeout_ms / ranking_timeout_ms / fusion_timeout_ms
        Hard per-stage caps. ``0.0`` means "no cap" (the current default); non-zero
        values are honoured once the producers light up, and treated as telemetry
        budgets today.
    model_fallback_policy
        How the runtime degrades when the success-probability model breaches its
        timeout: ``"legacy_score"`` reverts to geometry-only ranking, ``"skip"``
        omits the probability signal, ``"fail_closed"`` fail-closes with a typed
        reason.
    fusion_fallback_policy
        How the runtime degrades when fusion breaches its timeout: ``"skip"``
        proceeds with single-view evidence, ``"fail_closed"`` fail-closes.
    emit_breach_events
        When ``True`` and ``enabled``, the service emits a
        ``RobotWatchdogEvent.SLO_BREACH`` on rising-edge p95 breaches. Default
        ``False``.
    breach_window_size
        Rolling window length (in attempts) for the in-runtime p95 that drives
        ``SLO_BREACH`` emission. The replay/soak ``--slo-gate`` computes p95 across
        the full pack.
    """

    enabled: bool = Field(default=False)
    decision_latency_slo_ms: float = Field(default=60.0, gt=0.0)
    ranking_latency_slo_ms: float = Field(default=80.0, gt=0.0)
    fusion_latency_slo_ms: float = Field(default=220.0, gt=0.0)
    decision_timeout_ms: float = Field(default=0.0, ge=0.0)
    ranking_timeout_ms: float = Field(default=0.0, ge=0.0)
    fusion_timeout_ms: float = Field(default=0.0, ge=0.0)
    model_fallback_policy: str = Field(default="legacy_score", min_length=1)
    fusion_fallback_policy: str = Field(default="skip", min_length=1)
    emit_breach_events: bool = Field(default=False)
    breach_window_size: int = Field(
        default=32,
        ge=2,
        le=4096,
        json_schema_extra={
            "runtime_mutable": True,
            "min_value": 2,
            "max_value": 4096,
            "max_abs_step": 16,
            "max_rel_step": 0.50,
        },
    )

    _ALLOWED_MODEL_FALLBACKS: ClassVar[frozenset[str]] = frozenset(
        {"legacy_score", "skip", "fail_closed"}
    )
    _ALLOWED_FUSION_FALLBACKS: ClassVar[frozenset[str]] = frozenset(
        {"skip", "fail_closed"}
    )

    @model_validator(mode="after")
    def _validate(self) -> "GraspingPerformanceConfig":
        if self.model_fallback_policy not in self._ALLOWED_MODEL_FALLBACKS:
            raise ValueError(
                "performance.model_fallback_policy must be one of "
                f"{sorted(self._ALLOWED_MODEL_FALLBACKS)!r}; got "
                f"{self.model_fallback_policy!r}"
            )
        if self.fusion_fallback_policy not in self._ALLOWED_FUSION_FALLBACKS:
            raise ValueError(
                "performance.fusion_fallback_policy must be one of "
                f"{sorted(self._ALLOWED_FUSION_FALLBACKS)!r}; got "
                f"{self.fusion_fallback_policy!r}"
            )
        # When a per-stage timeout is set, it must not exceed the
        # SLO budget for that stage — a timeout looser than the SLO
        # is silently useless and almost certainly a config mistake.
        pairs = (
            (
                "decision",
                self.decision_timeout_ms,
                self.decision_latency_slo_ms,
            ),
            (
                "ranking",
                self.ranking_timeout_ms,
                self.ranking_latency_slo_ms,
            ),
            (
                "fusion",
                self.fusion_timeout_ms,
                self.fusion_latency_slo_ms,
            ),
        )
        for name, timeout, slo in pairs:
            if timeout > 0.0 and timeout > slo:
                raise ValueError(
                    f"performance.{name}_timeout_ms ({timeout}) must "
                    f"not exceed performance.{name}_latency_slo_ms "
                    f"({slo})"
                )
        return self


class RobotGraspingCommitPolicyConfig(StrictModel):
    """Mandatory dense multi-view commit gate.

    When enabled (and the resolved mode is in :attr:`apply_modes`), the pick loop
    refuses to hand a winning candidate to execution unless the fused scene memory
    satisfies both ``views_accepted >= min_views_accepted`` AND the candidate's
    approach corridor (a :attr:`corridor_length_mm` x :attr:`corridor_radius_mm`
    capsule along its approach vector) has at least
    :attr:`min_corridor_hit_fraction` of its in-ROI voxels marked hit. On refusal
    the orchestrator triggers up to :attr:`max_reobserve_attempts` more captures
    before surfacing a typed
    :attr:`~backend.src.robot.grasping.loop.pick_loop.PickOutcome.NO_COMMIT_INSUFFICIENT_FUSION`.
    Default ``enabled=False`` is byte-identical.
    """

    enabled: bool = Field(
        default=False,
        description=(
            "Top-level switch. When False the commit gate is a no-op "
            "(byte-identical)."
        ),
    )
    min_views_accepted: int = Field(
        default=2,
        ge=1,
        le=64,
        description=(
            "Minimum number of accepted views that must already be "
            "fused before the orchestrator may commit a candidate to "
            "execution."
        ),
    )
    min_corridor_hit_fraction: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum fraction of voxels inside the candidate's "
            "approach corridor that must be marked hit in the fusion "
            "grid. 0.0 disables the corridor check (views-only gate)."
        ),
    )
    corridor_radius_mm: float = Field(
        default=25.0,
        gt=0.0,
        le=200.0,
        description=(
            "Radius (mm) of the cylindrical capsule around the "
            "candidate's approach line used for corridor evidence."
        ),
    )
    corridor_length_mm: float = Field(
        default=120.0,
        gt=0.0,
        le=1_000.0,
        description=(
            "Length (mm) of the corridor capsule along the candidate's "
            "approach vector, anchored at the grasp position and "
            "extending opposite the approach (i.e. into the camera-"
            "viewing volume where evidence should accumulate)."
        ),
    )
    max_reobserve_attempts: int = Field(
        default=1,
        ge=0,
        le=10,
        description=(
            "Per-pick budget of additional perception captures "
            "triggered when the gate refuses. After exhaustion the "
            "pick surfaces NO_COMMIT_INSUFFICIENT_FUSION."
        ),
    )
    apply_modes: tuple[str, ...] = Field(
        default=("auto", "dense_clutter", "dense_autonomous"),
        description=(
            "Mode labels that activate the gate. Defaults to all modes "
            "except EASY. The orchestrator skips the gate (and the "
            "reobserve loop) when its resolved mode label is not in this set."
        ),
    )

    @model_validator(mode="after")
    def _validate(self) -> "RobotGraspingCommitPolicyConfig":
        if not self.apply_modes:
            raise ValueError(
                "commit_policy.apply_modes must contain at least one "
                "mode label"
            )
        if len(set(self.apply_modes)) != len(self.apply_modes):
            raise ValueError(
                "commit_policy.apply_modes must not contain duplicates"
            )
        if "easy" in self.apply_modes:
            raise ValueError(
                "commit_policy.apply_modes must not include 'easy' "
                "(locked Q5: EASY is exempt from mandatory multi-view "
                "decisioning)"
            )
        return self


class RobotGraspingApproachValidationConfig(StrictModel):
    """Dense-mode swept-volume approach/retreat validation.

    When ``enabled`` (and the resolved mode is in :attr:`apply_modes`), the orchestrator validates each
    ranked candidate's approach/retreat SWEEP against the scene neighbour cloud (the calculator only
    checks the FINAL grasp pose) and executes the first sweep-clear candidate, refusing with
    ``APPROACH_PATH_BLOCKED`` if every candidate is blocked. Default ``enabled=False`` is byte-identical;
    EASY/single-object are exempt (``apply_modes`` excludes ``'easy'``).
    """

    enabled: bool = Field(
        default=False,
        description="Top-level switch. When False the approach validator is a no-op (byte-identical).",
    )
    standoff_mm: float = Field(
        default=80.0, ge=0.0, le=500.0,
        description="Pre-grasp standoff (mm) along -approach; the sweep is sampled standoff -> grasp.",
    )
    retreat_mm: float = Field(
        default=100.0, ge=0.0, le=500.0,
        description="Retreat distance (mm) after the grasp; the retreat leg is swept along +Z.",
    )
    num_approach_samples: int = Field(
        default=6, ge=0, le=64,
        description="Swept samples on the approach leg (0 disables the approach check).",
    )
    num_retreat_samples: int = Field(
        default=4, ge=0, le=64,
        description="Swept samples on the retreat leg (0 disables the retreat check).",
    )
    collision_margin_mm: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Extra margin (mm) added to the swept gripper box when testing collisions.",
    )
    apply_modes: tuple[str, ...] = Field(
        default=("dense_clutter", "dense_autonomous"),
        description=(
            "Mode labels that activate the validator (the dense modes by default). The orchestrator "
            "skips the check when its resolved mode label is not in this set. EASY is locked out."
        ),
    )

    @model_validator(mode="after")
    def _validate(self) -> "RobotGraspingApproachValidationConfig":
        if not self.apply_modes:
            raise ValueError("approach_validation.apply_modes must contain at least one mode label")
        if len(set(self.apply_modes)) != len(self.apply_modes):
            raise ValueError("approach_validation.apply_modes must not contain duplicates")
        if "easy" in self.apply_modes:
            raise ValueError("approach_validation.apply_modes must not include 'easy'")
        return self


class CameraExtrinsicsConfig(StrictModel):
    """One camera's calibration entry in the central ``fusion.cameras`` map.

    Keyed by a camera id (the same name used elsewhere for the camera, e.g. a sim
    ``robot.sim.cameras`` key or a production rig id). Each camera declares HOW it is
    mounted and WHERE its persisted calibration artifact lives, so every camera in a
    multi-view rig is calibrated + loaded INDIVIDUALLY.
    """

    enabled: bool = Field(
        default=True,
        description="Whether this camera participates. Disabled cameras are skipped by the resolver map.",
    )
    mounting_mode: Literal["eye_to_hand", "eye_in_hand"] = Field(
        default="eye_to_hand",
        description=(
            "eye_to_hand = fixed camera; the artifact is a CAMERA->BASE Extrinsics "
            "(``save_extrinsics``) -> a StaticCameraToBaseResolver. eye_in_hand = wrist "
            "camera; the artifact is a CAMERA->TOOL transform (``save_cam_to_tool``) -> an "
            "EyeInHandFrameResolver that composes the live TCP each frame."
        ),
    )
    extrinsics_artifact_path: str = Field(
        description="Path to this camera's persisted calibration JSON (per mounting_mode).",
    )


class RobotGraspingFusionConfig(StrictModel):
    """Bounded multi-view fusion substrate config.

    Configures the deterministic voxel-occupancy grid that
    :class:`backend.src.robot.grasping.multiview.fusion.SceneFusion` maintains
    across perception captures within a pick session. The deployment posture is
    shadow-only: even with ``enabled=True`` the runtime merely ingests perception
    frames and emits telemetry; the commit gate is what turns fused evidence into
    dense-mode decisioning.

    All limits are hard caps. Validators enforce positive/finite numerics,
    ``depth_min_mm < depth_max_mm``, per-axis ROI extent a positive multiple of
    ``voxel_size_mm``, and ROI × voxel-size never exceeding ``max_voxels`` (so a
    misconfiguration cannot allocate a gigabyte-scale grid).
    """

    enabled: bool = Field(
        default=False,
        description=(
            "Top-level switch. When False the orchestrator behaves "
            "byte-identically: no fusion module is constructed, no "
            "telemetry emitted, no perception ingest performed."
        ),
    )
    max_views: int = Field(
        default=6,
        ge=1,
        le=64,
        description=(
            "Maximum number of accepted views retained simultaneously. "
            "Sliding FIFO: oldest accepted view evicted on overflow."
        ),
    )
    max_view_age_s: float = Field(
        default=20.0,
        gt=0.0,
        le=600.0,
        description=(
            "Accepted views older than this are dropped on the next "
            "ingest (wall-clock relative to the newest accepted view)."
        ),
    )
    voxel_size_mm: float = Field(
        default=6.0,
        gt=0.0,
        le=100.0,
        description="Cubic voxel edge length in millimetres.",
    )
    roi_extent_mm: tuple[float, float, float] = Field(
        default=(600.0, 600.0, 360.0),
        description=(
            "Total X/Y/Z extent (mm) of the fusion ROI, centered on "
            "the workspace origin. Each axis must be > 0 and an "
            "integer multiple of ``voxel_size_mm``."
        ),
    )
    max_voxels: int = Field(
        default=1_000_000,
        ge=1_000,
        le=8_000_000,
        description=(
            "Hard cap on total voxel count "
            "(``roi_x/voxel_size * roi_y/voxel_size * roi_z/voxel_size``)."
        ),
    )
    depth_min_mm: float = Field(
        default=80.0,
        gt=0.0,
        le=10_000.0,
        description="Minimum valid depth sample (mm). Below: rejected.",
    )
    depth_max_mm: float = Field(
        default=1_400.0,
        gt=0.0,
        le=10_000.0,
        description="Maximum valid depth sample (mm). Above: rejected.",
    )
    intrinsics_atol: float = Field(
        default=1e-6,
        ge=0.0,
        le=1.0,
        description=(
            "Absolute tolerance for per-element intrinsic-matrix drift "
            "between views (strict frame contract). Views whose "
            "intrinsics differ from the first accepted view by more "
            "than this are refused."
        ),
    )
    commit_policy: RobotGraspingCommitPolicyConfig = Field(
        default_factory=RobotGraspingCommitPolicyConfig,
        description=(
            "Mandatory dense multi-view commit gate. Defaults to "
            "``enabled=False`` (byte-identical)."
        ),
    )
    active_perception_use_fusion: bool = Field(
        default=False,
        description=(
            "Read-side wiring switch. When True the viewpoint planner "
            "may consume fused-state information-gain when scoring next "
            "viewpoints. Default False keeps the planner path "
            "byte-identical and isolates the commit-gate behavior change."
        ),
    )
    extrinsics_artifact_path: str | None = Field(
        default=None,
        description=(
            "Path to a persisted eye-to-hand Extrinsics JSON artifact (CAMERA -> BASE), as written "
            "by ``backend.src.calibration.serialization.save_extrinsics``. When set AND ``enabled`` is "
            "True, ``AutonomousGraspService.from_robot_config`` builds a StaticCameraToBaseResolver from "
            "the loaded transform so the multi-view fusion substrate + the commit gate become "
            "REACHABLE in production. ``None`` (default) builds no resolver (byte-identical). "
            "FAIL-CLOSED: a set-but-unloadable path raises at construction (a misconfigured fusion "
            "deployment must not silently run with an unreachable gate). Eye-in-hand cells leave this "
            "None and pass a ``frame_resolver`` kwarg in code (the live TCP-composed resolver cannot be "
            "serialized)."
        ),
    )
    cameras: dict[str, CameraExtrinsicsConfig] = Field(
        default_factory=dict,
        description=(
            "MULTI-CAMERA calibration map: camera_id -> {enabled, mounting_mode, extrinsics_artifact_path}. "
            "The central place to declare + individually calibrate several cameras (multi-view). "
            "``build_config_frame_resolvers`` turns it into a {camera_id -> FrameResolver} map (eye_to_hand "
            "-> StaticCameraToBaseResolver, eye_in_hand -> EyeInHandFrameResolver). Empty (default) = the "
            "legacy single-camera path via ``extrinsics_artifact_path`` (byte-identical). FAIL-CLOSED: an "
            "enabled camera with a missing/invalid artifact raises at construction."
        ),
    )

    @model_validator(mode="after")
    def _validate(self) -> "RobotGraspingFusionConfig":
        if self.depth_min_mm >= self.depth_max_mm:
            raise ValueError(
                "fusion.depth_min_mm must be strictly less than "
                f"depth_max_mm; got {self.depth_min_mm} >= "
                f"{self.depth_max_mm}"
            )
        if len(self.roi_extent_mm) != 3:
            raise ValueError(
                "fusion.roi_extent_mm must have exactly 3 components "
                f"(x, y, z); got {self.roi_extent_mm!r}"
            )
        if any((not isinstance(v, (int, float))) or v <= 0.0 for v in self.roi_extent_mm):
            raise ValueError(
                "fusion.roi_extent_mm components must all be > 0; got "
                f"{self.roi_extent_mm!r}"
            )
        vs = float(self.voxel_size_mm)
        eps = 1e-9
        for axis_label, extent in zip("xyz", self.roi_extent_mm):
            ratio = float(extent) / vs
            if abs(ratio - round(ratio)) > eps:
                raise ValueError(
                    f"fusion.roi_extent_mm[{axis_label}]={extent} must "
                    f"be an integer multiple of voxel_size_mm={vs}"
                )
        total_voxels = 1
        for extent in self.roi_extent_mm:
            total_voxels *= int(round(float(extent) / vs))
        if total_voxels > self.max_voxels:
            raise ValueError(
                "fusion ROI/voxel combo would allocate "
                f"{total_voxels} voxels which exceeds max_voxels="
                f"{self.max_voxels}. Coarsen voxel_size_mm or shrink "
                "roi_extent_mm."
            )
        return self


class GraspingParallelJawGeometryConfig(StrictModel):
    """Parallel-jaw collision envelope (mm) in the local grasp frame (X closes, Y binormal, Z approach).

    Defaults match the shipped 2F-85-class box model; every value flows straight into
    :class:`backend.src.robot.grasping.collision.ParallelJawGripperModel`.
    """

    finger_length_mm: float = Field(default=60.0, gt=0.0, le=500.0)
    finger_thickness_mm: float = Field(default=12.0, gt=0.0, le=200.0)
    finger_width_mm: float = Field(default=22.0, gt=0.0, le=200.0)
    finger_pad_overlap_mm: float = Field(default=2.0, ge=0.0, le=100.0)
    fingertip_depth_mm: float = Field(default=6.0, gt=0.0, le=200.0)
    palm_depth_mm: float = Field(default=35.0, gt=0.0, le=500.0)
    palm_width_mm: float = Field(default=70.0, gt=0.0, le=500.0)


class GraspingSuctionCupGeometryConfig(StrictModel):
    """Suction-cup collision envelope (mm): cup, then shaft, then wrist mount back along -approach.

    Feeds :class:`backend.src.robot.grasping.collision.SuctionCupGripperModel`; defaults model a
    ~30 mm-diameter cup on a slim shaft with a wrist coupler.
    """

    cup_radius_mm: float = Field(default=15.0, gt=0.0, le=200.0)
    cup_height_mm: float = Field(default=25.0, gt=0.0, le=200.0)
    contact_tip_depth_mm: float = Field(default=2.0, ge=0.0, le=100.0)
    shaft_radius_mm: float = Field(default=10.0, gt=0.0, le=200.0)
    shaft_length_mm: float = Field(default=40.0, gt=0.0, le=500.0)
    mount_radius_mm: float = Field(default=30.0, gt=0.0, le=300.0)
    mount_depth_mm: float = Field(default=20.0, gt=0.0, le=300.0)


class GraspingGripperGeometryConfig(StrictModel):
    """Which gripper collision envelope the calculator filters grasp candidates against.

    ``kind`` picks the active model and the matching sub-block sizes it; ``outer_margin_mm`` inflates
    whichever is chosen. Defaults reproduce the historically hardcoded parallel-jaw envelope
    byte-for-byte, so leaving this unset changes nothing. Set ``kind: suction`` for a vacuum
    end-effector so suction candidates are checked against a cup envelope, not a finger envelope.
    """

    kind: Literal["parallel_jaw", "suction"] = Field(default="parallel_jaw")
    outer_margin_mm: float = Field(default=0.0, ge=0.0, le=100.0)
    parallel_jaw: GraspingParallelJawGeometryConfig = Field(
        default_factory=GraspingParallelJawGeometryConfig
    )
    suction: GraspingSuctionCupGeometryConfig = Field(
        default_factory=GraspingSuctionCupGeometryConfig
    )


class RobotGraspingConfig(StrictModel):
    """Vendor-neutral grasping-behaviour surface.

    Sits as a sibling of :class:`RobotSafetyConfig` under :class:`RobotConfig`.
    All knobs are *behavioural* (mode selection, closed-loop refinement,
    verification, recovery, decision); safety remains exclusively under
    ``robot.safety``.

    The runtime derives per-mode behaviour from the
    :class:`backend.src.robot.execution.autonomous_grasp.GraspBehaviorProfile`
    table. Values here apply *on top* of that profile: they tighten or disable
    optional behaviour, but cannot relax safety limits or unlock recovery actions
    the mode profile forbids. ``default_mode`` is resolved against
    :class:`backend.src.robot.grasping.types.modes.GraspMode` at use time so the config
    layer imports no runtime modules.
    """

    # Stable :class:`GraspMode` name resolved at runtime. One of: "easy", "auto", "closed_loop",
    # "dense_clutter", "dense_autonomous".
    default_mode: str = Field(default="auto", min_length=1)
    # Upper bound on autonomous attempts per pick() call.
    max_attempts: int = Field(default=5, ge=1, le=50)
    #: When a silhouette is near-isotropic its principal axis is numerically degenerate, so aim the jaw
    #: along the RADIAL direction (base -> grasp) instead of following the noise. This is a property of
    #: the ARM, not of the scene: MEASURED on-box 2026-07-24, a UR3e real-vision (M2) cell goes
    #: **5/10 -> 10/10** with it on (a vision-detected grasp has a varying azimuth, and a UR3e cannot
    #: plan a tangential top-down close at any azimuth: 0/4 vs 4/4 radial). A UR5e absorbs both, so the
    #: default stays False and only a cell that needs it declares it (robot.ur3e.yaml does).
    isotropic_radial_closing: bool = Field(default=False)
    record_log_path: str | None = Field(
        default=None,
        description=(
            "Opt-in JSONL path for production GraspAttemptRecord logging. When set, "
            "AutonomousGraspService.from_robot_config / from_components call "
            "enable_record_logging(path), so every pick() appends one record (the soak / KPI / RL "
            "data source). Default null -> off, byte-identical. The loader applies ${ENV} "
            'substitution, e.g. record_log_path: "${WILLY_RECORD_LOG:-}" -> an empty (unset) value '
            "is treated as off."
        ),
    )
    closed_loop: GraspingClosedLoopConfig = Field(
        default_factory=GraspingClosedLoopConfig
    )
    verification: GraspingVerificationConfig = Field(
        default_factory=GraspingVerificationConfig
    )
    dense_recovery: GraspingDenseRecoveryConfig = Field(
        default_factory=GraspingDenseRecoveryConfig
    )
    decision: GraspingDecisionConfig = Field(
        default_factory=GraspingDecisionConfig
    )
    feasibility: GraspingFeasibilityConfig = Field(
        default_factory=GraspingFeasibilityConfig
    )
    occlusion: GraspingOcclusionConfig = Field(
        default_factory=GraspingOcclusionConfig
    )
    ordering: GraspingOrderingConfig = Field(
        default_factory=GraspingOrderingConfig
    )
    recovery: GraspingRecoveryConfig = Field(
        default_factory=GraspingRecoveryConfig
    )
    uncertainty: GraspingUncertaintyConfig = Field(
        default_factory=GraspingUncertaintyConfig
    )
    success_model: GraspingSuccessModelConfig = Field(
        default_factory=GraspingSuccessModelConfig
    )
    watchdog: GraspingWatchdogConfig = Field(
        default_factory=GraspingWatchdogConfig
    )
    performance: GraspingPerformanceConfig = Field(
        default_factory=GraspingPerformanceConfig
    )
    fusion: RobotGraspingFusionConfig = Field(
        default_factory=RobotGraspingFusionConfig
    )
    approach_validation: RobotGraspingApproachValidationConfig = Field(
        default_factory=RobotGraspingApproachValidationConfig,
        description=(
            "Dense-mode swept-volume approach/retreat validator. Default disabled "
            "(byte-identical); independent of fusion (it needs only the per-frame neighbour masks)."
        ),
    )
    gripper_geometry: GraspingGripperGeometryConfig = Field(
        default_factory=GraspingGripperGeometryConfig,
        description=(
            "Gripper collision envelope the calculator filters grasp candidates against "
            "(parallel-jaw or suction-cup). Default parallel_jaw with the shipped dims -> byte-identical."
        ),
    )
