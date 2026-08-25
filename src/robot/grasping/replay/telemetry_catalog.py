"""Define frozen telemetry-completeness requirements and optional extensions.

Maps each ``AutonomousGraspOutcome`` to the additional
``GraspAttemptRecord`` fields required for incident-grade telemetry.
Top-level fields and ``extra.<key>`` entries are validated uniformly;
universal record fields are already enforced by the record itself.

Also catalogs optional extension telemetry for capabilities such as success
probability, multi-view fusion, uncertainty, drift/OOD, runtime SLOs, failure
taxonomy, adaptation, and model lifecycle.
"""


from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from src.robot.execution.autonomous_grasp import AutonomousGraspOutcome
from src.robot.grasping.telemetry.outcome_logging import GraspAttemptRecord


_O = AutonomousGraspOutcome


TELEMETRY_CATALOG: dict[str, frozenset[str]] = {
    _O.SUCCEEDED.value: frozenset({"execution", "verification"}),
    _O.NO_TARGET.value: frozenset(),
    # A cancelled attempt requires nothing: it stopped before it could produce an execution block, and
    # demanding one would make every operator stop look like a malformed record.
    _O.CANCELLED.value: frozenset(),
    _O.NO_VALID_GRASP.value: frozenset(),
    _O.EXECUTION_FAILED.value: frozenset({"execution"}),
    _O.REFINEMENT_FAILED.value: frozenset({"refinement"}),
    _O.VERIFICATION_FAILED.value: frozenset({"execution", "verification"}),
    _O.RECOVERY_EXHAUSTED.value: frozenset({"recovery_actions"}),
    _O.UNSAFE_RECOVERY_REFUSED.value: frozenset({"recovery_actions"}),
    _O.MISSING_CAMERA_FRAME.value: frozenset(),
    _O.TARGET_LOST_DURING_REFINE.value: frozenset({"refinement"}),
    _O.REFINEMENT_DIVERGED.value: frozenset({"refinement"}),
    _O.MODE_NOT_AVAILABLE.value: frozenset(),
    _O.DECISION_FAIL_CLOSED.value: frozenset(
        {
            "extra.decision_reason_code",
            "extra.uncertainty_score",
            "extra.threshold_used",
        }
    ),
    _O.DECISION_RECOVER_PENDING.value: frozenset(
        {"extra.decision_reason_code"}
    ),
    _O.UNCERTAINTY_FAIL_CLOSED.value: frozenset(
        {
            "extra.decision_reason_code",
            "extra.uncertainty_score",
            "extra.threshold_used",
        }
    ),
    _O.NO_COMMIT_INSUFFICIENT_FUSION.value: frozenset(),
    # Drift / OOD watchdog terminal outcomes. Both require the
    # watchdog telemetry triple so a replay can attribute the
    # fail-closure cleanly.
    _O.DRIFT_BLOCKED_AUTO.value: frozenset(
        {
            "extra.decision_reason_code",
            "extra.drift_severity",
            "extra.degraded_mode_active",
        }
    ),
    _O.OOD_BLOCKED_AUTO.value: frozenset(
        {
            "extra.decision_reason_code",
            "extra.ood_flagged",
            "extra.degraded_mode_active",
        }
    ),
}


def _field_present(record: GraspAttemptRecord, name: str) -> bool:
    if name.startswith("extra."):
        key = name[len("extra.") :]
        extra = record.extra or {}
        return key in extra and extra[key] is not None
    value = getattr(record, name, None)
    if value is None:
        return False
    # Empty sequences (e.g. recovery_actions=()) count as missing.
    if isinstance(value, (tuple, list)) and len(value) == 0:
        return False
    if isinstance(value, dict) and len(value) == 0:
        return False
    return True


def audit_record(record: GraspAttemptRecord) -> tuple[str, ...]:
    """Return the names of required fields that are missing."""

    required = TELEMETRY_CATALOG.get(record.final_outcome)
    if required is None:
        # Unknown outcome is itself an audit failure surfaced as the
        # special ``final_outcome`` token so callers can distinguish
        # it from a recognised-but-incomplete record.
        return ("final_outcome:unknown",)
    return tuple(
        sorted(name for name in required if not _field_present(record, name))
    )


def audit_records(
    records: Iterable[GraspAttemptRecord] | Sequence[GraspAttemptRecord],
) -> list[tuple[str, tuple[str, ...]]]:
    """Return ``(attempt_id, missing_fields)`` for every offender."""

    offenders: list[tuple[str, tuple[str, ...]]] = []
    for r in records:
        missing = audit_record(r)
        if missing:
            offenders.append((r.attempt_id, missing))
    return offenders


#: Frozen contract version. Bumped only on additive field additions
#: that downstream tools must learn to read. Removing or retyping any
#: field already published here is a breaking change and requires a
#: deliberate, documented schema-version bump on
#: :class:`GraspAttemptRecord` itself.
EXTRA_TELEMETRY_VERSION: int = 1


def _is_str_or_none(v: Any) -> bool:
    return v is None or isinstance(v, str)


def _is_bool_or_none(v: Any) -> bool:
    return v is None or isinstance(v, bool)


def _is_finite_number_or_none(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, bool):  # bool is a subclass of int; reject here
        return False
    if not isinstance(v, (int, float)):
        return False
    f = float(v)
    return f == f and f not in (float("inf"), float("-inf"))


def _is_unit_interval_or_none(v: Any) -> bool:
    if not _is_finite_number_or_none(v):
        return False
    return v is None or 0.0 <= float(v) <= 1.0


def _is_non_negative_int_or_none(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, bool):
        return False
    return isinstance(v, int) and v >= 0


def _is_non_negative_number_or_none(v: Any) -> bool:
    if not _is_finite_number_or_none(v):
        return False
    return v is None or float(v) >= 0.0


def _is_string_float_map_or_none(v: Any) -> bool:
    if v is None:
        return True
    if not isinstance(v, Mapping):
        return False
    for key, value in v.items():
        if not isinstance(key, str):
            return False
        if not _is_finite_number_or_none(value) or value is None:
            return False
    return True


def _is_list_of_dicts_or_none(v: Any) -> bool:
    if v is None:
        return True
    if not isinstance(v, (list, tuple)):
        return False
    return all(isinstance(item, Mapping) for item in v)


def _is_kendall_tau_or_none(v: Any) -> bool:
    """float in ``[-1.0, 1.0]`` or ``None``."""

    if v is None:
        return True
    if not _is_finite_number_or_none(v):
        return False
    return -1.0 <= float(v) <= 1.0


_DRIFT_SEVERITY_VALUES: frozenset[str] = frozenset(
    {"none", "low", "moderate", "high", "severe"}
)
_ADAPTATION_MODE_VALUES: frozenset[str] = frozenset(
    {"off", "recommend_only", "apply_with_guardrails"}
)
_MODEL_LIFECYCLE_VALUES: frozenset[str] = frozenset(
    {"shadow", "canary", "active"}
)

# RL optimisation extension layer (telemetry contract). RL telemetry
# fields are required immediately on every RL-influenced outcome.
# "RL-influenced" means ``record.extra['rl_mode']`` is one of the
# RL-active modes (``rl_shadow``/``rl_active``/``rl_experimental``).

_RL_MODE_VALUES: frozenset[str] = frozenset(
    {
        "geometry_only",
        "hybrid_ml",
        "rl_shadow",
        "rl_active",
        "rl_experimental",
    }
)
_RL_ACTIVE_MODE_VALUES: frozenset[str] = frozenset(
    {"rl_shadow", "rl_active", "rl_experimental"}
)
_RL_ACTION_TOKENS: frozenset[str] = frozenset(
    {"grasp", "retry", "reobserve", "recover", "skip", "accept", "noop"}
)
_RL_ROUTER_PATH_VALUES: frozenset[str] = frozenset(
    {
        "geometry_only",
        "hybrid_ml",
        "rl_shadow",
        "rl_active",
        "rl_experimental",
        "deformable_specialist",
        "fallback_baseline",
    }
)


def _is_rl_mode_or_none(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v in _RL_MODE_VALUES)


def _is_rl_action_or_none(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v in _RL_ACTION_TOKENS)


def _is_rl_router_path_or_none(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v in _RL_ROUTER_PATH_VALUES)


def _is_drift_severity_or_none(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v in _DRIFT_SEVERITY_VALUES)


def _is_adaptation_mode_or_none(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v in _ADAPTATION_MODE_VALUES)


def _is_model_lifecycle_or_none(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v in _MODEL_LIFECYCLE_VALUES)


#: Per-field validator + capability-group tag. The tag is informational
#: today; a future strict gate may reference it when promoting a field
#: to required.
EXTRA_TELEMETRY_FIELDS: tuple[tuple[str, str, Any], ...] = (
    # success probability + calibration
    ("predicted_success_probability", "success_probability", _is_unit_interval_or_none),
    ("success_probability_model_version", "success_probability", _is_str_or_none),
    ("success_probability_calibration_bin", "success_probability", _is_non_negative_int_or_none),
    ("model_lifecycle_phase", "success_probability", _is_model_lifecycle_or_none),
    # multi-view fusion
    ("fused_view_count", "multi_view_fusion", _is_non_negative_int_or_none),
    ("fusion_evidence_quality", "multi_view_fusion", _is_unit_interval_or_none),
    ("multi_view_occlusion_reduced", "multi_view_fusion", _is_bool_or_none),
    # failure taxonomy + recommendations
    ("failure_taxonomy_class", "failure_taxonomy", _is_str_or_none),
    ("failure_recommendation_id", "failure_taxonomy", _is_str_or_none),
    # uncertainty as runtime driver. ``uncertainty_score`` is *already*
    # required for ``DECISION_FAIL_CLOSED`` by the strict catalog above;
    # this entry only adds the *type* constraint for all other outcomes
    # when the field happens to be present.
    ("uncertainty_score", "uncertainty", _is_unit_interval_or_none),
    ("uncertainty_channels", "uncertainty", _is_string_float_map_or_none),
    ("uncertainty_disagreement", "uncertainty", _is_unit_interval_or_none),
    # drift / OOD watchdog
    ("drift_severity", "drift_ood", _is_drift_severity_or_none),
    ("ood_flagged", "drift_ood", _is_bool_or_none),
    ("degraded_mode_active", "drift_ood", _is_bool_or_none),
    # runtime SLOs
    ("decision_latency_ms", "latency", _is_non_negative_number_or_none),
    ("ranking_latency_ms", "latency", _is_non_negative_number_or_none),
    ("fusion_latency_ms", "latency", _is_non_negative_number_or_none),
    ("attempt_wall_time_s", "latency", _is_non_negative_number_or_none),
    # guarded adaptation
    ("adaptation_mode", "guarded_adaptation", _is_adaptation_mode_or_none),
    ("adaptation_applied_changes", "guarded_adaptation", _is_list_of_dicts_or_none),
    # RL optimisation extension layer.
    ("rl_mode", "rl_core", _is_rl_mode_or_none),
    ("rl_policy_id", "rl_core", _is_str_or_none),
    ("rl_artifact_version", "rl_core", _is_str_or_none),
    ("rl_action_proposed", "rl_core", _is_rl_action_or_none),
    ("rl_action_applied", "rl_core", _is_rl_action_or_none),
    ("rl_action_blocked_by_mask", "rl_core", _is_bool_or_none),
    ("rl_reason_features", "rl_core", _is_string_float_map_or_none),
    ("rl_confidence", "rl_core", _is_unit_interval_or_none),
    ("rl_baseline_action", "rl_core", _is_rl_action_or_none),
    ("rl_override", "rl_core", _is_bool_or_none),
    ("rl_fallback_triggered", "rl_core", _is_bool_or_none),
    ("rl_router_path", "rl_core", _is_rl_router_path_or_none),
    ("rl_fallback_reason_code", "rl_core", _is_str_or_none),
    # candidate-selection shadow router.
    ("rl_candidate_breakdown", "rl_candidate", _is_list_of_dicts_or_none),
    ("rl_candidate_agreement_top1", "rl_candidate", _is_bool_or_none),
    ("rl_candidate_agreement_kendall_tau", "rl_candidate", _is_kendall_tau_or_none),
    ("rl_candidate_mask_total", "rl_candidate", _is_non_negative_int_or_none),
    ("rl_candidate_pruned_count", "rl_candidate", _is_non_negative_int_or_none),
    # ranking shadow router.
    ("rl_ranking_policy_id", "rl_ranking", _is_str_or_none),
    ("rl_ranking_artifact_version", "rl_ranking", _is_str_or_none),
    ("rl_ranking_regret_top1", "rl_ranking", _is_bool_or_none),
    ("rl_ranking_kendall_tau", "rl_ranking", _is_kendall_tau_or_none),
    # sequencing shadow router.
    ("rl_sequencing_policy_id", "rl_sequencing", _is_str_or_none),
    ("rl_sequencing_artifact_version", "rl_sequencing", _is_str_or_none),
    ("rl_sequencing_action_proposed", "rl_sequencing", _is_str_or_none),
    ("rl_sequencing_action_baseline", "rl_sequencing", _is_str_or_none),
    ("rl_sequencing_action_agree", "rl_sequencing", _is_bool_or_none),
)


def extra_field_names() -> tuple[str, ...]:
    """Return the ordered tuple of extra-bag telemetry field names."""

    return tuple(name for name, _group, _validator in EXTRA_TELEMETRY_FIELDS)


def extra_field_group_map() -> dict[str, str]:
    """Return ``{field_name: capability_group}`` for every extra telemetry field."""

    return {name: group for name, group, _validator in EXTRA_TELEMETRY_FIELDS}


def audit_extra_record(record: GraspAttemptRecord) -> tuple[str, ...]:
    """Type-only audit of the extra telemetry fields: return the names whose *present* value violates the locked type contract."""

    extra = record.extra or {}
    offenders: list[str] = []
    for name, _group, validator in EXTRA_TELEMETRY_FIELDS:
        if name not in extra:
            continue
        if not validator(extra[name]):
            offenders.append(name)
    return tuple(offenders)


def audit_extra_records(
    records: Iterable[GraspAttemptRecord] | Sequence[GraspAttemptRecord],
) -> list[tuple[str, tuple[str, ...]]]:
    """Return ``(attempt_id, bad_fields)`` for every extra-field type violation."""

    offenders: list[tuple[str, tuple[str, ...]]] = []
    for r in records:
        bad = audit_extra_record(r)
        if bad:
            offenders.append((r.attempt_id, bad))
    return offenders


def extra_field_coverage(
    records: Iterable[GraspAttemptRecord] | Sequence[GraspAttemptRecord],
) -> dict[str, float]:
    """Return ``{field_name: fraction_of_records_with_non_null_value}`` (``0.0`` for every field on empty input)."""

    records = tuple(records)
    total = len(records)
    coverage: dict[str, float] = {}
    for name, _group, _validator in EXTRA_TELEMETRY_FIELDS:
        if total == 0:
            coverage[name] = 0.0
            continue
        present = 0
        for r in records:
            extra = r.extra or {}
            if name in extra and extra[name] is not None:
                present += 1
        coverage[name] = present / total
    return coverage


#: Names of every RL extra field. Order matches
#: :data:`EXTRA_TELEMETRY_FIELDS` insertion order.
RL_REQUIRED_TELEMETRY_FIELDS: tuple[str, ...] = (
    "rl_mode",
    "rl_policy_id",
    "rl_artifact_version",
    "rl_action_proposed",
    "rl_action_applied",
    "rl_action_blocked_by_mask",
    "rl_reason_features",
    "rl_confidence",
    "rl_baseline_action",
    "rl_override",
    "rl_fallback_triggered",
    "rl_router_path",
    "rl_fallback_reason_code",
)


def audit_rl_required_record(record: GraspAttemptRecord) -> tuple[str, ...]:
    """Missing RL fields when the record is RL-influenced (``rl_mode`` an RL-active mode), else empty tuple."""

    extra = record.extra or {}
    mode_value = extra.get("rl_mode")
    if mode_value not in _RL_ACTIVE_MODE_VALUES:
        return ()
    missing: list[str] = []
    for name in RL_REQUIRED_TELEMETRY_FIELDS:
        if name not in extra or extra[name] is None:
            missing.append(name)
    return tuple(missing)


def audit_rl_required_records(
    records: Iterable[GraspAttemptRecord] | Sequence[GraspAttemptRecord],
) -> list[tuple[str, tuple[str, ...]]]:
    """Return ``(attempt_id, missing_fields)`` for every RL presence violation."""

    offenders: list[tuple[str, tuple[str, ...]]] = []
    for r in records:
        missing = audit_rl_required_record(r)
        if missing:
            offenders.append((r.attempt_id, missing))
    return offenders
