"""Conservative offline promotion gate for off-policy-judgeable RL families.

Provides a deterministic, runtime-independent promotion verdict for candidate
policy artifacts evaluated against canonical replay packs. The gate extracts
policy-specific ``(behavior_action, target_action, reward)`` triples and
combines WIS with auditable Direct-Method estimates without touching runtime,
safety, replay, or router surfaces.

Off-policy-judgeable families are registered centrally through
``FAMILY_REGISTRY``; sequencing, perception-budget, and recovery are supported
while candidate and ranking remain measurement-only because their replay data
does not contain the required behavior-action identity.

WIS is the sole estimator driving the verdict. The reported DM estimates are
informational: the legacy target-policy self-estimate is explicitly
circular, while ``fitted_q_dm`` is an independent tabular fit over logged
``(state, action)`` rewards with coverage surfaced for auditability. FQE
remains a documented future seam.

Promotion is conservative: the candidate must exceed the configured baseline
lift floor with a sufficient effective sample size and a bootstrap-CI lower
bound meeting the required threshold. Policies proposing fewer than two
distinct actions abstain rather than receiving a degenerate pass.

Artifacts carry reproducibility metadata including dataset/config hashes,
seed, and best-effort source commit identity. Promotion produces a report
only; it never mutates policy artifacts.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from src.robot.grasping.constants import (
    RL_PROMOTION_LOG_FILE,
    create_grasping_logger,
)

from .honesty import build_promotion_honesty
from .online_state import is_online_mutated_artifact
from .ope import (
    BEHAVIOUR_POLICY_EPSILON,
    BOOTSTRAP_N,
    IMPORTANCE_WEIGHT_CLIP,
    ConfidenceInterval,
)
from .perception_budget_policy import (
    PERCEPTION_ACTIONS,
    load_linucb_perception_budget_policy,
)
from .recovery_policy import (
    RECOVERY_ACTIONS,
    load_linucb_recovery_policy,
)
from .sequencing_policy import (
    SEQUENCING_ACTIONS,
    load_lookup_table_sequencing_policy,
)
from ._artifact_io import hash_artifact as hash_file
from ._io import load_jsonl
from ._promotion_extractors import (
    EvaluationTriple,
    TripleExtractionResult,
    extract_sequencing_triples,
    extract_perception_triples,
    extract_recovery_triples,
)
from ._promotion_estimators import (
    DMEstimate,
    IndependentDMEstimate,
    WISLiftEstimate,
    _per_record_weight,  # noqa: F401 - re-exported for test_v7's by-name import
    compute_dm,
    compute_wis_lift,
    fit_tabular_dm,
)


PROMOTION_REPORT_SCHEMA_VERSION: int = 2

#: Numerical tolerance for floating-point comparisons in the gate.
#: Lifts and CI bounds within ``+/-`` this band of a threshold are
#: treated as meeting that threshold. Without this, WIS values that
#: are mathematically zero but arrive as ``-5e-15`` after summation
#: would erroneously trip ``"lift_below_threshold"``.
PROMOTION_NUMERIC_TOLERANCE: float = 1e-9

#: Verdict literals.
PROMOTION_VERDICT_PASS: str = "pass"
PROMOTION_VERDICT_FAIL: str = "fail"
PROMOTION_VERDICT_ABSTAIN: str = "abstain"
PROMOTION_VERDICTS: tuple[str, ...] = (
    PROMOTION_VERDICT_PASS,
    PROMOTION_VERDICT_FAIL,
    PROMOTION_VERDICT_ABSTAIN,
)

#: Registry keys. ``POLICY_FAMILIES`` is DERIVED from ``FAMILY_REGISTRY`` (defined below, once the
#: extractors exist) so the recognized-family set and the dispatch table can never drift apart.
POLICY_FAMILY_SEQUENCING: str = "v4_sequencing"
POLICY_FAMILY_PERCEPTION: str = "v5_perception_budget"
POLICY_FAMILY_RECOVERY: str = "v6_recovery"


# ---------------------------------------------------------------------------
# Exceptions.
# ---------------------------------------------------------------------------


class PromotionInputError(ValueError):
    """Raised when promotion inputs are malformed or inconsistent."""


# ---------------------------------------------------------------------------
# Thresholds + verdict.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromotionThresholds:
    """Conservative defaults; see the module docstring."""

    # A positive point-lift floor so a mathematically-zero (or float-noise) lift can no longer pass.
    # Promotion requires a real improvement, not "does no harm". The CI lower-bound floor stays 0.0.
    min_lift_over_baseline: float = 0.01
    min_lower_bound_lift: float = 0.0
    min_n_effective: float = 10.0
    min_records_with_weight: int = 5

    def __post_init__(self) -> None:
        if self.min_n_effective < 0:
            raise PromotionInputError("min_n_effective must be >= 0")
        if self.min_records_with_weight < 0:
            raise PromotionInputError("min_records_with_weight must be >= 0")


#: The promotion verdict is the gate between an offline artifact and a cell that
#: may act on it. ABSTAIN and FAIL are returned, never raised, and they are easy to
#: read past in a JSON report so each one is stated here with the numbers that
#: produced it.
logger = create_grasping_logger("RLPromotion", RL_PROMOTION_LOG_FILE)


def evaluate_verdict(
    *,
    wis: WISLiftEstimate,
    thresholds: PromotionThresholds,
    n_distinct_target_actions: int | None = None,
) -> tuple[str, tuple[str, ...]]:
    """Apply the gate to a WIS lift estimate.

    Returns ``(verdict, reasons)``. ``reasons`` is a tuple of typed
    failure-reason codes (empty when verdict is ``"pass"``).
    """

    reasons: list[str] = []
    # Abstain conditions (insufficient evidence) take priority.
    if wis.num_records == 0:
        logger.warning("Promotion ABSTAIN: the evaluation set has no records")
        return (PROMOTION_VERDICT_ABSTAIN, ("no_records",))
    if n_distinct_target_actions is not None and n_distinct_target_actions < 2:
        # The honest-abstain guard. Without it a do-nothing policy scores a green
        # pass, so it is worth saying out loud every time it fires.
        logger.warning(
            "Promotion ABSTAIN: the target policy proposes only %d distinct "
            "action(s) over %d record(s) degenerate, not judgeable",
            n_distinct_target_actions,
            wis.num_records,
        )
        return (PROMOTION_VERDICT_ABSTAIN, ("degenerate_single_action_policy",))
    if wis.num_records_with_weight < thresholds.min_records_with_weight:
        reasons.append("insufficient_records_with_weight")
    if wis.effective_sample_size < thresholds.min_n_effective:
        reasons.append("insufficient_effective_sample_size")
    if reasons:
        logger.warning(
            "Promotion ABSTAIN (%s): %d record(s) with weight (min %d), effective "
            "sample size %.2f (min %.2f)",
            ", ".join(reasons),
            wis.num_records_with_weight,
            thresholds.min_records_with_weight,
            wis.effective_sample_size,
            thresholds.min_n_effective,
        )
        return (PROMOTION_VERDICT_ABSTAIN, tuple(reasons))
    # Fail conditions.
    if wis.lift < thresholds.min_lift_over_baseline - PROMOTION_NUMERIC_TOLERANCE:
        reasons.append("lift_below_threshold")
    if (
        wis.lift_ci.lower
        < thresholds.min_lower_bound_lift - PROMOTION_NUMERIC_TOLERANCE
    ):
        reasons.append("lift_lower_bound_below_threshold")
    if reasons:
        logger.warning(
            "Promotion FAIL (%s): lift %.4f (min %.4f), CI lower bound %.4f "
            "(min %.4f) over %d record(s)",
            ", ".join(reasons),
            wis.lift,
            thresholds.min_lift_over_baseline,
            wis.lift_ci.lower,
            thresholds.min_lower_bound_lift,
            wis.num_records,
        )
        return (PROMOTION_VERDICT_FAIL, tuple(reasons))
    logger.info(
        "Promotion PASS: lift %.4f (CI lower %.4f) over %d record(s), effective "
        "sample size %.2f",
        wis.lift,
        wis.lift_ci.lower,
        wis.num_records,
        wis.effective_sample_size,
    )
    return (PROMOTION_VERDICT_PASS, ())


# ---------------------------------------------------------------------------
# Report.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromotionReport:
    """All inputs + estimates + verdict for one (policy, dataset, seed)."""

    schema_version: int
    policy_family: str
    policy_id: str
    policy_version: int
    policy_artifact_sha256: str
    dataset_id: str
    dataset_hash: str
    dataset_paths: tuple[str, ...]
    training_scope: str
    seed: int
    config_hash: str
    code_commit: str
    thresholds: PromotionThresholds
    wis: WISLiftEstimate
    dm: DMEstimate
    independent_dm: IndependentDMEstimate
    extraction: TripleExtractionResult
    verdict: str
    reasons: tuple[str, ...]
    generated_at_iso: str


def compute_config_hash(config: Mapping[str, Any]) -> str:
    """Stable sha256 over a sort-keys JSON dump of ``config``."""

    blob = json.dumps(dict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def resolve_code_commit(repo_root: Optional[Path] = None) -> str:
    """Best-effort ``git rev-parse HEAD``; ``"unknown"`` on any failure."""

    cwd = str(repo_root) if repo_root is not None else None
    try:
        proc = subprocess.run(  # noqa: S603 fixed args, no shell
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=False,
            capture_output=True,
            timeout=5.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    if proc.returncode != 0:
        return "unknown"
    out = proc.stdout.decode("utf-8", errors="replace").strip()
    if not out or any(c.isspace() for c in out):
        return "unknown"
    return out


def hash_pack_paths(
    paths: Sequence[Path], *, repo_root: Optional[Path] = None
) -> str:
    """Deterministic hash over sorted path-strings + file contents.

    When ``repo_root`` is provided, paths within the repo are hashed
    via their POSIX repo-relative form so the digest is portable
    across machines.
    """

    def _key(p: Path) -> str:
        if repo_root is None:
            return str(p)
        try:
            return p.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            return str(p)

    h = hashlib.sha256()
    for p in sorted(paths, key=_key):
        h.update(_key(p).encode("utf-8"))
        h.update(b"\x00")
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Family registry.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FamilySpec:
    """One promotable policy family: how to load it, extract its triples, and size its action space.

    A registry (rather than a hard-coded dispatch) so a new judgeable family is a single registration
    rather than a new code branch. NOTE: only families whose behaviour action is recoverable from the
    logs belong here candidate / ranking are NOT registered because the canonical replay packs never
    logged the per-candidate behaviour id, so their off-policy WIS collapses to the baseline mean (a
    structurally-guaranteed non-verdict); they stay measurement-only in ``ope.py``.
    """

    loader: Callable[[Path], Any]
    extractor: Callable[..., TripleExtractionResult]
    num_actions: int


FAMILY_REGISTRY: dict[str, _FamilySpec] = {
    POLICY_FAMILY_SEQUENCING: _FamilySpec(
        loader=load_lookup_table_sequencing_policy,
        extractor=extract_sequencing_triples,
        num_actions=len(SEQUENCING_ACTIONS),
    ),
    POLICY_FAMILY_PERCEPTION: _FamilySpec(
        loader=load_linucb_perception_budget_policy,
        extractor=extract_perception_triples,
        num_actions=len(PERCEPTION_ACTIONS),
    ),
    POLICY_FAMILY_RECOVERY: _FamilySpec(
        loader=load_linucb_recovery_policy,
        extractor=extract_recovery_triples,
        num_actions=len(RECOVERY_ACTIONS),
    ),
}

#: The recognized promotable families, DERIVED from the registry so it can never drift from the dispatch.
POLICY_FAMILIES: tuple[str, ...] = tuple(FAMILY_REGISTRY)


def evaluate_policy_for_promotion(
    *,
    policy_family: str,
    policy_artifact_path: Path,
    pack_paths: Sequence[Path],
    dataset_id: str,
    training_scope: str,
    seed: int,
    thresholds: Optional[PromotionThresholds] = None,
    repo_root: Optional[Path] = None,
    generated_at: Optional[datetime] = None,
) -> PromotionReport:
    """End-to-end: load -> extract -> WIS/DM -> verdict -> report."""

    if policy_family not in POLICY_FAMILIES:
        raise PromotionInputError(
            f"unknown policy_family={policy_family!r}; "
            f"supported: {POLICY_FAMILIES}"
        )
    if thresholds is None:
        thresholds = PromotionThresholds()
    artifact_path = Path(policy_artifact_path)
    if not artifact_path.exists():
        raise PromotionInputError(
            f"policy artifact not found at {artifact_path}"
        )
    # An online-mutated side-state is shadow-only and un-promotable: its coefficients no
    # longer match a frozen, reproducible training run. Freeze + re-evaluate a fresh artifact first.
    if is_online_mutated_artifact(json.loads(artifact_path.read_text(encoding="utf-8"))):
        raise PromotionInputError(
            f"policy artifact at {artifact_path} carries an online-mutated online_state stamp; "
            "online-mutated policies are shadow-only and cannot be promoted — freeze and "
            "re-evaluate a fresh artifact"
        )
    pack_paths = [Path(p) for p in pack_paths]
    if not pack_paths:
        raise PromotionInputError("at least one replay pack is required")
    for p in pack_paths:
        if not p.exists():
            raise PromotionInputError(f"replay pack not found: {p}")

    records: list[Mapping[str, Any]] = []
    for p in pack_paths:
        records.extend(load_jsonl(p))

    # Registry dispatch the family was validated against POLICY_FAMILIES above, so the lookup is
    # total. A new judgeable family is a registration in FAMILY_REGISTRY, not a branch here.
    spec = FAMILY_REGISTRY[policy_family]
    policy = spec.loader(artifact_path)
    extraction = spec.extractor(
        policy=policy, records=records, training_scope=training_scope
    )
    num_actions = spec.num_actions
    policy_id = policy.policy_id
    policy_version = policy.version

    wis = compute_wis_lift(
        extraction.triples,
        num_actions=num_actions,
        seed=seed,
    )
    dm = compute_dm(extraction.triples)
    # The INDEPENDENT data-fit DM (report-only honesty cross-check; never feeds the verdict).
    independent_dm = fit_tabular_dm(extraction.triples)
    # Count the DISTINCT actions the target policy proposes across the eval set; a single-action
    # (degenerate) policy is un-judgeable by WIS and must abstain, not pass.
    n_distinct_target_actions = len({t.target_action for t in extraction.triples})
    verdict, reasons = evaluate_verdict(
        wis=wis,
        thresholds=thresholds,
        n_distinct_target_actions=n_distinct_target_actions,
    )

    config_hash = compute_config_hash(
        {
            "policy_family": policy_family,
            "training_scope": training_scope,
            "thresholds": {
                "min_lift_over_baseline": thresholds.min_lift_over_baseline,
                "min_lower_bound_lift": thresholds.min_lower_bound_lift,
                "min_n_effective": thresholds.min_n_effective,
                "min_records_with_weight": thresholds.min_records_with_weight,
            },
            "epsilon": BEHAVIOUR_POLICY_EPSILON,
            "clip": IMPORTANCE_WEIGHT_CLIP,
            "bootstrap_n": BOOTSTRAP_N,
            "num_actions": num_actions,
        }
    )
    dataset_hash = hash_pack_paths(pack_paths, repo_root=repo_root)
    code_commit = resolve_code_commit(repo_root)
    if generated_at is None:
        generated_at_iso = "1970-01-01T00:00:00+00:00"
    else:
        generated_at_iso = generated_at.astimezone(timezone.utc).isoformat()

    def _dataset_path_str(p: Path) -> str:
        if repo_root is None:
            return str(p)
        try:
            return p.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            return str(p)

    return PromotionReport(
        schema_version=PROMOTION_REPORT_SCHEMA_VERSION,
        policy_family=policy_family,
        policy_id=policy_id,
        policy_version=policy_version,
        policy_artifact_sha256=hash_file(artifact_path),
        dataset_id=dataset_id,
        dataset_hash=dataset_hash,
        dataset_paths=tuple(_dataset_path_str(p) for p in pack_paths),
        training_scope=training_scope,
        seed=seed,
        config_hash=config_hash,
        code_commit=code_commit,
        thresholds=thresholds,
        wis=wis,
        dm=dm,
        independent_dm=independent_dm,
        extraction=extraction,
        verdict=verdict,
        reasons=reasons,
        generated_at_iso=generated_at_iso,
    )


# ---------------------------------------------------------------------------
# JSON serialization.
# ---------------------------------------------------------------------------


def _ci_to_dict(ci: ConfidenceInterval) -> dict[str, Any]:
    return {
        "method": ci.method,
        "lower": ci.lower,
        "upper": ci.upper,
        "n": ci.n,
    }


def build_promotion_report_artifact(report: PromotionReport) -> dict[str, Any]:
    """Serialize ``report`` to a deterministic JSON-ready dict."""

    return {
        "schema_version": report.schema_version,
        **build_promotion_honesty(
            dataset_id=report.dataset_id,
            dataset_hash=report.dataset_hash,
            verdict=report.verdict,
            reasons=report.reasons,
        ),
        "report_kind": "promotion",
        "policy_family": report.policy_family,
        "policy_id": report.policy_id,
        "policy_version": report.policy_version,
        "policy_artifact_sha256": report.policy_artifact_sha256,
        "dataset_id": report.dataset_id,
        "dataset_hash": report.dataset_hash,
        "dataset_paths": list(report.dataset_paths),
        "training_scope": report.training_scope,
        "seed": report.seed,
        "config_hash": report.config_hash,
        "code_commit": report.code_commit,
        "generated_at_iso": report.generated_at_iso,
        "thresholds": {
            "min_lift_over_baseline": report.thresholds.min_lift_over_baseline,
            "min_lower_bound_lift": report.thresholds.min_lower_bound_lift,
            "min_n_effective": report.thresholds.min_n_effective,
            "min_records_with_weight": report.thresholds.min_records_with_weight,
        },
        "estimators": {
            "wis": {
                "estimator": "WIS_self_normalized_epsilon_smoothed",
                "epsilon": BEHAVIOUR_POLICY_EPSILON,
                "importance_weight_clip": IMPORTANCE_WEIGHT_CLIP,
                "bootstrap_n": BOOTSTRAP_N,
                "target_value": report.wis.target_value,
                "baseline_value": report.wis.baseline_value,
                "lift": report.wis.lift,
                "target_ci_bootstrap": _ci_to_dict(report.wis.target_ci),
                "lift_ci_bootstrap": _ci_to_dict(report.wis.lift_ci),
                "num_records": report.wis.num_records,
                "num_records_with_weight": report.wis.num_records_with_weight,
                "num_clipped": report.wis.num_clipped,
                "effective_sample_size": report.wis.effective_sample_size,
                "weight_concentration_index": (
                    report.wis.weight_concentration_index
                ),
            },
            "direct_method": {
                "estimator": "DirectMethod_policy_expected_reward",
                "value": report.dm.value,
                "num_records": report.dm.num_records,
                # This is the policy's OWN expected-reward self-estimate (theta_hat = b/A), NOT fit
                # from logged outcomes it is CIRCULAR and informational only. Use fitted_q_dm below for
                # an independent, data-derived cross-check.
                "circular": True,
                "note": "policy self-estimate, not fit from logged data; informational, does not feed verdict",
            },
            "fitted_q_dm": {
                # INDEPENDENT data-fit Direct Method Q(s,a)=mean logged reward per (state,action)
                # cell, evaluated at the target action. Report-only; does NOT feed the verdict (WIS does).
                "estimator": "TabularFittedQ_mean_logged_reward",
                "value": report.independent_dm.value,
                "fraction_covered": report.independent_dm.fraction_covered,
                "num_state_action_cells": report.independent_dm.num_state_action_cells,
                "num_covered_records": report.independent_dm.num_covered_records,
                "num_records": report.independent_dm.num_records,
            },
            "fqe": {
                "algorithm": "placeholder_sequential_rl_not_implemented",
                "note": (
                    "This gate scopes offline bandit policies; FQE is a "
                    "placeholder for future sequential RL."
                ),
            },
        },
        "extraction": {
            "num_records_total": report.extraction.num_records_total,
            "num_records_in_scope": report.extraction.num_records_in_scope,
            "num_records_dropped": report.extraction.num_records_dropped,
            "num_triples": len(report.extraction.triples),
        },
        "verdict": report.verdict,
        "reasons": list(report.reasons),
    }


def write_promotion_report(report: PromotionReport, path: Path) -> str:
    """Write ``report`` to ``path`` and return its sha256 hex digest."""

    artifact = build_promotion_report_artifact(report)
    blob = json.dumps(artifact, sort_keys=True, indent=2) + "\n"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(blob, encoding="utf-8")
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    logger.info(
        "Wrote %s promotion report for %s to %s (%d bytes, sha256 %s)",
        report.verdict,
        report.policy_id,
        path,
        len(blob.encode("utf-8")),
        digest[:16],
    )
    return digest


def load_promotion_report(path: Path) -> dict[str, Any]:
    """Load a promotion-report JSON with strict schema check."""

    p = Path(path)
    if not p.exists():
        raise PromotionInputError(f"promotion report not found at {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise PromotionInputError(
            f"promotion report at {p} must be a JSON object"
        )
    schema = data.get("schema_version")
    if schema != PROMOTION_REPORT_SCHEMA_VERSION:
        raise PromotionInputError(
            f"promotion report schema_version drift at {p}: "
            f"expected {PROMOTION_REPORT_SCHEMA_VERSION}, got {schema!r}"
        )
    family = data.get("policy_family")
    if family not in POLICY_FAMILIES:
        raise PromotionInputError(
            f"promotion report at {p} has unknown policy_family={family!r}"
        )
    verdict = data.get("verdict")
    if verdict not in PROMOTION_VERDICTS:
        raise PromotionInputError(
            f"promotion report at {p} has invalid verdict={verdict!r}"
        )
    return data


__all__ = (
    "PROMOTION_REPORT_SCHEMA_VERSION",
    "PROMOTION_NUMERIC_TOLERANCE",
    "PROMOTION_VERDICT_PASS",
    "PROMOTION_VERDICT_FAIL",
    "PROMOTION_VERDICT_ABSTAIN",
    "PROMOTION_VERDICTS",
    "POLICY_FAMILY_SEQUENCING",
    "POLICY_FAMILY_PERCEPTION",
    "POLICY_FAMILY_RECOVERY",
    "POLICY_FAMILIES",
    "PromotionInputError",
    "EvaluationTriple",
    "TripleExtractionResult",
    "WISLiftEstimate",
    "DMEstimate",
    "IndependentDMEstimate",
    "FAMILY_REGISTRY",
    "PromotionThresholds",
    "PromotionReport",
    "extract_sequencing_triples",
    "extract_perception_triples",
    "extract_recovery_triples",
    "compute_wis_lift",
    "compute_dm",
    "fit_tabular_dm",
    "evaluate_verdict",
    "evaluate_policy_for_promotion",
    "compute_config_hash",
    "resolve_code_commit",
    "hash_file",
    "hash_pack_paths",
    "build_promotion_report_artifact",
    "write_promotion_report",
    "load_promotion_report",
)
