"""Replay soak / records CLI handlers.

The records / records-gate / sim-soak / soak / soak-report mode handlers + their soak-only constants. main()
imports these for its dispatch (and _SIM_SOAK_REPORT_RELATIVE_PATH for help-text); argparse + dispatch stay in
__main__.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import yaml

from config.schema.robot.kpi_schema import KpiThresholdsConfig
from src.robot.grasping.constants import (
    REPLAY_SOAK_CLI_LOG_FILE,
    create_grasping_logger,
)
from src.robot.execution.autonomous_grasp import AutonomousGraspOutcome
from src.robot.grasping.telemetry.outcome_logging import iter_jsonl
from src.robot.grasping.replay.canonical_datasets import (
    repo_root_from_module,
)
from src.robot.grasping.replay.kpi import compute_kpis
from src.robot.grasping.replay.soak import (
    SoakScenarioSpec,
    DEFAULT_SOAK_REPORT_RELATIVE_PATH,
    SOAK_DEFAULT_ATTEMPTS,
    _aggregate_baseline_pick_success_rate,
    build_soak_report,
    evaluate_soak_gate_over_records,
    generate_soak_records,
)
from src.robot.grasping.replay.telemetry_catalog import (
    audit_records,
    audit_extra_records,
)

# Logging for this module.
logger = create_grasping_logger("SoakCLI", REPLAY_SOAK_CLI_LOG_FILE)

_VALID_OUTCOMES = frozenset(v.value for v in AutonomousGraspOutcome)
_SIM_SOAK_REPORT_RELATIVE_PATH = "logs/u12/sim_soak_report.json"


def _load_thresholds(path: Path) -> KpiThresholdsConfig:
    with path.open() as fh:
        return KpiThresholdsConfig.model_validate(yaml.safe_load(fh))


def _records_mode(records_path: Path) -> int:
    records = tuple(iter_jsonl(records_path))
    summary = compute_kpis(records)
    offenders = audit_records(records)
    extra_offenders = audit_extra_records(records)
    payload = {
        "mode": "records",
        "records_path": str(records_path),
        "kpi": summary.to_dict(),
        "telemetry_offenders": [
            {"attempt_id": aid, "missing": list(missing)}
            for aid, missing in offenders
        ],
        "extra_type_offenders": [
            {"attempt_id": aid, "bad_fields": list(bad)}
            for aid, bad in extra_offenders
        ],
    }
    log = logger.warning if (offenders or extra_offenders) else logger.info
    log(
        "KPI roll-up over %d record(s) from %s: %d telemetry offender(s), "
        "%d extra-type offender(s)",
        len(records),
        records_path,
        len(offenders),
        len(extra_offenders),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if offenders or extra_offenders:
        return 2
    return 0


def _records_gate_mode(records_path: Path) -> int:
    """Apply the record-intrinsic soak thresholds over a REAL GraspAttemptRecord log."""

    records = tuple(iter_jsonl(records_path))
    repo_root = repo_root_from_module()
    baseline_path = (
        repo_root / "docs" / "baselines" / "u_plus_baseline_v1.json"
    )
    baseline_pick: float | None = None
    if baseline_path.exists():
        try:
            baseline_payload = json.loads(
                baseline_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            baseline_payload = None
        if isinstance(baseline_payload, dict):
            baseline_pick = _aggregate_baseline_pick_success_rate(
                baseline_payload
            )
    gate_record, violations = evaluate_soak_gate_over_records(
        records, baseline_pick_rate=baseline_pick
    )
    gate: dict[str, object] = dict(gate_record)
    for pack_dependent_key in (
        "slo_packs_pass",
        "drift_gate_pass",
        "ood_gate_pass",
        "easy_attempt_wall_time_within_budget",
    ):
        gate[pack_dependent_key] = "not_applicable"
    gate["passes"] = not violations
    payload = {
        "mode": "records-gate",
        "records_path": str(records_path),
        "input_provenance": "real_grasp_attempt_record_log",
        "note": (
            "record-intrinsic soak thresholds over a REAL log: unlike --soak-report's synthetic "
            "self-check, this CAN fail. Pack-dependent keys (slo/drift/ood/wall-time) are "
            "not_applicable without the on-disk canonical packs."
        ),
        "baseline_pick_rate": baseline_pick,
        "gate": gate,
        "violations": list(violations),
    }
    log = logger.info if not violations else logger.warning
    log(
        "Records gate over %s: passes=%s%s",
        records_path,
        not violations,
        "" if not violations else f" violations: {'; '.join(violations)}",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not violations else 1


def _sim_soak_report_mode(
    records_path: Path, out_override: Path | None, *, min_attempts: int
) -> int:
    """Real-SIM-records soak quality gate + persistent report over a REAL Isaac-sim GraspAttemptRecord log.

    Honest by design: pick_success_rate is REPORTED but NOT gated (``baseline_pick_rate=None`` -> no
    cross-population comparison against the SYNTHETIC baseline, which is a different distribution); the
    record-intrinsic INTEGRITY keys (min_attempts/untyped/unbounded/telemetry/extra/dead_loop) gate.
    ``false_positive_grasp_rate`` is shown but structurally 0 (no secondary verifier in sim). Pack-dependent
    keys are not_applicable. Exit 0 iff no violations.
    """
    records = tuple(iter_jsonl(records_path))
    summary = compute_kpis(records)
    gate_record, violations = evaluate_soak_gate_over_records(
        records, baseline_pick_rate=None, min_attempts=min_attempts,
    )
    gate: dict[str, object] = dict(gate_record)
    for pack_dependent_key in (
        "slo_packs_pass",
        "drift_gate_pass",
        "ood_gate_pass",
        "easy_attempt_wall_time_within_budget",
    ):
        gate[pack_dependent_key] = "not_applicable"
    gate["passes"] = not violations
    payload = {
        "mode": "sim-soak-report",
        "report_version": 1,
        "records_path": str(records_path),
        "provenance": {
            "input": "real_sim_grasp_record_log",
            "measures": [
                "record_intrinsic_integrity_over_real_sim_picks",
                "sim_pick_success_rate_reported_not_gated",
            ],
            "does_not_measure": [
                "real_hardware_grasp_quality",
                "false_positive_grasp_rate (structurally 0 -- no secondary verifier in sim)",
            ],
            "note": (
                "REAL Isaac-physics sim picks (NOT hardware-representative). Gates the record-intrinsic "
                "INTEGRITY keys; pick_success_rate is REPORTED but NOT gated (no comparable sim baseline yet "
                "-> baseline_pick_rate=None; the synthetic baseline is a different distribution). "
                "Pack-dependent keys (slo/drift/ood/wall-time) need the on-disk canonical packs -> "
                f"not_applicable. sim min_attempts floor={min_attempts} (NOT the synthetic 2000). Run "
                "--soak-report for the contract self-check; this --sim-soak-report is the quality signal."
            ),
        },
        "total_attempts": len(records),
        "min_attempts_floor": int(min_attempts),
        "baseline_pick_rate": None,
        "kpi": summary.to_dict(),
        "gate": gate,
        "violations": list(violations),
    }
    repo_root = repo_root_from_module()
    out_path = (
        out_override if out_override is not None
        else repo_root / _SIM_SOAK_REPORT_RELATIVE_PATH
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_bytes = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    out_path.write_text(report_bytes)
    log = logger.info if not violations else logger.warning
    log(
        "Sim-soak report over %d record(s) from %s -> %s (%d bytes): passes=%s%s",
        len(records),
        records_path,
        out_path,
        len(report_bytes),
        not violations,
        "" if not violations else f" violations: {'; '.join(violations)}",
    )
    print(
        json.dumps(
            {
                "mode": "sim-soak-report",
                "report_path": str(out_path),
                "gate_passes": gate["passes"],
                "violations": list(violations),
                "total_attempts": len(records),
                "min_attempts_floor": int(min_attempts),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not violations else 1


def _soak_mode(thresholds_path: Path) -> int:
    thresholds = _load_thresholds(thresholds_path)
    total = thresholds.soak.min_attempts
    per = total // 3
    remainder = total - per * 3
    records = (
        *generate_soak_records(
            SoakScenarioSpec(
                name="easy",
                mode="easy",
                attempts=per,
                failure_class_weights={
                    "succeeded": 19.0,
                    "no_valid_grasp": 1.0,
                },
                recovery_success_rate=0.0,
                cycle_time_mean_s=1.5,
                cycle_time_jitter_s=0.2,
                seed=11,
            )
        ),
        *generate_soak_records(
            SoakScenarioSpec(
                name="auto",
                mode="auto",
                attempts=per,
                failure_class_weights={
                    "succeeded": 14.0,
                    "execution_failed": 2.0,
                    "no_valid_grasp": 2.0,
                    "decision_fail_closed": 1.0,
                },
                recovery_success_rate=0.5,
                cycle_time_mean_s=2.5,
                cycle_time_jitter_s=0.5,
                seed=22,
            )
        ),
        *generate_soak_records(
            SoakScenarioSpec(
                name="dense",
                mode="dense_clutter",
                attempts=per + remainder,
                failure_class_weights={
                    "succeeded": 990.0,
                    "execution_failed": 5.0,
                    "no_valid_grasp": 4.0,
                    "recovery_exhausted": 1.0,
                },
                recovery_success_rate=0.6,
                cycle_time_mean_s=3.5,
                cycle_time_jitter_s=0.7,
                seed=33,
            )
        ),
    )
    summary = compute_kpis(records)
    offenders = audit_records(records)
    violations: list[str] = []
    if summary.dead_loop_rate > thresholds.soak.dead_loop_rate_max:
        violations.append(
            f"dead_loop_rate {summary.dead_loop_rate:.4f} > "
            f"{thresholds.soak.dead_loop_rate_max:.4f}"
        )
    if offenders:
        violations.append(f"telemetry_offenders={len(offenders)}")
    untyped = [
        r.attempt_id for r in records if r.final_outcome not in _VALID_OUTCOMES
    ]
    if untyped:
        violations.append(f"untyped_outcomes={len(untyped)}")
    payload = {
        "mode": "soak",
        "attempts": len(records),
        "thresholds_path": str(thresholds_path),
        "kpi": summary.to_dict(),
        "violations": violations,
    }
    log = logger.info if not violations else logger.warning
    log(
        "Synthetic soak over %d generated attempt(s) (thresholds %s): passes=%s%s",
        len(records),
        thresholds_path,
        not violations,
        "" if not violations else f" violations: {'; '.join(violations)}",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not violations else 1


def _soak_report_mode(
    out_override: Path | None,
    *,
    total_attempts: int = SOAK_DEFAULT_ATTEMPTS,
) -> int:
    """Consolidated soak gate: build the soak report, write it to disk, exit non-zero on any locked violation."""

    repo_root = repo_root_from_module()
    payload, violations = build_soak_report(
        repo_root, total_attempts=total_attempts
    )
    if out_override is not None:
        out_path = out_override
    else:
        out_path = repo_root / DEFAULT_SOAK_REPORT_RELATIVE_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_bytes = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    out_path.write_text(report_bytes)
    log = logger.info if not violations else logger.warning
    log(
        "Soak report (%d synthetic attempt(s)) -> %s (%d bytes): passes=%s%s",
        int(cast("int", payload["total_attempts"])),
        out_path,
        len(report_bytes),
        not violations,
        "" if not violations else f" violations: {'; '.join(violations)}",
    )
    print(
        json.dumps(
            {
                "mode": "soak-report",
                "report_path": str(out_path),
                "gate_passes": cast("dict[str, Any]", payload["gate"])["passes"],
                "violations": list(violations),
                "total_attempts": payload["total_attempts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not violations else 1
