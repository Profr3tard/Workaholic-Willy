"""Eval CLI command group (ope / promote-policy)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.robot.grasping.constants import (
    RL_COMMANDS_EVAL_LOG_FILE,
    create_grasping_logger,
)

from .ope import (
    MalformedOPEInputError,
    build_ope_report,
    load_records_for_ope,
    sequencing_target_action_from_policy_path,
)
from .promotion import (
    POLICY_FAMILIES,
    POLICY_FAMILY_PERCEPTION,
    PromotionInputError,
    PromotionThresholds,
    build_promotion_report_artifact,
    evaluate_policy_for_promotion,
    write_promotion_report,
)


from ._cli_common import (
    COMMITTED_MANIFEST_DIR_REL,
    COMMITTED_RL_POLICY_DIR_REL,
    DEFAULT_OPE_REPLAY_PACKS,
    OPE_EXIT_MALFORMED_INPUT,
    SEQUENCING_POLICY_FILENAME,
    OPE_REPORT_FILENAME,
    PERCEPTION_PROMOTION_REPORT_FILENAME,
    RECOVERY_PROMOTION_REPORT_FILENAME,
    _relpath,
    _resolve_repo_root,
    _resolve_under_repo,
)


# Logging for this module.
logger = create_grasping_logger("RLEvalCLI", RL_COMMANDS_EVAL_LOG_FILE)


def _cmd_promote_policy(args: argparse.Namespace) -> int:
    """Promotion gate (offline, report-only)."""

    repo_root = _resolve_repo_root(args.repo_root)
    family = args.policy_family
    if family not in POLICY_FAMILIES:
        logger.error("promote-policy refused: unknown --policy-family %r", family)
        print(
            f"promote-policy: unknown --policy-family {family!r}; "
            f"supported: {POLICY_FAMILIES}",
            file=sys.stderr,
        )
        return 2

    artifact_arg = _resolve_under_repo(args.policy_artifact, repo_root)

    if args.replay_pack:
        pack_paths = [_resolve_under_repo(p, repo_root) for p in args.replay_pack]
    else:
        pack_paths = [repo_root / rel for rel in DEFAULT_OPE_REPLAY_PACKS]

    thresholds = PromotionThresholds(
        min_lift_over_baseline=float(args.min_lift_over_baseline),
        min_lower_bound_lift=float(args.min_lower_bound_lift),
        min_n_effective=float(args.min_n_effective),
        min_records_with_weight=int(args.min_records_with_weight),
    )

    logger.info(
        "promote-policy %s: artifact %s over %d replay pack(s), dataset %r, scope "
        "%s, seed %d, thresholds min_lift %.4f / min_ci_lower %.4f / min_n_eff "
        "%.1f / min_records_with_weight %d",
        family,
        artifact_arg,
        len(pack_paths),
        args.dataset_id,
        args.training_scope,
        int(args.seed),
        thresholds.min_lift_over_baseline,
        thresholds.min_lower_bound_lift,
        thresholds.min_n_effective,
        thresholds.min_records_with_weight,
    )
    try:
        report = evaluate_policy_for_promotion(
            policy_family=family,
            policy_artifact_path=artifact_arg,
            pack_paths=pack_paths,
            dataset_id=args.dataset_id,
            training_scope=args.training_scope,
            seed=int(args.seed),
            thresholds=thresholds,
            repo_root=repo_root,
        )
    except PromotionInputError as exc:
        logger.error("promote-policy refused: %s", exc)
        print(f"promote-policy: {exc}", file=sys.stderr)
        return 2

    if args.output:
        out_path = _resolve_under_repo(args.output, repo_root)
    else:
        default_name = (
            PERCEPTION_PROMOTION_REPORT_FILENAME
            if family == POLICY_FAMILY_PERCEPTION
            else RECOVERY_PROMOTION_REPORT_FILENAME
        )
        out_path = repo_root / COMMITTED_RL_POLICY_DIR_REL / default_name

    sha = write_promotion_report(report, out_path)

    artifact = build_promotion_report_artifact(report)
    summary = {
        "verdict": report.verdict,
        "reasons": list(report.reasons),
        "policy_id": report.policy_id,
        "policy_family": report.policy_family,
        "dataset_id": report.dataset_id,
        "num_triples": artifact["extraction"]["num_triples"],
        "wis_target_value": artifact["estimators"]["wis"]["target_value"],
        "wis_baseline_value": artifact["estimators"]["wis"]["baseline_value"],
        "wis_lift": artifact["estimators"]["wis"]["lift"],
        "wis_lift_ci_lower": (
            artifact["estimators"]["wis"]["lift_ci_bootstrap"]["lower"]
        ),
        "dm_value": artifact["estimators"]["direct_method"]["value"],
        "output_path": str(out_path),
        "output_sha256": sha,
    }
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


def _cmd_ope(args: argparse.Namespace) -> int:
    repo_root = _resolve_repo_root(args.repo_root)
    # override the default canonical pair; if neither override nor
    # defaults exist on disk we fall back to the sequencing training dataset
    # manifest splits.
    if args.replay_pack:
        pack_paths = [_resolve_under_repo(p, repo_root) for p in args.replay_pack]
    else:
        pack_paths = [repo_root / rel for rel in DEFAULT_OPE_REPLAY_PACKS]
        if not all(p.exists() for p in pack_paths):
            # Fallback: load the sequencing dataset splits via the committed manifest.
            manifest_path = (
                repo_root
                / COMMITTED_MANIFEST_DIR_REL
                / f"{args.fallback_dataset_id}.json"
            )
            if not manifest_path.exists():
                logger.error(
                    "OPE refused: canonical replay packs missing and no fallback "
                    "manifest at %s",
                    manifest_path,
                )
                print(
                    f"OPE: canonical replay packs missing and fallback "
                    f"manifest not found at {manifest_path}",
                    file=sys.stderr,
                )
                return OPE_EXIT_MALFORMED_INPUT
            logger.warning(
                "OPE falling back to the %r dataset manifest splits: the canonical "
                "replay packs are not all on disk",
                args.fallback_dataset_id,
            )
            manifest = json.loads(manifest_path.read_text())
            split_paths = manifest.get("splits") or {}
            pack_paths = [
                repo_root / rel for rel in split_paths.values() if rel
            ]

    try:
        records = load_records_for_ope(pack_paths)
    except MalformedOPEInputError as exc:
        logger.error("OPE refused: malformed input: %s", exc)
        print(f"OPE: malformed input: {exc}", file=sys.stderr)
        return OPE_EXIT_MALFORMED_INPUT

    sequencing_path = _resolve_under_repo(args.sequencing_artifact, repo_root)
    if not sequencing_path.exists():
        logger.error(
            "OPE refused: sequencing artifact not found at %s", sequencing_path
        )
        print(
            f"OPE: sequencing artifact not found at {sequencing_path}",
            file=sys.stderr,
        )
        return OPE_EXIT_MALFORMED_INPUT
    sequencing_target = sequencing_target_action_from_policy_path(sequencing_path)

    try:
        report = build_ope_report(
            records=records,
            sequencing_target_action_for_state=sequencing_target,
            dataset_id=args.dataset_id,
            dataset_paths=[_relpath(p, repo_root) for p in pack_paths],
            rng_seed=args.seed,
        )
    except MalformedOPEInputError as exc:
        logger.error("OPE refused: malformed input during report build: %s", exc)
        print(f"OPE: malformed input during report build: {exc}", file=sys.stderr)
        return OPE_EXIT_MALFORMED_INPUT

    out_path = (
        Path(args.output)
        if args.output
        else repo_root / COMMITTED_RL_POLICY_DIR_REL / OPE_REPORT_FILENAME
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(report, sort_keys=True, indent=2) + "\n"
    out_path.write_text(body)
    logger.info(
        "Wrote OPE report to %s (%d bytes)", out_path, len(body.encode("utf-8"))
    )

    summary = {
        "schema_version": report["schema_version"],
        "report_kind": report["report_kind"],
        "dataset_id": report["dataset_id"],
        "num_records": report["num_records"],
        "output_path": str(out_path),
        "wis_candidate": report["sections"]["candidate"]["estimators"]["wis"]["value"],
        "wis_ranking": report["sections"]["ranking"]["estimators"]["wis"]["value"],
        "wis_sequencing": report["sections"]["sequencing"]["estimators"]["wis"]["value"],
        "dm_sequencing": report["sections"]["sequencing"]["estimators"]["direct_method"]["value"],
    }
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


def register_eval_commands(
    sub: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    op = sub.add_parser(
        "ope",
        help=(
            "off-policy evaluation harness over the candidate/ranking/"
            "sequencing shadow telemetry"
        ),
    )
    op.add_argument(
        "--dataset-id",
        default="v5_ope_canonical",
        help="stable id surfaced in the report metadata",
    )
    op.add_argument("--seed", type=int, default=1, help="bootstrap RNG seed")
    op.add_argument(
        "--replay-pack",
        action="append",
        default=None,
        help=(
            "override replay-pack path(s) (repeatable). Default: "
            "tests/data/replay/replay_easy_canonical_v1.jsonl + "
            "tests/data/replay/replay_dense_canonical_v1.jsonl"
        ),
    )
    op.add_argument(
        "--sequencing-artifact",
        default=f"{COMMITTED_RL_POLICY_DIR_REL}/{SEQUENCING_POLICY_FILENAME}",
        help="path to the sequencing-policy artifact",
    )
    op.add_argument(
        "--fallback-dataset-id",
        default="v1_bootstrap",
        help=(
            "dataset id used to locate splits if canonical replay "
            "packs are missing (fallback)"
        ),
    )
    op.add_argument(
        "--output",
        default=None,
        help=(
            "output path for the OPE report JSON (default: "
            f"{COMMITTED_RL_POLICY_DIR_REL}/{OPE_REPORT_FILENAME})"
        ),
    )
    op.set_defaults(handler=_cmd_ope)

    pp = sub.add_parser(
        "promote-policy",
        help=(
            "offline promotion gate (WIS + DM) for a perception-budget "
            "or recovery policy artifact"
        ),
    )
    pp.add_argument(
        "--policy-family",
        required=True,
        choices=POLICY_FAMILIES,
        help="policy family (v5_perception_budget | v6_recovery)",
    )
    pp.add_argument(
        "--policy-artifact",
        required=True,
        help="path to the candidate policy artifact JSON",
    )
    pp.add_argument(
        "--replay-pack",
        action="append",
        default=None,
        help=(
            "replay-pack path (repeatable). Default: canonical easy + "
            "dense packs"
        ),
    )
    pp.add_argument(
        "--dataset-id",
        default="v7_promotion_canonical",
        help="stable dataset id surfaced in the report metadata",
    )
    pp.add_argument(
        "--training-scope",
        default="dense",
        help="training scope filter (default: dense)",
    )
    pp.add_argument("--seed", type=int, default=1, help="bootstrap RNG seed")
    pp.add_argument(
        "--min-lift-over-baseline",
        type=float,
        default=0.01,
        help="minimum point lift over baseline for verdict=pass (positive floor; default 0.01)",
    )
    pp.add_argument(
        "--min-lower-bound-lift",
        type=float,
        default=0.0,
        help=(
            "minimum bootstrap-CI lower bound on the lift for "
            "verdict=pass"
        ),
    )
    pp.add_argument(
        "--min-n-effective",
        type=float,
        default=10.0,
        help="minimum WIS effective sample size required to grade",
    )
    pp.add_argument(
        "--min-records-with-weight",
        type=int,
        default=5,
        help=(
            "minimum number of records where target policy agrees "
            "with behavior (required to grade)"
        ),
    )
    pp.add_argument(
        "--output",
        default=None,
        help=(
            "output path for the promotion-report JSON (default: "
            f"{COMMITTED_RL_POLICY_DIR_REL}/<family>_promotion_v1.json)"
        ),
    )
    pp.set_defaults(handler=_cmd_promote_policy)
