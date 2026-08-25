"""RL CLI entry-point.

Subcommands:

* ``build-dataset`` assemble JSONL/optional-parquet splits from
  canonical packs (and optional extras) under
  ``logs/rl/datasets/<dataset_id>/`` and write the manifest to
  ``docs/baselines/rl_datasets/<dataset_id>.json``.
* ``audit-leakage`` re-run leakage audits on an existing manifest.
* ``replay-env-check`` emit deterministic SAR fingerprints for the
  recorded-observation and geometric-rerun environments.
"""

from __future__ import annotations

import argparse
from typing import Sequence



from ._commands_eval import register_eval_commands
from ._commands_train import register_train_commands
from ._commands_hardening import register_hardening_commands
from ._commands_dataset import register_dataset_commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.robot.grasping.rl",
        description="RL tooling (offline dataset builder + replay env).",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="repo root override (default: auto-detected)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    register_dataset_commands(sub)

    register_train_commands(sub)

    register_eval_commands(sub)

    register_hardening_commands(sub)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
