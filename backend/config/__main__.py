"""``python -m backend.config`` — validate the configuration tree.

Exit codes:
* 0 — config loads and validates cleanly
* 1 — :class:`backend.config.ConfigError` raised (file, parse or schema error)
* 2 — bad CLI arguments

Examples::

    python -m backend.config                 # validate the default tree
    python -m backend.config --data ./data   # validate a custom tree
    python -m backend.config --print         # also print the parsed config
"""

from __future__ import annotations

import argparse
import sys

from .loader import ConfigError, load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.config",
        description="Validate (and optionally pretty-print) the YAML configuration tree.",
    )
    parser.add_argument(
        "--data",
        metavar="DIR",
        default=None,
        help="path to a custom data/ directory (default: backend/config/data)",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="dump the validated AppConfig as JSON",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.data)
    except ConfigError as exc:
        print(f"config error:\n{exc}", file=sys.stderr)
        return 1

    if args.print:
        print(cfg.model_dump_json(indent=2))
    else:
        print(f"OK — config under {args.data or '<default>'} validates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
