"""``python -m backend.config`` — validate the configuration tree, and ask it about itself.

Exit codes:
* 0 — the command succeeded (the config validates; a query ran)
* 1 — :class:`backend.config.ConfigError` raised (file, parse or schema error)
* 2 — bad CLI arguments

A query that finds nothing still exits 0: ``explain`` prints "NOT A KNOWN KEY" with suggestions and
``where`` prints "no config key matches". Asking about a key that turns out not to exist is a successful
answer to a legitimate question, not a failure of the tool.

Examples::

    python -m backend.config                             # validate the default tree
    python -m backend.config --data ./data                # validate a custom tree
    python -m backend.config --print                      # also print the parsed config

    # ...and the questions reading the YAML cannot answer:
    python -m backend.config explain robot.safety.self_collision.planner_margin_mm --profile sim,ur3e
    python -m backend.config where gripper

``explain`` reports a key's type, constraints, default, WHICH FILE AND LAYER set the winning value, the
whole override chain, and the comment its author wrote above that line — the measured WHY that
``yaml.safe_load`` throws away. ``where`` searches the SCHEMA rather than the files, so it finds the
fields no YAML mentions (measured: 107 of them, including ``robot.ur.model``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .loader import (
    ConfigError,
    active_profile,
    load_config,
    profile_layers,
    set_active_profile,
)


def _emit(text: str) -> None:
    """Print without ever dying on the console's encoding.

    The config's own comments contain box-drawing characters and typographic dashes, and a stock
    Windows console is cp1252. Printing an explanation there raised UnicodeEncodeError -- so the tool
    whose entire job is to explain a confusing config CRASHED on the confusing config. Unrepresentable
    characters are replaced; losing a dash beats losing the answer.
    """
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        print(text)
        return
    except UnicodeEncodeError:
        pass
    try:
        print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))
    except UnicodeEncodeError:  # a console that cannot even take its own replacement character
        print(text.encode("ascii", errors="replace").decode("ascii"))


def _shared_options() -> argparse.ArgumentParser:
    """The flags that work on BOTH sides of the subcommand -- a FRESH parser on every call.

    Two argparse facts collide here, and the collision is silent:

    1. ``default=SUPPRESS`` is what makes ``config --profile sim explain KEY`` work at all. With a normal
       default, the subparser's copy of the flag writes that default into the namespace *after* the
       top-level parser already stored the real value, so a flag given BEFORE the subcommand was
       discarded and the tool answered about the base tree.
    2. :meth:`ArgumentParser.set_defaults` **mutates the Action objects it matches**. ``parents=`` shares
       Action instances rather than copying them, so one shared parent parser plus ``set_defaults`` on
       the top level rewrote the SUBPARSER's ``SUPPRESS`` back to ``None`` -- reintroducing (1) through
       the very call meant to fix it. Measured: ``--profile sim explain KEY`` reported "no YAML sets
       this" for a key the sim layer plainly sets.

    Handing every parser its own instance keeps ``set_defaults`` local to the top-level parser.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--data", metavar="DIR", default=argparse.SUPPRESS,
        help="path to a custom data/ directory (default: backend/config/data)",
    )
    parser.add_argument(
        "--print", action="store_true", default=argparse.SUPPRESS,
        help="dump the validated AppConfig as JSON",
    )
    parser.add_argument(
        "--profile", metavar="CHAIN", default=argparse.SUPPRESS,
        help="profile layer chain to load, e.g. 'sim,ur3e' (default: whatever WILLY_PROFILE says)",
    )
    return parser


def _dotted_get(obj: object, path: str) -> object | None:
    """Follow a dotted path through the validated config; ``None`` if any segment is missing."""
    current: object | None = obj
    for part in path.split("."):
        if current is None:
            return None
        current = getattr(current, part, None) if not isinstance(current, dict) else current.get(part)
    return current


def main(argv: list[str] | None = None) -> int:
    # Shared options live on a PARENT parser so they work on both sides of the subcommand:
    # `config --profile X explain KEY` and `config explain KEY --profile X` both read naturally, and a
    # tool nobody can invoke correctly is not an ergonomics improvement.
    parser = argparse.ArgumentParser(
        prog="python -m backend.config", parents=[_shared_options()],
        description="Validate the YAML configuration tree, or ask it where a value came from.",
    )
    parser.set_defaults(data=None, profile=None, **{"print": False})
    sub = parser.add_subparsers(dest="command")
    explain_cmd = sub.add_parser(
        "explain", parents=[_shared_options()],
        help="what a key means, where it was set, and why that value",
    )
    explain_cmd.add_argument("key", help="dotted key, e.g. robot.sim.robot_model")
    where_cmd = sub.add_parser(
        "where", parents=[_shared_options()],
        help="find config keys by substring -- searches the SCHEMA, so it finds unwritten ones",
    )
    where_cmd.add_argument("needle", help="substring to look for, e.g. gripper")
    where_cmd.add_argument(
        "--tier", default=None,
        help="show only this tier (safety | site | tuned | advanced), or 'all'. A tier is a DISPLAY "
             "filter: every field stays settable whether it is shown or not.",
    )
    where_cmd.add_argument(
        "--limit", type=int, default=40, metavar="N",
        help="max keys listed PER TIER (default 40). `where grasping` matches 188 keys, so the default "
             "elides most of them; raise it when you are hunting rather than browsing.",
    )
    dec_cmd = sub.add_parser(
        "decisions", parents=[_shared_options()],
        help="only the values that DIFFER from their schema default -- what this cell actually decides",
    )
    dec_cmd.add_argument(
        "--section", default=None, metavar="PREFIX",
        help="limit to a dotted prefix, e.g. robot.safety",
    )
    dec_cmd.add_argument(
        "--tier", default=None,
        help="show only this tier (safety | site | tuned | advanced), or 'all'",
    )
    args = parser.parse_args(argv)

    if args.command == "where":  # schema-only: needs no tree and cannot fail on a broken one
        from .explain import find_keys

        # `--tier` and `--limit` were parsed and then dropped on the floor here, so the flag the help
        # text advertised did nothing and every listing silently stopped at 40 keys per tier. A filter
        # that is accepted and ignored is worse than no filter: it answers a question you did not ask.
        _emit(find_keys(args.needle, limit=args.limit, tier=args.tier))
        return 0

    previous = None
    if args.profile is not None:
        previous = active_profile()
        set_active_profile(args.profile)
    try:
        cfg = load_config(args.data)
    except ConfigError as exc:
        print(f"config error:\n{exc}", file=sys.stderr)
        return 1
    finally:
        if args.profile is not None:
            set_active_profile(previous)

    if args.command == "decisions":
        from .explain import decisions

        root = Path(args.data).resolve() if args.data else Path(__file__).resolve().parent / "data"
        layers = profile_layers(args.profile if args.profile is not None else active_profile())
        _emit(decisions(cfg, root, layers, section=args.section, tier=args.tier))
        return 0

    if args.command == "explain":
        from .explain import explain_key

        root = Path(args.data).resolve() if args.data else Path(__file__).resolve().parent / "data"
        layers = profile_layers(args.profile if args.profile is not None else active_profile())
        _emit(explain_key(args.key, root, layers, value=_dotted_get(cfg, args.key)))
        return 0

    if args.print:
        _emit(cfg.model_dump_json(indent=2))
    else:
        # Report the chain that was actually LOADED. Reading only `args.profile` printed
        # "(no profile)" while WILLY_PROFILE=sim was in force and its overlays were applied -- the
        # banner denied the very layering it had just performed. `explain` and `decisions` already
        # fell back to active_profile(); this is the same fallback.
        chain = " -> ".join(
            profile_layers(args.profile if args.profile is not None else active_profile())
        ) or "(no profile)"
        _emit(f"OK - config under {args.data or '<default>'} validates.  layers: {chain}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
