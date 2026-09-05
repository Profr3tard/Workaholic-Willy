"""``python -m src.config``: validate the configuration tree, and ask it about itself.

Exit codes:
* 0: the command succeeded (the config validates; a query ran)
* 1: :class:`src.config.ConfigError` raised (file, parse or schema error)
* 2: bad CLI arguments

A query that finds nothing still exits 0: ``explain`` prints "not a known key" with suggestions and
``where`` prints "no config key matches". Asking about a key that turns out not to exist is a successful
answer to a legitimate question, not a failure of the tool.

Examples::

    python -m src.config                             # validate the default tree
    python -m src.config --data ./data                # validate a custom tree
    python -m src.config --print                      # also print the parsed config

    # ...and the questions reading the YAML cannot answer:
    python -m src.config explain robot.safety.self_collision.planner_margin_mm --profile sim,ur3e
    python -m src.config where gripper

``explain`` reports a key's type, constraints, default, which file and layer set the winning value, the
whole override chain, and the comment its author wrote above that line, the measured why that
``yaml.safe_load`` throws away. ``where`` searches the schema rather than the files, so it finds the
fields no YAML mentions (measured: 107 of them, including ``robot.ur.model``).
"""

from __future__ import annotations

import argparse
import sys

from src.contracts import UNSET

from .tree import ConfigTree


def _emit(text: str) -> None:
    """Print without ever dying on the console's encoding.

    The config's own comments contain box-drawing characters and typographic dashes, and a stock
    Windows console is cp1252, where printing one raises UnicodeEncodeError. Without this, the
    tool whose entire job is to explain a confusing config would die on the confusing config.
    Unrepresentable characters are replaced; losing a dash beats losing the answer.
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
    """The flags that work on both sides of the subcommand, a fresh parser on every call.

    Two argparse facts collide here, and the collision is silent:

    1. ``default=SUPPRESS`` is what makes ``config --profile sim explain KEY`` work at all. With a normal
       default, the subparser's copy of the flag writes that default into the namespace *after* the
       top-level parser already stored the real value, so a flag given before the subcommand was
       discarded and the tool answered about the base tree.
    2. :meth:`ArgumentParser.set_defaults` **mutates the Action objects it matches**. ``parents=`` shares
       Action instances rather than copying them, so one shared parent parser plus ``set_defaults`` on
       the top level rewrote the subparser's ``SUPPRESS`` back to ``None``, reintroducing (1)
       through the very call meant to fix it. Measured: ``--profile sim explain KEY`` reported "no
       YAML sets this" for a key the sim layer plainly sets.

    Handing every parser its own instance keeps ``set_defaults`` local to the top-level parser.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--data", metavar="DIR", default=argparse.SUPPRESS,
        help="path to a custom data/ directory (default: config)",
    )
    # `--print` stays on this shared parent deliberately, even though it applies only to the
    # default command. Removing it would make `config --print explain KEY` fail with argparse's
    # generic "unrecognized arguments", which does not say why; keeping it lets `main` refuse with a
    # sentence naming the command that does accept it. Same reason the flag is on both sides of the
    # subcommand in the first place: the parse has to succeed before anything can explain itself.
    parser.add_argument(
        "--print", action="store_true", default=argparse.SUPPRESS,
        help="dump the validated AppConfig as JSON",
    )
    parser.add_argument(
        "--profile", metavar="CHAIN", default=argparse.SUPPRESS,
        help="profile layer chain to load, e.g. 'sim,ur3e' (default: whatever WILLY_PROFILE says)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    # Shared options live on a parent parser so they work on both sides of the subcommand:
    # `config --profile X explain KEY` and `config explain KEY --profile X` both read naturally, and a
    # tool nobody can invoke correctly is not an ergonomics improvement.
    parser = argparse.ArgumentParser(
        prog="python -m src.config", parents=[_shared_options()],
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
        help="find config keys by substring: searches the SCHEMA, so it finds unwritten ones",
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
        help="only the values that DIFFER from their schema default: what this cell actually decides",
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

    # **a flag that cannot apply is rejected, not ignored.** This file already states the rule,
    # at the `where` branch below: "a filter that is accepted and ignored is worse than no filter: it
    # answers a question you did not ask." It was stated about `--tier`/`--limit` and two more flags
    # were breaking it, both measured on 2026-09-04:
    #
    #   `explain KEY --print`  printed the explanation, no JSON, exit 0. `--print` rides the shared
    #                          parent parser onto all three subcommands and is only read after every
    #                          one of them has returned, so it is unreachable for each.
    #   `where N --data D`     byte-identical to `where N`, exit 0, and `where N --profile nosuch`
    #                          exits 0 where every other branch exits 1 on that chain. `find_keys`
    #                          takes no root at all: it searches the schema compiled into this
    #                          checkout, so pointing the tool at a customer's tree silently answered
    #                          about ours.
    #
    # Rejected rather than made to work, because neither can work. `--print` dumps the validated
    # tree, which is the default command's answer and not an explanation's; `where` has no tree to
    # point anywhere. Exit 2 is argparse's own "bad arguments", already the documented meaning at the
    # top of this file, and the message names the command that does accept the flag.
    if args.command is not None and args.print:
        parser.error(
            f"--print dumps the whole validated config, which is what the default command does; "
            f"'{args.command}' answers a narrower question. Run 'python -m src.config --print'."
        )
    if args.command == "where":
        for flag, value in (("--data", args.data), ("--profile", args.profile)):
            if value is not None:
                parser.error(
                    f"'where' searches the SCHEMA compiled into this checkout, not a YAML tree, so "
                    f"{flag} cannot change its answer. Drop {flag}, or use 'explain' / 'decisions', "
                    f"which do read the tree."
                )

    if args.command == "where":  # schema-only: needs no tree and cannot fail on a broken one
        from .explain import find_keys

        # `--tier` and `--limit` were parsed and then dropped on the floor here, so the flag the help
        # text advertised did nothing and every listing silently stopped at 40 keys per tier. A filter
        # that is accepted and ignored is worse than no filter: it answers a question you did not ask.
        _emit(find_keys(args.needle, limit=args.limit, tier=args.tier))
        return 0

    # One value carrying root, chain and layers, and this function used to carry three copies of
    # two of them: `root` and `layers` were rebuilt from the same expression at the `decisions` and
    # `explain` branches, independently, four lines apart. `ConfigTree` derives all three once and
    # the ask methods take only the key, so the value and its provenance cannot come from different
    # places. They already had: `explain_in` was measured printing `robot.sim.enabled = True` above
    # the robot.yaml line that sets it to `false`.
    #
    # And the profile is an argument now, not an environment variable set and put back. The old
    # dance defeated the `source` argument that `_validated_chain` carries for exactly one purpose:
    # naming the thing the operator typed. `--profile nosuch` blamed `WILLY_PROFILE` for a value
    # nobody had exported.
    loaded = ConfigTree.from_directory(
        root=args.data if args.data is not None else UNSET,
        profile=args.profile if args.profile is not None else UNSET,
    ).load()
    if not loaded.ok:
        print(loaded.render(), file=sys.stderr)
        return loaded.exit_code

    if args.command == "decisions":
        _emit(loaded.decisions(section=args.section, tier=args.tier))
        return 0

    if args.command == "explain":
        _emit(loaded.explain(args.key).render())
        return 0

    if args.print:
        _emit(loaded.config.model_dump_json(indent=2))
    else:
        _emit(loaded.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
