"""Verify the motion stack's external engines are wired on this box, FOR THIS CELL.

    python -m src.robot.safety.planning --check
    python -m src.robot.safety.planning --check --profile ur3e     # read the cell's own model
    python -m src.robot.safety.planning --check --model ur3e --json
    python -m src.robot.safety.planning --doctor                   # LOAD every engine (slower)

Prints the resolved status of both engines Workaholic-Willy's motion stack builds on the cuRobo
planner sidecar and the Coal / python-fcl exact-mesh collision engine and exits:

* ``0`` fully anchored: the cuRobo env Python is present AND an exact-mesh engine (+ the reference
  mesh bundle) is importable.
* ``1`` partially anchored: at least one engine is missing, so a degraded fallback is in force
  (blind IK instead of cuRobo, and/or the capsule proxy instead of exact meshes).

``--check`` reads paths; **``--doctor`` reads reality.** It imports Coal and runs a real distance
query, spawns the cuRobo sidecar's own interpreter to see what actually resolves there (including the
robot descriptor ``--check`` can only guess at, and whether a SECOND kernel backend exists), and
classifies an OS application-control refusal as its own outcome exit ``2``, separate from ``1``,
because "Windows is blocking a binary" and "this box has no GPU env" need opposite responses. It costs
a subprocess and a few seconds, so it stays opt-in rather than folded into ``--check``.
See :mod:`.doctor` and ``docs/code-integrity.md``.
"""

from __future__ import annotations

import argparse
import json
import sys

from .environment import probe_planning_environment


def _model_from_config(profile: str | None, data_dir: str | None) -> tuple[str, str]:
    """``(model, where_it_came_from)`` for the cell this box is configured to drive.

    Falls back to the ``ur5e`` default rather than failing.
    """
    from config.loader import (
        ConfigError,
        active_profile,
        load_config,
        reload_config,
        set_active_profile,
    )

    previous = active_profile()
    try:
        if profile is not None:
            set_active_profile(profile)
            reload_config()
        config = load_config(data_dir)
    except (ConfigError, OSError, ValueError) as exc:
        return "ur5e", f"default (the config did not load: {type(exc).__name__})"
    finally:
        if profile is not None:
            set_active_profile(previous)
            reload_config()

    robot = getattr(config, "robot", None)
    if robot is None:
        return "ur5e", "default (this config tree configures no robot)"
    declared = getattr(getattr(robot.safety, "self_collision", None), "kinematics_model", None)
    if declared:
        return str(declared), "robot.safety.self_collision.kinematics_model"
    vendor_model = getattr(getattr(robot, "ur", None), "model", None)
    if vendor_model:
        return str(vendor_model), "robot.ur.model"
    return "ur5e", "default (no model declared in this config)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.robot.safety.planning",
        description="Report whether the cuRobo planner + the exact-mesh collision engine are wired.",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="probe both engines, print the status report, exit 0 iff fully anchored (default action)",
    )
    parser.add_argument(
        "--doctor", action="store_true",
        help="deep check: LOAD every engine and name whatever refused. Exit 2 if an OS policy blocks.",
    )
    parser.add_argument(
        "--model", default=None,
        help="robot model to probe (e.g. ur3e). Default: read from the config this box would load.",
    )
    parser.add_argument("--profile", default=None, help="profile chain to read the model from")
    parser.add_argument("--data", default=None, help="config data directory")
    parser.add_argument(
        "--json", action="store_true",
        help="emit the reading as JSON instead of prose (for the operator console and for scripts)",
    )
    args = parser.parse_args(argv)

    if args.model:
        model, source = args.model, "--model"
    else:
        model, source = _model_from_config(args.profile, args.data)

    if args.doctor:
        from .doctor import run_doctor

        report = run_doctor(model=model, robot_config=f"{model}.yml")
        if args.json:
            print(json.dumps(
                {
                    "model": model,
                    "model_source": source,
                    "healthy": report.healthy,
                    "policy_blocked": report.blocked,
                    "probes": [
                        {"name": p.name, "status": str(p.status), "detail": p.detail, "remedy": p.remedy}
                        for p in report.probes
                    ],
                    "blocked_files": list(report.blocked_files),
                },
                indent=2,
            ))
        else:
            print(report.report())
            print(f"  (model: {model} -- from {source})")
        return report.exit_code

    env = probe_planning_environment(robot_config=f"{model}.yml", kinematics_model=model)

    if args.json:
        print(json.dumps(
            {
                "model": model,
                "model_source": source,
                "fully_anchored": env.fully_anchored,
                "curobo": {
                    "available": env.curobo.available,
                    "python_path": env.curobo.python_path,
                    "robot_config": env.curobo.robot_config,
                    "available_means": "the cuRobo environment's Python interpreter exists on disk; "
                                       "the robot descriptor inside it is NOT verified from here",
                },
                "collision": {
                    "available": env.collision.available,
                    "engine": env.collision.engine,
                    "mesh_bundle_present": env.collision.mesh_bundle_present,
                    "coal_prefix": env.collision.coal_prefix,
                },
            },
            indent=2,
        ))
    else:
        print(env.report())
        print(f"  (model: {model} -- from {source})")
    return 0 if env.fully_anchored else 1


if __name__ == "__main__":
    sys.exit(main())
