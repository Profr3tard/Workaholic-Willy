"""cuRobo motion planning (process-isolated GPU planner) for Workaholic-Willy.

cuRobo runs collision-aware global trajectory optimisation on the GPU. It cannot
share the Isaac process (its ``warp`` version clashes with Isaac's), so it runs as
a process-isolated stdio sidecar: :class:`CuroboPlanClient` (this env, stdlib-only,
no cuRobo/Isaac imports) spawns ``curobo_planner_server.py`` with the cuRobo env's
python and talks newline-delimited JSON over the pipe.

This is a **firm dependency for real collision-aware motion** used by BOTH the
Isaac sim driver (:mod:`src.robot.drivers.sim`) and the real-UR execution
path (:mod:`src.robot.drivers.ur.curobo_motion`). It nonetheless stays an
out-of-process service (never imported in-process; cannot be a pip line — py3.10 +
CUDA build, ``warp`` clash, no wheel). The env is located via ``WILLY_CUROBO_PYTHON``
(default ``ext_deps/curobo_env/python.exe``); the reference robot/gripper descriptor
lives under :mod:`.robot` (``WILLY_CUROBO_ROBOT``). See ``docs/curobo-setup.md``.

This package is also the **anchor** for the external engines the motion stack builds on:
:mod:`.environment` owns every env-var name, default path, and availability probe for BOTH
cuRobo and the Coal / python-fcl exact-mesh collision engine, and
``python -m src.robot.safety.planning --check`` reports what is wired on this box.
"""

from __future__ import annotations

from .curobo_client import CuroboPlanClient, CuroboUnavailableError, curobo_env_available
from .environment import (
    CollisionEngineStatus,
    CuroboStatus,
    PlanningEnvironment,
    probe_collision_engine,
    probe_curobo,
    probe_planning_environment,
)

__all__ = [
    "CuroboPlanClient",
    "CuroboUnavailableError",
    "curobo_env_available",
    "CollisionEngineStatus",
    "CuroboStatus",
    "PlanningEnvironment",
    "probe_collision_engine",
    "probe_curobo",
    "probe_planning_environment",
]
