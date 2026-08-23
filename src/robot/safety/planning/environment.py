"""Single anchor point for the two external engines the motion stack builds on.

Workaholic-Willy's perceive -> grasp -> move stack now *depends on* two engines that cannot be plain
``pip`` dependencies. Rather than hide that behind optional-import guards scattered across the code,
this module makes the dependency explicit, named, and probeable in ONE place:

* **cuRobo** the GPU collision-aware trajectory planner, run as a process-isolated sidecar (see
  :mod:`.curobo_client`). Located on this box via the cuRobo env's Python interpreter.
* **Coal / python-fcl** the exact mesh-vs-mesh distance engine behind the fail-closed self-collision
  guard (see :mod:`.._fcl_self_collision`). Coal is preferred, python-fcl is the fallback.

Why they are not hard imports (and never can be):

* cuRobo needs Python 3.10 + a CUDA build + a ``warp`` version that clashes with the one Isaac ships,
  and has no wheel so it can only run out-of-process, in its own interpreter.
* Coal has no Windows wheel (install it from conda-forge and point :data:`ENV_COAL_PREFIX` at that env).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.robot.constants import PLANNING_ENVIRONMENT_LOG_FILE, create_robot_logger
from src.utility.paths import project_root

__all__ = [
    "ENV_CUROBO_PYTHON",
    "ENV_CUROBO_ROBOT",
    "ENV_CUROBO_CUBOID_CACHE",
    "ENV_CUROBO_STDERR",
    "ENV_CUROBO_MAX_ATTEMPTS",
    "ENV_CUROBO_GRAPH_FROM_ATTEMPT",
    "ENV_COAL_PREFIX",
    "COLLISION_MESH_DIR",
    "curobo_python_path",
    "curobo_robot_config",
    "curobo_cuboid_cache",
    "curobo_env_available",
    "collision_mesh_bundle",
    "inject_coal_prefix",
    "import_collision_engine",
    "CuroboStatus",
    "CollisionEngineStatus",
    "PlanningEnvironment",
    "probe_curobo",
    "probe_collision_engine",
    "probe_planning_environment",
]

# Logger for this module, and the only logger the --check CLI uses.
logger = create_robot_logger("PlanningEnvironment", PLANNING_ENVIRONMENT_LOG_FILE)

# --- cuRobo planner sidecar -----------------------------------------------------------------------
# The client reads the first four; the last two are read by the Python 3.10 sidecar
# (``curobo_planner_server.py``), which cannot import this module.
ENV_CUROBO_PYTHON = "WILLY_CUROBO_PYTHON"              #: interpreter of the cuRobo env
ENV_CUROBO_ROBOT = "WILLY_CUROBO_ROBOT"               #: cuRobo robot descriptor (default ``ur5e.yml``)
ENV_CUROBO_CUBOID_CACHE = "WILLY_CUROBO_CUBOID_CACHE"  #: reserved collision-world cuboid slots
ENV_CUROBO_STDERR = "WILLY_CUROBO_STDERR"             #: optional server-stderr log file
ENV_CUROBO_MAX_ATTEMPTS = "WILLY_CUROBO_MAX_ATTEMPTS"  #: sidecar: plan attempts (seed batches)
ENV_CUROBO_GRAPH_FROM_ATTEMPT = "WILLY_CUROBO_GRAPH_FROM_ATTEMPT"  #: sidecar: first graph-seeded attempt

_DEFAULT_CUROBO_ROBOT = "ur5e.yml"
_DEFAULT_CUROBO_CUBOID_CACHE = "16"


def curobo_python_path() -> str:
    """Path to the cuRobo env's Python (env override, else the in-repo ``ext_deps`` install)."""
    return os.environ.get(
        ENV_CUROBO_PYTHON, str(project_root() / "ext_deps" / "curobo_env" / "python.exe")
    )


def curobo_robot_config() -> str:
    """The cuRobo robot descriptor file name the sidecar loads."""
    return os.environ.get(ENV_CUROBO_ROBOT, _DEFAULT_CUROBO_ROBOT)


def curobo_cuboid_cache() -> str:
    """How many collision-world cuboid slots the sidecar reserves at boot."""
    return os.environ.get(ENV_CUROBO_CUBOID_CACHE, _DEFAULT_CUROBO_CUBOID_CACHE)


def curobo_env_available() -> bool:
    """True iff the cuRobo env's Python EXISTS on this box.

    A light, build-time check that does NOT spawn or JIT-warm the server: it lets a caller resolve a
    ``"curobo"`` request down to blind IK at build time (so the per-planner config matches the planner
    that will actually run). Deeper boot/JIT failures on a present env are caught later by the driver's
    move-time auto-fallback.
    """
    return Path(curobo_python_path()).exists()


# --- Coal / python-fcl exact-mesh collision engine ------------------------------------------------
ENV_COAL_PREFIX = "WILLY_COAL_PREFIX"  #: conda env prefix that provides Coal (Windows has no wheel)

#: Home of the committed, DH-baked, vertex-validated per-link collision meshes shared by the Coal
#: self-collision guard and the cuRobo sphere fit.
COLLISION_MESH_DIR = Path(__file__).resolve().parents[1] / "data"


def collision_mesh_bundle(model: str = "ur5e", variant: str | None = None) -> Path:
    """Path to the ``{variant or model}_collision_meshes.npz`` mesh bundle in :data:`COLLISION_MESH_DIR`."""
    return COLLISION_MESH_DIR / f"{(variant or model).lower()}_collision_meshes.npz"


def inject_coal_prefix() -> None:
    """If :data:`ENV_COAL_PREFIX` points to a conda env, make its Coal importable.

    On Windows the Coal extension's dependency DLLs live in ``<prefix>/Library/bin`` (not found via
    PATH for extension modules since 3.8) and the package in ``<prefix>/Lib/site-packages``. We APPEND
    site-packages so the host interpreter's own numpy still wins (Coal works against numpy 1.x and 2.x).
    Falls back to the in-repo ``ext_deps/coal_env`` when the variable is unset. No-op off Windows / when
    neither is present.
    """
    prefix = os.environ.get(ENV_COAL_PREFIX)
    if not prefix:
        default = project_root() / "ext_deps" / "coal_env"
        if not default.is_dir():
            return
        prefix = str(default)
    bindir = os.path.join(prefix, "Library", "bin")
    site = os.path.join(prefix, "Lib", "site-packages")
    if hasattr(os, "add_dll_directory") and os.path.isdir(bindir):
        try:
            os.add_dll_directory(bindir)
        except OSError:
            pass
    if os.path.isdir(site) and site not in sys.path:
        sys.path.append(site)
        # An sys.path mutation that decides whether the exact-mesh guard exists at all. Worth a line:
        # "Coal is missing" and "Coal is present but this prefix was never injected" look identical
        # from the guard's side and have completely different fixes.
        logger.info("injected the Coal prefix %s onto sys.path", prefix)


def import_collision_engine() -> tuple[Any, str] | tuple[None, None]:
    """Import the exact-mesh engine: ``(module, "coal")`` preferred, ``(module, "fcl")`` fallback."""
    for _attempt in (0, 1):
        try:
            import coal  # type: ignore[import-not-found]
            logger.debug("exact-mesh collision engine resolved: coal")
            return coal, "coal"
        except Exception:  # noqa: BLE001 - optional; inject the prefix once, then try python-fcl
            if _attempt == 0:
                inject_coal_prefix()
    try:
        import fcl  # type: ignore[import-not-found]
        logger.debug("exact-mesh collision engine resolved: python-fcl (Coal was not importable)")
        return fcl, "fcl"
    except Exception:  # noqa: BLE001 - neither engine present -> capsule fallback
        # DEBUG, not WARNING: ``_fcl_self_collision.make_backend`` already warns loudly about the
        # capsule fallback with its status token, and this helper is called several times per build.
        logger.debug("no exact-mesh collision engine importable (neither coal nor fcl)")
        return None, None


# --- Typed status snapshots (what the --check CLI reports) ----------------------------------------


@dataclass(frozen=True, slots=True)
class CuroboStatus:
    """Whether the cuRobo planner sidecar is wired on this box.

    **``available`` means the env PYTHON exists -- nothing more.** It does not mean the sidecar will
    plan. In particular it says nothing about ``robot_config``: the sidecar resolves that name against
    ``get_content_root()/configs/robot/`` INSIDE the cuRobo installation, in a separate environment this
    process cannot introspect without spawning it and this probe is deliberately spawn-free.
    """

    python_path: str
    available: bool
    robot_config: str

    @property
    def summary(self) -> str:
        state = "AVAILABLE" if self.available else "MISSING"
        return (
            f"cuRobo planner: {state}  (python={self.python_path}, robot={self.robot_config} "
            f"NOT verified: the descriptor lives in the sidecar env; confirm it was built for this "
            f"robot)"
        )


@dataclass(frozen=True, slots=True)
class CollisionEngineStatus:
    """Which exact-mesh engine is importable + whether the reference mesh bundle ships."""

    engine: str | None  # "coal" (preferred) | "fcl" (fallback) | None (capsule-only)
    mesh_bundle_present: bool
    coal_prefix: str | None
    model: str = "ur5e"  #: which robot's bundle was checked the bundles are PER-ROBOT

    @property
    def available(self) -> bool:
        return self.engine is not None and self.mesh_bundle_present

    @property
    def summary(self) -> str:
        engine = self.engine or "none (capsule fallback)"
        mesh = "present" if self.mesh_bundle_present else "MISSING"
        prefix = f", coal_prefix={self.coal_prefix}" if self.coal_prefix else ""
        return (
            f"exact-mesh collision engine: {engine}  ({self.model} mesh bundle {mesh}{prefix})"
        )


@dataclass(frozen=True, slots=True)
class PlanningEnvironment:
    """Snapshot of both external engines, the anchor status for the whole motion stack."""

    curobo: CuroboStatus
    collision: CollisionEngineStatus

    @property
    def fully_anchored(self) -> bool:
        """True iff BOTH the cuRobo env and an exact-mesh engine (+ bundle) are present."""
        return self.curobo.available and self.collision.available

    def report(self) -> str:
        verdict = "fully anchored" if self.fully_anchored else "partially anchored (degraded fallbacks active)"
        return (
            "Workaholic-Willy motion-stack external engines\n"
            f"  {self.curobo.summary}\n"
            f"  {self.collision.summary}\n"
            f"  => {verdict}"
        )


def probe_curobo(robot_config: str | None = None) -> CuroboStatus:
    """Light presence check of the cuRobo env (no spawn / no JIT)."""
    py = curobo_python_path()
    return CuroboStatus(
        python_path=py, available=Path(py).exists(),
        robot_config=robot_config or curobo_robot_config(),
    )


def probe_collision_engine(model: str = "ur5e") -> CollisionEngineStatus:
    """Import-probe the exact-mesh engine + check THIS model's mesh bundle ships."""
    _mod, kind = import_collision_engine()
    return CollisionEngineStatus(
        engine=kind,
        mesh_bundle_present=collision_mesh_bundle(model).exists(),
        coal_prefix=os.environ.get(ENV_COAL_PREFIX),
        model=model,
    )


def probe_planning_environment(
    *, robot_config: str | None = None, kinematics_model: str = "ur5e",
) -> PlanningEnvironment:
    """Probe both engines and return the combined anchor status FOR A SPECIFIC CELL."""
    return PlanningEnvironment(
        curobo=probe_curobo(robot_config), collision=probe_collision_engine(kinematics_model),
    )
