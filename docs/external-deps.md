# External dependencies — what we use, where, and why it lives outside the repo

> **Purpose (Tim, 2026-06-30).** Willy's sim + motion-planning stack relies on a few **heavy, machine-local**
> components that are deliberately NOT vendored into the repo (multi-GB CUDA/sim installs, GPL/large binaries,
> machine-specific paths). This doc is the single place that records **exactly what we use, where it lives, how the
> repo bridges to it, and what is machine-local vs. in-repo** — so a future rebuild, a new workstation, or a
> "pull it all into the repo" effort has the full picture in one glance. Per-tool setup lives in the linked
> `docs/*-setup.md`; this is the umbrella + the integration/rebuild checklist.

> **Install convention (2026-07-03).** Everything below **except Isaac Sim** installs into the in-repo
> **[`ext_deps/`](../ext_deps/)** folder (its contents are gitignored, so nothing heavy is committed). The
> code defaults resolve there with **no env vars set**; the `WILLY_*` overrides still point elsewhere if you
> install outside the repo. See [`ext_deps/README.md`](../ext_deps/README.md).

## TL;DR — the three external systems

| System | What it is | Installs at | Repo bridges via | Used by |
|---|---|---|---|---|
| **Isaac Sim 5.1** | The validation simulator (UR5e + 2F-85 cell) | `D:/isaacsim/isaac-sim-standalone-5.1.0-windows-x86_64` | its bundled `python.bat` (py3.11) | every `backend/src/willy_sim/run_*.py` |
| **cuRobo** | Collision-aware motion planner (the `motion_planner="curobo"` path, now the default) | env `ext_deps/curobo_env` (py3.10), source `ext_deps/curobo` | `WILLY_CUROBO_PYTHON` → a stdio sidecar | `curobo_client.py` ↔ `curobo_planner_server.py` |
| **Coal** | Exact mesh self-collision backend (`self_collision.backend="fcl"`) | env `ext_deps/coal_env` | `WILLY_COAL_PREFIX` | `safety/_fcl_self_collision.py` |

Everything else (perception models, torch, etc.) installs via the repo's own `requirements/` split into whatever
Python runs Willy; only the three above are out-of-repo heavy systems. **Detail per system below.**

---

## 1. Isaac Sim 5.1 — the simulator

- **What / where:** NVIDIA Isaac Sim 5.1 standalone at
  `D:/isaacsim/isaac-sim-standalone-5.1.0-windows-x86_64`. Drives the UR5e + 2F-85 cell, cameras, physics. Its
  **bundled `python.bat`** is a full interpreter (Python **3.11**, isaacsim, torch **cu128** for the Blackwell
  RTX 5080, **warp 1.8.2**). All on-box runs use it:
  `…/python.bat -m backend.src.willy_sim.run_<x> …`.
- **Repo bridge:** none beyond the import boundary — the sim driver lazy-imports `isaacsim.*` only after
  `mock_mode` is ruled out (so macOS/CI never touch it). The sim config layer (`backend/src/robot/drivers/sim/`)
  is Isaac-SDK-free until `connect()`.
- **Machine-local vs in-repo:** Isaac itself is machine-local (multi-GB). The cell config + scene authoring is
  in-repo (`backend/config/data_sim/`, `backend/src/willy_sim/`).
- **Known caveat:** Isaac's headless `SimulationApp.close()` can segfault on shutdown (`omni.graph` in
  `Py_FinalizeEx`) — a KNOWN Isaac teardown bug (see `drivers/sim/session.py:152`); it fires AFTER the run's
  result, harmless.

## 2. cuRobo — the collision-aware motion planner

> Full setup: **[curobo-setup.md](curobo-setup.md)**. Arc history + measure>map detail: memory
> `curobo-motion-planning`. Why it is now the default: the `★ MOTION-PLANNING` section of
> [.ai-memory/system-hardening-plan.md](../.ai-memory/system-hardening-plan.md).

- **What:** NVIDIA [cuRobo](https://curobo.org) plans global, collision-free joint trajectories. The sim driver's
  `motion_planner="curobo"` path (now the **default**, see `SimRobotConfig.motion_planner`) executes cuRobo's full
  trajectory (planner-owns-final-motion), gated by the Coal-mesh self-collision preflight.
- **Why out-of-process (NOT in the Isaac env):** cuRobo needs `warp ~1.14`; Isaac ships `warp 1.8.2`; one Python
  process holds only one `warp`. So cuRobo runs in its **own env** and the Isaac driver talks to it over stdio.
  - **Env (machine-local):** `ext_deps/curobo_env` — a `micromamba` env: **Python 3.10.20** (cuRobo's
    supported version), `cuda-toolkit=12.8` (nvcc), torch **2.7.x + cu128**, `nvidia-cuda-nvrtc-cu12` (the
    runtime JIT — cuRobo defaults to cuda.core NVRTC, so NO MSVC/nvcc source build is needed for the default path).
  - **Source (machine-local):** `ext_deps/curobo` — NVlabs/curobo clone, `pip install -e .[cu12] --no-build-isolation`.
  - **The ur5e robot config (machine-local, generated):** `ext_deps/curobo/curobo/content/configs/robot/ur5e.yml`
    (+ `assets/robot/ur_description/ur5e.urdf`). cuRobo ships no ur5e — we **generate** it from on-box ingredients
    (Isaac's ur5e URDF + Lula spheres + mesh-fit collision spheres) via the **in-repo** generator
    [`docs/curobo/build_ur5e_config.py`](curobo/build_ur5e_config.py). Re-run that (with the cuRobo env's python)
    to regenerate after a sphere-model change.
- **Repo bridge (env vars, all read by `curobo_client.py` / `curobo_planner_server.py`):**
  | Env var | Default | Meaning |
  |---|---|---|
  | `WILLY_CUROBO_PYTHON` | `ext_deps\curobo_env\python.exe` | the cuRobo env interpreter that runs the sidecar; its EXISTENCE is the build-time availability probe (`curobo_env_available()`) |
  | `WILLY_CUROBO_ROBOT` | `ur5e.yml` | the cuRobo robot config name (resolved in the cuRobo content dir) |
  | `WILLY_CUROBO_CUBOID_CACHE` | `16` | reserved collision-world cuboid slots (sized at boot for `set_world` obstacles) |
  | `WILLY_CUROBO_MAX_ATTEMPTS` | `16` | `plan_pose` attempts (reliability on tight queries) |
  | `WILLY_CUROBO_GRAPH_FROM_ATTEMPT` | `1` | first graph-seeded attempt (graph-from-0 was worse) |
  | `WILLY_CUROBO_STDERR` | unset | path to capture the sidecar's `[plan]` diagnostics (on-box debugging) |
- **In-repo code (the py3.11 side, stdlib-only, no cuRobo import):**
  `backend/src/robot/drivers/sim/curobo_client.py` (spawns + drives the warm sidecar) and
  `curobo_planner_server.py` (the py3.10 sidecar script — shipped in the repo but RUN by the external cuRobo
  python, never imported by Willy; its cuRobo imports are `type: ignore`-d for the project mypy).
- **Safety net:** if `WILLY_CUROBO_PYTHON` doesn't exist (no cuRobo env — CI, macOS, a fresh box), the driver
  AUTO-FALLS-BACK to the blind `"ik"` path (build-time + move-time), and `mock_mode` short-circuits before cuRobo
  is ever reached. So the repo stays green WITHOUT the external env; cuRobo is a measured *enhancement*, not a
  hard dependency.

## 3. Coal — the exact mesh self-collision backend

> Full setup: **[coal-setup.md](coal-setup.md)**. Arc: memory `h3-9g-coal-continuous-guard`, builds on
> `h3-9f-selfcoll-fidelity`.

- **What:** [Coal](https://github.com/coal-library/coal) (the maintained hpp-fcl successor) does vertex-exact
  mesh-vs-mesh self-collision for the `self_collision.backend="fcl"` path (the STANDARD sim check, and the gate
  on cuRobo's planned final config). ~1.5× faster than python-fcl, proven byte-identical at the distance level.
- **Env (machine-local):** `ext_deps/coal_env` — a `micromamba`/conda env with `coal` (`conda install coal -c
  conda-forge`). On Windows Coal has no pip wheel.
- **Repo bridge:** `WILLY_COAL_PREFIX` → the env prefix; `safety/_fcl_self_collision.py:_inject_coal_prefix()`
  adds `<prefix>/Library/bin` (DLLs) + `<prefix>/Lib/site-packages` so the HOST interpreter imports Coal.
- **Safety net:** if neither Coal (`WILLY_COAL_PREFIX` unset/missing) nor `python-fcl` is importable, the backend
  returns `None` and the guard falls back to the capsule path — no hard crash; CI/macOS unaffected (the mesh check
  is simply unavailable there).
- **The mesh authority (IN-REPO):** `backend/src/robot/safety/data/ur5e_collision_meshes.npz` — the DH-baked
  UR5e per-link collision meshes (vertex-exact <0.15 mm vs the Isaac USD). This is the **single source of truth**
  consumed by BOTH the Coal self-collision backend AND the cuRobo sphere-fit (`build_ur5e_config.py`) — so the
  planner's collision model and the safety gate derive from the same geometry.

---

## Repo-integration / fresh-machine rebuild checklist

To bring this stack up on a new box (or to plan pulling it into the repo), recreate:

1. **Isaac Sim 5.1** standalone → its `python.bat` is the on-box runner. (Machine-local; not vendorable.)
2. **Coal env** (`micromamba create -p <prefix> coal -c conda-forge`) → set `WILLY_COAL_PREFIX`. See
   [coal-setup.md](coal-setup.md). *(Optional — `python-fcl` or the capsule fallback keeps the repo runnable.)*
3. **cuRobo env + source** (`micromamba create -p ext_deps/curobo_env -c conda-forge -c nvidia python=3.10
   cuda-toolkit=12.8 git-lfs`; clone NVlabs/curobo to `ext_deps/curobo`; `pip install -e .[cu12]
   --no-build-isolation`) → set `WILLY_CUROBO_PYTHON` if the path differs. See [curobo-setup.md](curobo-setup.md).
   *(Optional — absent env → auto-fallback to "ik".)*
4. **Generate the ur5e cuRobo config** with the cuRobo env's python:
   `ext_deps/curobo_env/python.exe docs/curobo/build_ur5e_config.py` → writes `ur5e.yml`/`.urdf` into the cuRobo
   content dir. The arm-sphere surface augmentation needs `curobo.sphere_fit` + `trimesh` (present in the cuRobo
   env); absent → it degrades to Lula-only arm spheres (still a valid config).
5. **Paths to update if your layout differs** (all overridable, none hard-baked into the package): the three env
   vars above; the generator's `ISAAC_UR5E` / `CUROBO` constants in `docs/curobo/build_ur5e_config.py`.

**Honesty note:** everything here is **bucket-① sim** — software collision-aware planning + mesh self-collision in
simulation. It is NOT certified functional safety; the real-HW transfer (real UR base-yaw, real gripper mesh,
real planner timing) is the open H6 validation. The repo runs fully WITHOUT any of these three external systems
(mock mode + auto-fallbacks); they are measured enhancements for the on-box validation cell.
