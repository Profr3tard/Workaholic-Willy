# cuRobo motion-planning setup (machine-local)

The `motion_planner="curobo"` path (sim driver) plans **global, collision-free** trajectories with NVIDIA
[cuRobo](https://curobo.org). cuRobo can **not** run inside the Isaac process: cuRobo needs `warp ~1.14` and
Isaac 5.1 ships `warp 1.8.2`, and one Python process holds only one `warp` (first on `sys.path` wins). So
cuRobo runs in its **own conda env** and the Isaac driver talks to it over stdio (see
[`curobo_client.py`](../backend/src/robot/drivers/sim/curobo_client.py) /
[`curobo_planner_server.py`](../backend/src/robot/drivers/sim/curobo_planner_server.py)). Process isolation
solves the warp clash **and** keeps the validated Isaac env untouched. Like the Coal env, this is
**machine-local — not committed**; reproduce it with the steps below.

## 1. The cuRobo env (contained, no system install except MSVC)

cuRobo's default backend is **`cuda.core` runtime JIT** (`CUROBO_USE_PYBIND=0`) — it compiles kernels at
runtime via NVRTC (a pip wheel), so the install COMPILES NOTHING and needs **no nvcc / no MSVC**. (The
deprecated pybind backend would; we don't use it.)

```bash
# micromamba already on this box (Coal): D:\coal\Library\bin\micromamba.exe, root D:\coal\mamba_root
export MAMBA_ROOT_PREFIX=/d/coal/mamba_root
MM=/d/coal/Library/bin/micromamba.exe

# py3.10 (cuRobo's supported version) + nvcc + git-lfs, into a contained env
"$MM" create -y -p ext_deps/curobo_env -c conda-forge -c nvidia python=3.10 cuda-toolkit=12.8 git-lfs

# torch matching Isaac's (Blackwell sm_120 / CUDA 12.8)
ext_deps/curobo_env/python.exe -m pip install "torch==2.7.0" --index-url https://download.pytorch.org/whl/cu128
```

## 2. Clone + install cuRobo (JIT backend, no compile)

```bash
git lfs install
git clone --depth 1 https://github.com/NVlabs/curobo.git ext_deps/curobo
cd /d/dev/curobo
# the [cu12] extra adds the cuda.core JIT runtime (cuda-core, nvrtc, cuda-runtime); no source compile
ext_deps/curobo_env/python.exe -m pip install -e ".[cu12]" --no-build-isolation
# smoke test (first run JIT-warms; UTF-8 so the cp1252 console can print the cuRobo "✓")
PYTHONIOENCODING=utf-8 ext_deps/curobo_env/python.exe -m curobo.examples.getting_started.motion_planning
```

> The contained env was bootstrapped after Tim installed **VS Build Tools** via
> `winget install --id Microsoft.VisualStudio.2022.BuildTools` — kept around in case the deprecated pybind
> build is ever needed, but the default JIT path above does not use it.

## 3. The ur5e robot config (assembled, not shipped)

cuRobo ships `ur10e`/`franka` but **no ur5e**. We assemble one from on-box ingredients (Isaac's own canonical
`ur5e.urdf` + Isaac's Lula ur5e collision spheres), writing `ur5e.urdf` + `ur5e.yml` into the cuRobo clone's
content. Run the reproducible builder:

```bash
ext_deps/curobo_env/python.exe docs/curobo/build_ur5e_config.py
```

The builder fixes an upstream `self_collision_ignore` typo (`forarm_link` → `forearm_link`) — harmless with
ur10e's sparse spheres, but with the dense Lula ur5e spheres it left `forearm↔wrist_1` self-colliding in
every config, so `plan_pose` returned `None` everywhere until renamed.

## 4. Wiring the driver to it (opt-in, default-off)

The driver path is reached only when `motion_planner="curobo"` (config field or owner-set
`arm._motion_planner`); the default `"ik"` is byte-identical. The client finds the env via env vars (defaults
below):

| env var | default |
|---|---|
| `WILLY_CUROBO_PYTHON` | `ext_deps\curobo_env\python.exe` |
| `WILLY_CUROBO_ROBOT` | `ur5e.yml` |
| `WILLY_CUROBO_CUBOID_CACHE` | `16` (boot world = a table + N−1 placeholder cuboids; `set_world` fills the slots with scene obstacles) |
| `WILLY_CUROBO_STDERR` | *(unset)* → server stderr discarded; set to a path to capture cuRobo diagnostics |

On the first `move()` the driver spawns the server (JIT-warm ~8–25 s), then each plan is ~tens of ms; the
driver EXECUTES cuRobo's full trajectory (planner-owns-final-motion, no blind IK-snap) and closes the server
on `disconnect()`.

**Validated on-box (2026-06-29):** a full UR5e clutter pick through cuRobo — every move planned + executed to
≤0.1 mm, the cube lifted 99.5 mm. Honesty: this is a **sim (bucket-①)** capability; it is not certified motion
safety.
