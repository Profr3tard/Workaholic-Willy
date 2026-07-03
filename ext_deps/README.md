# `ext_deps/` — local install root for Workaholic-Willy's external dependencies

This folder is the **one canonical place** where Willy's heavy, machine-local dependencies get
installed. Nothing here is committed — the whole folder is gitignored except this README and the
`.gitignore` — but the **code defaults now look here**, so a fresh box has one obvious install target
instead of scattered `D:\…` paths.

**Isaac Sim is the deliberate exception** — it is a multi-GB standalone installer with its own bundled
Python and stays where NVIDIA puts it (`D:\isaacsim\…`). Everything *else* Willy bridges to lives here.

None of these are hard dependencies: without them Willy falls back (cuRobo → blind-IK, Coal → capsule
self-collision, SuctionNet → analytical scorer), and mock mode / CI never touch them. They are measured
enhancements for the on-box validation cell.

## Layout & what satisfies which default

| Subfolder | What to install | Wired via (default when unset) |
|---|---|---|
| `curobo_env/` | micromamba env, **Python 3.10** + cuRobo deps (torch cu128, cuda-toolkit 12.8) | `WILLY_CUROBO_PYTHON` → `ext_deps/curobo_env/python.exe` |
| `curobo/` | NVlabs/curobo source clone (`pip install -e .[cu12]`) | the cuRobo package is found via `curobo_env` (pip -e) |
| `coal_env/` | micromamba env with **Coal** (`conda install coal -c conda-forge`) | `WILLY_COAL_PREFIX` → `ext_deps/coal_env` |
| `suctionnet/` | SuctionNet-baseline repo + weights (optional, experimental) | `WILLY_SUCTIONNET_PREFIX` / `WILLY_SUCTIONNET_WEIGHTS` (set manually) |

Install elsewhere if you prefer — just set the matching `WILLY_*` env var and the default is ignored.

## Install (from the repo root)

> Full per-tool detail — versions, the ur5e config generator, troubleshooting — lives in
> [`docs/external-deps.md`](../docs/external-deps.md) (umbrella) and the linked
> [`docs/curobo-setup.md`](../docs/curobo-setup.md) · [`docs/coal-setup.md`](../docs/coal-setup.md) ·
> [`docs/suctionnet-setup.md`](../docs/suctionnet-setup.md).

```bash
# Coal — exact mesh self-collision backend
micromamba create -y -p ext_deps/coal_env coal -c conda-forge

# cuRobo — collision-aware motion planner (own py3.10 env; warp differs from Isaac's)
micromamba create -y -p ext_deps/curobo_env -c conda-forge -c nvidia python=3.10 cuda-toolkit=12.8 git-lfs
ext_deps/curobo_env/python.exe -m pip install "torch==2.7.0" --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/NVlabs/curobo ext_deps/curobo
ext_deps/curobo_env/python.exe -m pip install -e ext_deps/curobo/.[cu12] --no-build-isolation
# generate the ur5e cuRobo config (writes into cuRobo's content dir):
ext_deps/curobo_env/python.exe docs/curobo/build_ur5e_config.py

# SuctionNet — optional learned suction net (bucket-③; ~0.07 sim2real on Isaac RGB, so opt-in)
git clone https://github.com/graspnet/suctionnet-baseline ext_deps/suctionnet
export WILLY_SUCTIONNET_PREFIX="ext_deps/suctionnet/neural_network"
export WILLY_SUCTIONNET_WEIGHTS="ext_deps/suctionnet/weights/realsense-deeplabplus-RGBD"
```

That's it — the runtime picks up `ext_deps/curobo_env` and `ext_deps/coal_env` with **no env vars set**.
Verify the wiring with `python -m backend.config` (green) and, on-box, any `run_*` pick.
