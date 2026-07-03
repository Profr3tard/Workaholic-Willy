# Coal setup (the exact-mesh self-collision engine) — reinstall guide

The exact mesh-vs-mesh self-collision backend (`self_collision.backend='fcl'`, H3.9f) and the continuous
collision-avoidance guard (H3.9g B2) run on **Coal** — the maintained successor to `hpp-fcl`. This is the
engine that powers the closest-distance queries; `python-fcl` is the guarded fallback.

> ⚠ **Honesty (safety-critical).** This is a **software collision-AVOIDANCE layer**, not "the safety
> system". It reduces collision risk in **simulation (bucket-①)**. The real-hardware safety **guarantee**
> is independent **certified functional safety** — hardware E-stop circuits + ISO 10218 / ISO/TS 15066 /
> ISO 13849 — running independently of this software (**bucket-③**). A green sim result is **not**
> "commercially safe". Coal makes the *check* better; it does **not** certify the robot.

## Why Coal (and not pip `hpp-fcl`)

`hpp-fcl` on PyPI is effectively dead (no release in >12 months). Coal (`coal` on conda-forge, v3.x) is
the maintained project: faster GJK, a usable distance lower-bound. On this codebase the swap was proven
**behavior-identical** to `python-fcl` at the distance level (natural 19.640 mm, flipped 0.0 mm — the
values the guard triggers on); Coal is ~1.5× faster (future RT headroom). See `requirements/safety-fcl.txt`.

## Windows reinstall (no system conda) — what actually works

Coal has **no Windows pip wheel** (the cmeel/Gepetto stack is Linux/macOS-first). On Windows install it
from **conda-forge** via a self-contained **micromamba** (no system install, no PATH/registry changes),
then bridge it into the project `.venv` and the Isaac standalone python with `WILLY_COAL_PREFIX`.

```powershell
# 1) micromamba (single binary) into D:\coal
New-Item -ItemType Directory -Force D:\coal | Out-Null
Invoke-WebRequest -Uri "https://micro.mamba.pm/api/micromamba/win-64/latest" -OutFile D:\coal\micromamba.tar.bz2
tar -xf D:\coal\micromamba.tar.bz2 -C D:\coal          # -> D:\coal\Library\bin\micromamba.exe

# 2) a coal env (conda-forge, python 3.11 to match the .venv / Isaac ABI = cp311)
$env:MAMBA_ROOT_PREFIX = "D:\coal\mamba_root"
# corporate network? schannel revocation checks often fail -> allow it:
Set-Content D:\coal\mamba_root\.mambarc "ssl_no_revoke: true`nssl_verify: true" -Encoding utf8
& D:\coal\Library\bin\micromamba.exe create -y -p ext_deps\coal_env -c conda-forge --ssl-no-revoke python=3.11 coal
```

```bash
# 3) point the backend at that env (the module injects its DLL dir + site-packages, and uses the HOST
#    interpreter's own numpy — verified against both numpy 2.x (.venv) and numpy 1.26 (Isaac)).
export WILLY_COAL_PREFIX="ext_deps/coal_env"     # PowerShell: $env:WILLY_COAL_PREFIX="ext_deps\coal_env"
```

### Verify

```bash
# import + a distance query (in any python 3.11):
WILLY_COAL_PREFIX="ext_deps/coal_env" .venv/Scripts/python.exe -c \
  "import os,sys; p=os.environ['WILLY_COAL_PREFIX']; os.add_dll_directory(p+'/Library/bin'); \
   sys.path.append(p+'/Lib/site-packages'); import coal; print('coal', coal.__version__)"

# the safety suite picks Coal up automatically when WILLY_COAL_PREFIX is set (reject detail['backend']=='coal'):
WILLY_COAL_PREFIX="ext_deps/coal_env" .venv/Scripts/python.exe -m pytest tests/test_safety_self_collision.py -q
```

## Linux / macOS

`conda install coal -c conda-forge` (or in a fresh micromamba env) — `import coal` then works directly, so
`WILLY_COAL_PREFIX` is usually unnecessary. The cmeel pip wheels also exist there if you prefer pip.

## Fallback

If Coal is absent, the backend imports `python-fcl` instead (pip-installable, ships Windows wheels:
`pip install -r requirements/safety-fcl.txt`). If neither (nor the mesh bundle) is present, the guard
returns no mesh backend and the one-shot self-collision guard falls back to its capsule path — no crash.
The default `self_collision.backend='capsule'` needs **neither** dependency.

## Notes

- The env (`ext_deps/coal_env`) is **machine-local — gitignored, not committed**; `WILLY_COAL_PREFIX` overrides its location. (Micromamba itself installs at `D:\coal` — that is the package-manager tool, not the env.)
- Match the env's python to the consumers (cp311 here: `.venv` 3.11.9, Isaac 3.11.13).
- The continuous guard (H3.9g B2) is **opt-in** (`continuous_guard=True`, default off = byte-identical) and
  needs `WILLY_COAL_PREFIX` on this box; without it the guard logs a WARN and stays OFF (it does not run
  unguarded silently).
