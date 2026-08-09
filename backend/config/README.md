# 🧩 Configuration

> Type-safe, immutable YAML configuration for Workaholic-Willy — validated by Pydantic v2, loaded once at startup.

<sub>[⬅ Workaholic-Willy](../../README.md) · `backend / config`</sub>

This is the bottom of the dependency stack: every other subsystem reads a frozen `AppConfig` and nobody
mutates it. The public entry point is `load_config()` → a validated, immutable `AppConfig`; schemas live
under `schema/`, the YAML you actually edit lives under `data/`, and `python -m backend.config` lints the
tree without booting the app. Typos in YAML are rejected at load time, not silently dropped.

## Contents

| Path | Role |
|------|------|
| [`__init__.py`](__init__.py) | Public API: `load_config`, `reload_config`, `ConfigError`, and the section models (`AppConfig`, `CameraConfig`, `ModelsConfig`, `RobotConfig`, `RuntimeConfig`). Schema classes import eagerly; the loader helpers load lazily so schema-only imports need no YAML dependency. |
| [`loader.py`](loader.py) | The pipeline: YAML read → `${VAR}` env substitution → `WILLY_PROFILE` overlay deep-merge → Pydantic validation. Result cached by `(data_dir, profile)`. |
| [`_merge.py`](_merge.py) | `_deep_merge` — the recursive dict merge used by profile overlays and the U11 adaptation overlay. |
| [`__main__.py`](__main__.py) | The `python -m backend.config` validator CLI. |
| [`editor.py`](editor.py) | Library-only runtime overlay editor (validate / save / switch profiles). **No web server** — see Notes. |
| [`schema/`](schema/) | Pydantic `StrictModel` schemas: [`app.py`](schema/app.py), [`runtime.py`](schema/runtime.py), [`camera/`](schema/camera/), [`models/`](schema/models/), [`robot/`](schema/robot/) (split per vendor + subsystem: `sim`, `ur`, `kuka`, `dummy`, `safety`, `grasping`, `calibration`, `kpi`, `rl`). |
| [`data/`](data/) | The YAML tree you edit: `camera/`, `models/`, `robot/`, `app/`, and [`grasping_presets/`](data/grasping_presets/). |

## Usage

```python
from backend.config import load_config

cfg = load_config()                            # validated, immutable AppConfig

cfg.camera.cameras.active_mode                 # "auto"
cfg.camera.stereomatcher.num_disparities       # 320
cfg.camera.hand_eye.eye_in_hand.mode           # "eye_in_hand"
cfg.models.stt.model_id                        # "openai/whisper-small"
cfg.runtime.event_hub.buffer_size              # 512
```

`cfg` is **frozen**: `cfg.camera.cameras.active_mode = "rig"` raises a `ValidationError`. Configs are
values, not state. Schema classes are also safe to import on their own, without the loader:

```python
from backend.config.schema.camera import CameraSystemConfig, WebcamPairRigConfig
```

Lint the tree from the shell:

```bash
python -m backend.config              # validate the shipped data/ tree      (exit 0 = OK)
python -m backend.config --print      # also dump the parsed AppConfig as JSON
python -m backend.config --data DIR   # validate a custom data/ directory
```

Exit codes: `0` validates · `1` `ConfigError` (file / parse / schema) · `2` bad CLI arguments.

## The config tree

`load_config()` merges the `data/` files into one `AppConfig`:

```
AppConfig
├── camera : CameraConfig
│   ├── cameras       (active_mode, rigs[], stereo_calibration — ChArUco/ArUco board)
│   ├── stereomatcher (SGBM + WLS + temporal; YAML keeps OpenCV camelCase)
│   └── hand_eye      (independent eye-to-hand / eye-in-hand workflows)
├── models : ModelsConfig
│   ├── handdetect, gesturedetect              (MediaPipe)
│   ├── objectdetector, segmenter, simplifier  (HF + optional `optim:` block)
│   └── stt                                     (Whisper)
├── robot  : RobotConfig | None   (absent ⇒ camera-only mode)
└── runtime: RuntimeConfig        (event_hub, decision_images, interaction, run_registry, image_encoding)
```

Every section inherits `StrictModel`, so `extra="forbid"` (unknown keys rejected) and `frozen=True`
(no post-construction mutation) hold everywhere.

Layout on disk (paths relative to `data/`):

```
camera/cam.yaml            (required)   camera/stereomatcher.yaml   (required)
camera/hand_eye.yaml       (optional)
models/*.yaml              (auto-discovered — every top-level key merged; duplicates rejected)
robot/robot.yaml           (optional)   robot/kpi_thresholds.yaml   (locked KPI gate)
app/runtime.yaml           (optional — schema defaults otherwise)
grasping_presets/{easy,dense_clutter,verification_heavy}.yaml
```

**What the shipped YAML contains — and what it deliberately does not.** A file states what *this cell
decided*; everything it stays silent about is the schema default, in force and unchanged. `robot.yaml` used
to restate 226 lines of `grasping:` / `rl:` defaults across 15 sub-blocks, of which **zero differed from
the schema** — measured by deleting each block and re-loading every shipped profile chain. They are gone;
the features are not. Nothing about the config's *capability* changed: `extra="forbid"` still accepts every
field, and adding a block back with `enabled: true` turns it on.

This is why the tree must be asked rather than read:

```bash
python -m backend.config decisions            # only what differs from the default — the actual decisions
python -m backend.config where fusion --tier all   # every knob, written or not
python -m backend.config explain robot.grasping.fusion.enabled
```

Two kinds of line stay written even when they equal their default, because a leaner file that hides them is
not a better file: **safety** bounds, and the **site facts** a bring-up must find (`ur.ip`,
`kuka.controller_ip`, `sim.robot_model`, `grasping.default_mode`, `rl.mode`). Rig catalogues
(`camera/cam.yaml`) also keep their full per-rig fields — ragged entries where one rig lists `fps` and the
next does not are harder to read, not easier. When you add a field, document it on the **schema** field
(`#:` or a plain `#` block above it — `explain` harvests both); only write it into a YAML if the cell is
actually choosing something.

**Validation you can rely on** — schema errors name the offending file (and, where Pydantic gives it, the
field path): `numDisparities` must be a positive multiple of 16, `blockSize` odd, `temporal_alpha ∈ [0,1]`;
duplicate `rig_id` rejected; `aruco_dict_name` checked against the OpenCV catalogue; duplicate model keys
across `models/*.yaml` rejected. Camera rigs are a discriminated union on `source`
(`webcam_pair` · `single_device` · `rgbd`). Config owns validated *data* only — it never opens a camera or
runs calibration (that is `backend.src.camera`).

## Profiles, overlays & env vars

Set `WILLY_PROFILE` to layer per-file overlays on top of the base YAML. For a base `foo.yaml`, the loader
deep-merges `foo.<profile>.yaml` from the same directory.

**Profiles compose.** `WILLY_PROFILE` takes a comma-separated **chain** of layers, applied left-to-right:

```bash
WILLY_PROFILE=sim                 # the Isaac cell (a UR5e)
WILLY_PROFILE=sim,ur3e            # ...the same cell driving a UR3e
WILLY_PROFILE=sim,ur3e,tiltcam    # ...and with the real rig's tilted D435 pair
```

Shipped layers: **`sim`** (the Isaac cell — overlays `robot.sim.yaml`, `camera/hand_eye.sim.yaml`,
`models/object.sim.yaml`, `models/segmenting.sim.yaml`), **`ur3e`** (reach-anchored geometry + kinematics
for the shorter arm), **`tiltcam`** (two tilted eye-to-hand D435s instead of one nadir camera), **`web`**.

Layers are independent *dimensions*, not alternative whole configurations — which is why they chain rather
than fork. The `sim` layer carries measured values (notably the detector `torch_dtype` reset that
small-object recall depends on); a profile per combination would have to copy those per robot, and the
copies would drift. Chained, each value is stated once. It also keeps an experiment interpretable: change
the robot layer alone, or the camera layer alone, and a measured difference is attributable to it.

> The old separate `data_sim/` tree is gone: **sim is now a profile of `data/`**, not a second tree.
> `backend.src.willy_sim` loads it via `load_sim_config(robot_model=..., extra_profiles=...)`, which builds
> the chain for you (every runner's `--robot-model` flag routes through it).

Merge rules (see `_merge._deep_merge`):

- dictionaries merge recursively, per key;
- scalars and lists **replace** the base value;
- a later layer wins over an earlier one; the base YAML is the earliest layer of all;
- a `null` overlay leaf **keeps** the base value (a partial overlay never accidentally wipes a field);
- the sentinel string `"__null__"` is the **only** way an overlay can *unset* a base to `None` — e.g.
  `models/object.sim.yaml` drops the production `optim.torch_dtype` so the sim detector runs its validated
  fp16-autocast-over-fp32-weights recall path.

If `WILLY_PROFILE` names a layer with no matching `*.<layer>.yaml` under `data/`, loading fails with
`ConfigError` — checked **per layer**, so a typo in the middle of a chain cannot merge as a silent no-op
and leave the cell running another robot's geometry.

Any string in any YAML may reference an environment variable, substituted **before** parsing (so the result
must stay valid YAML):

```yaml
robot:  { ur: { ip: ${ROBOT_IP:-192.168.1.100} } }   # default supplied
models: { stt: { model_path: ${MODEL_DIR}/whisper } } # required: ConfigError names MODEL_DIR if unset
```

## Robot & fail-closed safety

`robot/robot.yaml` is **vendor-block shaped**: the top-level `vendor:` key (`ur` · `kuka` · `sim` · `dummy`)
selects which sibling block the runtime consults. There is no flat `connection:` shim — it was removed. The
`sim` block carries `mock_mode` (a pure-Python kinematic mock, no Isaac) plus Isaac scene-authoring extras
read by `backend.src.willy_sim` (not the bare driver). KUKA is schema-validated but its EthernetKRL driver
is unvalidated on real hardware; controller-side KRL/EKI templates ship under
[`data/robot/templates/kuka/`](data/robot/templates/kuka/).

`robot.safety` is a **vendor-neutral** block applied *on top of* controller-side limits (UR safety planes,
KUKA SafeOperation, …) — the more restrictive value always wins, so it can only ever tighten the cell. It is
split into per-guard sub-blocks, each with its own `enforce: bool` (default `true`); `enforce: false` drops
that guard from the per-move pipeline at construction time:

| Guard block | Checks |
|-------------|--------|
| `limits` | workspace-face margins (per-move velocity/accel live in `motion_limits`) |
| `joint_limits` | per-axis angle margins |
| `ik_quality` | joint jump, singular value, condition number, limit proximity |
| `motion_continuity` | max joint / orientation / TCP step per move |
| `payload` | mass + CoG + inertia envelope |
| `self_collision` | capsule or `fcl` mesh backend; optional axis-aligned `fixtures` |
| `dwell` | post-stop dwell + steady-before-motion (consumed by the execution layer) |

The single entry point is `backend.src.robot.safety.SafetyPreflight`; each driver builds one via
`SafetyPreflight.from_safety_config(...)` and runs `evaluate()` before commanding motion. Rejections surface
as precise typed `MotionStatus` reasons (`WORKSPACE_REJECTED`, `JOINT_LIMIT_REJECTED`, `IK_QUALITY_REJECTED`,
`SELF_COLLISION_REJECTED`, `PAYLOAD_REJECTED`, `CONTINUITY_REJECTED`). No data → no motion, by design.

`robot.grasping` configures optional grasp behaviour (`default_mode`, `closed_loop`, `verification`,
`dense_recovery`, `gripper_geometry`) layered on top of the locked mode profiles — its values **cannot relax
safety** or unlock a recovery action the active mode forbids. Defaults reproduce the shipped open-loop jaw
pick byte-identically.

## Notes

- **Grasping presets bypass schema validation.** The three overlays under
  [`data/grasping_presets/`](data/grasping_presets/) (`easy` = min-risk single pick; `dense_clutter` = bin
  picking with bounded `next_viewpoint` recovery + uncertainty fail-closed; `verification_heavy` =
  closed-loop with mandatory post-grasp verification) are merged onto `robot.grasping` via
  `backend.src.robot.grasping.replay.presets.apply_preset(base, name)` — they are operator overlays, not
  validated `AppConfig` fields. Full operator reference: [`QUICKSTART.md`](../../QUICKSTART.md).
- **`kpi_thresholds.yaml` is the locked production KPI gate**, consumed by
  `python -m backend.src.robot.grasping.replay` (soak / KPI / baseline). The gate is a *synthetic* contract
  self-check — telemetry/KPI consistency, not real grasp quality.
- **`editor.py` is a library module — there is no web server, FastAPI app, or frontend in this repo.** Its
  functions (`profile_status`, `switch_profile`, `load_config`, `read/validate/save_overlay_bundle`,
  `apply_basic_settings`) are shaped so a *future* HTTP layer could expose them, but no routes exist today.
  All saves go through the same Pydantic schemas as `load_config()`; a save while a pipeline is running is
  rejected (`PIPELINE_ALREADY_RUNNING`); on validation failure the previous overlay is restored before the
  error surfaces. The basic-settings `robot_ip` key is vendor-aware — it lands on `robot.ur.ip` for UR,
  `robot.kuka.controller_ip` for KUKA, and is ignored otherwise.
- **Caching.** `load_config()` is cached by absolute `data_dir` + active profile; call `reload_config()` to
  invalidate after editing files (used by the test suite and the editor helpers). A custom data directory
  (`load_config("/path/to/data")`) must follow the same `camera/ models/ robot/ app/` layout.
- **Requirements.** Deps are split under [`requirements/`](../../requirements/); `pydantic` + `PyYAML` are
  load-bearing base deps. The optional UR stack (`ur_rtde`, `ur_ikfast`) is **not** in the default install.
- **Tests.** `tests/test_config_loader.py`, `tests/test_config_schema.py`, `tests/test_config_editor_cli.py`
  cover immutability, extra-rejection, env substitution, file-scoped errors, overlay merge semantics,
  duplicate detection, ArUco validation, independent hand-eye workflows, editor write guards, and the CLI.

## See also

- [⬅ Workaholic-Willy](../../README.md) — project root & architecture
- [`QUICKSTART.md`](../../QUICKSTART.md) — operator-facing config + preset reference
- [safety/](../src/robot/safety/README.md) — the `SafetyPreflight` the `robot.safety` block builds
- [grasping/](../src/robot/grasping/grasping_README.md) & [replay/](../src/robot/grasping/replay/README.md) — grasping presets + the KPI/soak gate that reads `kpi_thresholds.yaml`
- [examples/scripts/](../../examples/scripts/README.md) — `validate_config` / `inspect_profile` (diff a profile overlay)
- [docs/runbooks/](../../docs/runbooks/) — the operator procedures that *change* config in anger:
  [recalibration_eth_eih.md](../../docs/runbooks/recalibration_eth_eih.md) (camera extrinsics),
  [degraded_mode_ops.md](../../docs/runbooks/degraded_mode_ops.md) (running with a guard or sensor down),
  [rl_mode_switching.md](../../docs/runbooks/rl_mode_switching.md) and
  [rl_policy_rollback.md](../../docs/runbooks/rl_policy_rollback.md) (the `robot.rl` block)
