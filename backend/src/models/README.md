# Models Module

> Path: `backend/src/models/`
> Owns: every neural network the rest of WORKAHOLIC-WILLY talks to —
> object detection, instance segmentation, hand landmarks, speech-to-text,
> text simplification.
>
> **The grasp pipeline's default perception is the GroundingDINO detector + SAM2 segmenter.** Two
> alternates ship behind a config seam (`models.factory`): **RT-DETR** (closed-set, runs WITHOUT a
> prompt) and **OneFormer** (universal segmentation, higher-accuracy, GPU-heavy). Hand/gesture
> detection (`handdetection/`, MediaPipe) and speech-to-text (`speech/`) are **OPTIONAL, standalone
> surfaces NOT wired into grasping.** MediaPipe is an optional extra
> (`pip install -r requirements/voice.txt`): the modules import cleanly without it and raise a clear,
> actionable error only when a detector is actually constructed.

---

## TL;DR

```python
from backend.config.loader import load_config
from backend.src.models.factory import build_object_detector, build_segmenter

cfg = load_config()

# Config-driven backend selection (models.detector / models.segmenter_backend):
det = build_object_detector(cfg.models)   # GroundingDINO (prompt) or RT-DETR (no prompt)
seg = build_segmenter(cfg.models)         # SAM2 (realtime) or OneFormer (research)

det_result = det.detect(image_bgr, prompt="bottle")        # Detection (prompt optional for RT-DETR)
seg_result = seg.segment_detection(image_bgr, det_result)  # SegmentationResult
```

* Every model wrapper accepts a typed config from `backend/config/schema/models/`.
* All public results are **frozen dataclasses** — numpy fields are
  read-only, validated on construction.

---

## Numerics & frame contract

| Surface                         | Image format | Coord units | Frame  |
|---------------------------------|--------------|-------------|--------|
| Pipeline → model wrappers       | BGR (cv2)    | pixels      | image  |
| Model wrappers → HF / MediaPipe | RGB          | pixels      | image  |
| `Detection.box`                 | -            | pixels      | image  |
| `SegmentationResult.bbox_xyxy`  | -            | pixels      | image  |
| `HandPosition3D.position_*`     | -            | **mm**      | base / cam |
| `WhisperSpeechToText` audio     | float32      | -           | -      |

BGR↔RGB conversion happens inside each wrapper via
`backend.src.utility.vision.bgr_to_rgb` / `rgb_to_bgr` — callers never
swap channels themselves.

---

## Directory layout

```
models/
├── __init__.py
├── constants.py             # MODELS_LOG_DIR + per-model log files
├── _helpers.py              # Detection/SegmentationResult (shared result type)
├── _inference.py            # build_load_kwargs / finalize_model / autocast_ctx
├── factory.py               # build_object_detector / build_segmenter (config-driven selection)
├── detection/
│   ├── zeroshot/detector.py      # GroundingDinoObjectDetector (open-vocab, prompt)
│   └── closed_set/detector.py     # RtDetrObjectDetector (RT-DETR, fixed classes, NO prompt)
├── segmentation/
│   ├── realtime/segmenter.py      # Sam2Segmenter (box-prompted, fast)
│   └── research/segmenter.py      # OneFormerSegmenter (universal, higher-accuracy, GPU-heavy)
├── handdetection/
│   ├── __init__.py
│   ├── base_detector.py     # MediaPipe LIVE_STREAM (thread-safe)
│   └── palm_finder.py       # PalmDetector + HandFinder + dataclasses
├── simplifying/
│   └── simplifier.py        # TextSimplifier
├── speech/
│   └── speech_to_text.py    # WhisperSpeechToText
└── models_README.md         # this file
```

---

## Public API

```python
from backend.src.models.detection import Detection
from backend.src.models.detection.zeroshot.detector   import GroundingDinoObjectDetector
from backend.src.models.detection.closed_set.detector  import RtDetrObjectDetector
from backend.src.models.segmentation import SegmentationResult
from backend.src.models.segmentation.realtime.segmenter import Sam2Segmenter
from backend.src.models.segmentation.research.segmenter import OneFormerSegmenter
from backend.src.models.factory import build_object_detector, build_segmenter
from backend.src.models.handdetection import (
    PalmDetector, HandFinder, PalmDetection, HandPosition3D,
    calculate_palm_center, draw_hand_landmarks,
)
from backend.src.models.simplifying.simplifier   import TextSimplifier
from backend.src.models.speech.speech_to_text    import WhisperSpeechToText
```

### `Detection`

Frozen, slotted. Constructed by `GroundingDinoObjectDetector.detect()`.
`__post_init__` enforces:

* `box` length 4 with `x1 > x0` and `y1 > y0`, all finite.
* `score ∈ [0, 1]`.
* `label` non-empty string.
* `x_center`, `y_center` finite.

### `SegmentationResult`

Frozen + slotted (R9 — consistent with the sibling `Detection` / `PalmDetection` /
`HandPosition3D` result types). It crosses into `backend/src/robot/grasping/` as a public input via
the read-only `SegmentationLike` protocol; the dense-vision perception source rebuilds a relabelled /
mask-filled instance with `dataclasses.replace`, not in-place mutation. The `mask` is a `uint8` array
of shape `(H, W)` with values in `{0, 1}`.

### `PalmDetection`

Frozen, slotted. **Exactly 21 landmarks** (MediaPipe layout) — anything
else raises `ValueError`. `palm_center_xy` checked finite,
`hand_index >= 0`.

### `HandPosition3D`

Frozen, slotted. `position_base` and `position_cam` are coerced to
read-only `float64` arrays of shape `(3,)`. Mutating
`hp.position_base[0] = ...` raises at the numpy level.

---

## Concurrency: `BaseHandDetector`

MediaPipe's `LIVE_STREAM` running mode invokes `result_callback` from
its own worker threads while the pipeline thread reads
`hands` / `frame_landmarks` / `gesture`. **All shared state is guarded
by `self._lock` (`threading.Lock`)** and exposed via two safe
accessors:

```python
hands, frame_landmarks, gesture = detector.snapshot()           # sync
hands, frame_landmarks, gesture = await detector.asnapshot()    # async
```

Direct attribute reads still work for backward compatibility, but new
code should prefer `snapshot()` — it returns shallow copies, so
iteration is immune to in-flight callback writes.

---

## Logging

Every model wrapper logs through the shared rotating-file helper:

```python
from backend.src.utility.log_cfg import create_logger
from backend.src.models.constants import (
    MODELS_LOG_DIR,
    DETECTOR_LOG_FILE,
    SEGMENTER_LOG_FILE,
    HAND_FINDER_LOG_FILE,
    WHISPER_LOG_FILE,
    SIMPLIFIER_LOG_FILE,
)
```

Output paths (CWD-relative; resolved by `log_cfg`):

| Wrapper                  | Log file                              |
|--------------------------|---------------------------------------|
| `GroundingDinoObjectDetector` | `logs/backend/models/GroundingDINOObjectDetector.log` |
| `RtDetrObjectDetector`        | `logs/backend/models/rtdetr_detector.log` |
| `Sam2Segmenter`          | `logs/backend/models/sam2_segmenter.log`  |
| `OneFormerSegmenter`     | `logs/backend/models/oneformer_segmenter.log` |
| `HandFinder`             | `logs/backend/models/hand_finder.log`     |
| `WhisperSpeechToText`    | `logs/backend/models/whisper_model.log`   |
| `TextSimplifier`         | `logs/backend/models/simplifier_model.log`|

---

## Debug-image lifecycle

| Wrapper                       | Path (under `WILLY_DEBUG_DIR` or `<root>/logs/debug/`) | Cap |
|-------------------------------|---------------------------------------------------------|-----|
| `GroundingDinoObjectDetector` | `detection/detection_debug_<YYYYMMDD_HHMMSS>.png`       | 200 |
| `Sam2Segmenter`               | `segmentation/sam2_debug_<YYYYMMDD_HHMMSS_ffffff>.png`  | 200 |

Both routed through `backend.src.utility.paths.debug_dir(name, max_files=...)`,
which maintains a FIFO rotation. Debug writes are **opt-in** via the
`debug_images=True` / `save_debug=True` constructor kwargs and are off
by default.

---

## Inference helpers (`_inference.py`)

Three entry points used by every torch-based wrapper:

| Helper              | Purpose                                                       |
|---------------------|---------------------------------------------------------------|
| `build_load_kwargs` | Returns the `from_pretrained` kwargs (dtype, attn impl).      |
| `finalize_model`    | `.eval()`, optional channels-last, optional `torch.compile`. |
| `autocast_ctx`      | `torch.autocast` on CUDA, `nullcontext` on CPU.              |

`finalize_model` no longer swallows compile / channels-last failures —
it logs them at `WARNING` level and falls back to eager.

---

## Deployment checklist

1. **Models cached** — set `local=True` and `model_path=...` to avoid
   network calls at import time.
2. **GPU recommended** — `WhisperSpeechToText`, `Sam2Segmenter`, and
   `GroundingDinoObjectDetector` log a CPU warning if no CUDA device is
   available; everything still works on CPU but at single-digit FPS.
3. **MediaPipe assets** — `HandDetectConfig.model_path` and
   `GestureDetectConfig.model_path` must point at downloaded `.task`
   files. MediaPipe runs CPU-only.
4. **Debug images off** in production — set `debug_images=False` /
   `save_debug=False`. Even with the 200-file cap, write contention
   adds latency.
5. **Calibrated stereo / RGB-D** for `HandFinder` 3-D back-projection;
   the same `T_cam_to_base` used by grasping applies here.

---

## Wiring + lifecycle

There is **no web server / AppState in this repo** — each wrapper is constructed directly from its
config (model id/path + `InferenceOptimization`) where it is used. In the grasp pipeline that is the
Isaac sim cell: `run_m2_pick` / `run_dense_pick --vision` build `GroundingDinoObjectDetector` +
`Sam2Segmenter` once per run. `WhisperSpeechToText`, `TextSimplifier` and `HandFinder` / `PalmDetector`
are **standalone optional modules with no current production consumer** (the speech/handover pipeline
they were written for does not exist in this repo).

## Tests

These wrappers are torch/transformers/MediaPipe-heavy and are **omitted from CI coverage** (no GPU,
no weights on CI); the grasp-path wrappers are proven **on-box** instead.

* `tests/test_models_imports.py` + `tests/test_omitted_module_imports.py` — import-smoke for the
  optional wrappers (incl. the MediaPipe optional-dependency guard) so the package stays importable on
  macOS/CI without torch/MediaPipe installed.
* `tests/test_whisper_poll.py` — the torch-free `speech/_poll.py` helper.
* **On-box** (the only runtime exercise of `detector.py` + `segmenter.py` + `_inference.py`):
  `run_m2_pick --runs N` (GroundingDINO + SAM2 SOLO pick, baseline ~8/10) and
  `run_dense_pick --vision` (detect_all + SAM2 multi-object). `speech` / `simplifying` / `handdetection`
  have **no test and no on-box gate** — they are exercised only by import-smoke.

```powershell
& .\.venv\Scripts\Activate.ps1
python -m pytest -q tests/test_models_imports.py tests/test_omitted_module_imports.py tests/test_whisper_poll.py
```
