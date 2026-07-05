# Camera Package

Path: `backend/src/camera/`

The camera package owns camera setup, image acquisition, frame streaming, and camera-level orchestration. It sits between type-safe camera config and the calibration/stereo runtime.

It does not own calibration math, hand-eye solving, extrinsics persistence, generic SE(3) geometry, robot drivers, model inference, or application UI workflows.

## Structure

```text
backend/src/camera/
├── __init__.py
├── frame_provider.py
├── stereo_capture.py
└── setup/
    ├── quality.py
    ├── devices/
    │   ├── webcam.py
    │   └── stereocamera.py
    └── image_taking/
        ├── frames.py
        ├── webcam.py
        ├── single.py
        └── rgbd.py         # OpenCvRGBDStreamer + RealSenseRGBDStreamer
```

## Public Entry Points

```python
from backend.src.camera.orchestration import FrameProvider
from backend.src.camera.pipeline import StereoCapturePipeline
from backend.src.camera.setup.image_taking import RGBDFrame, StereoFrame

# Convenience package root imports are also supported.
from backend.src.camera import FrameProvider, StereoCapturePipeline
```

## Neighboring Packages

| Package | Responsibility boundary |
| --- | --- |
| `backend.config` | Validates `CameraSystemConfig`, rig configs, calibration config, and matcher config. |
| `backend.src.camera` | Opens devices, captures frames, resolves rig runtime setup. |
| `backend.src.calibration` | Owns stereo calibration, stereo reconstruction, hand-eye calibration, extrinsics, and quality metrics. |
| `backend.src.geometry` | Owns frame-safe pose, transform, quaternion, and matrix math. |
| `backend.src.utility` | Owns generic IO, paths, logging setup, timing, units, and small image helpers. |

## Frame Types

Frame dataclasses are defined once in `backend/src/camera/setup/image_taking/frames.py` and re-exported from `backend.src.camera.setup.image_taking` and `backend.src.camera`.

```python
from backend.src.camera.setup.image_taking import RGBDFrame, StereoFrame
```

`StereoFrame(left, right)` validates non-empty numeric left/right arrays and matching image height/width. `RGBDFrame(color, depth)` validates a non-empty numeric color image and numeric depth image. Empty depth is allowed only as the documented RGB-D fallback when an OpenCV backend exposes color but not depth.

Camera frames are image arrays, not robotics transforms. Robotics coordinates stay in `backend.src.geometry` and `backend.src.calibration`.

## RGB-D backends

An `rgbd` rig picks its driver via `RGBDDeviceRigConfig.rgbd_backend`, and `FrameProvider` builds the matching streamer (`setup/image_taking/rgbd.py`):

| `rgbd_backend` | Streamer | Depth source |
| --- | --- | --- |
| `opencv` (default) | `OpenCvRGBDStreamer` | Generic `cv2.VideoCapture` + OpenNI `CAP_OPENNI_*` channels; no vendor SDK. |
| `realsense` | `RealSenseRGBDStreamer` | Intel RealSense via `pyrealsense2`: hardware-aligned depth, device depth-scale → uint16 mm, emitter / laser-power / visual-preset, a post-processing filter chain (`realsense.post_processing`), and a colour-intrinsics accessor. |

The RealSense stack is an **optional layer** — `pyrealsense2` is deferred-imported, so the library runs without it; install `requirements/camera-realsense.txt` on a machine with a RealSense attached. The real device round-trip is validated on-box; the driver logic is unit-tested with an injected fake SDK (`tests/test_realsense_streamer.py`).

## FrameProvider

`FrameProvider` is a rig-ID keyed runtime facade over lower-level streamers.

```python
from backend.src.camera.orchestration import FrameProvider

with FrameProvider(rigs) as provider:
    raw = provider.grab("webcam_pair_1")
```

API:

| Method | Behavior |
| --- | --- |
| `open()` / `release()` | Open or release all registered streamers. Context-manager safe. |
| `grab(rig_id)` | Return raw `StereoFrame` or `RGBDFrame`. Requires `open()`. |
| `grab_rectified(rig_id)` | Return rectified `StereoFrame`. Requires a `StereoCam3D`. |
| `is_stereo(rig_id)` | True for `webcam_pair` and `single_device` rigs. |
| `is_rgbd(rig_id)` | True for RGB-D rigs. |
| `rig_ids` | Config-order rig IDs. |
| `get_rig_config(rig_id)` | Original rig config object. |
| `get_stereo_rig_index(rig_id)` | The `StereoCam3D` rig index, skipping RGB-D rigs. |

`FrameProvider` does not crop, split, resize, or configure camera quality. That logic is delegated to streamers under `camera.setup.image_taking`.

Unknown rig IDs raise `UnknownCameraRigError`. Grabbing before opening raises `FrameProviderStateError`. Calling `grab_rectified()` without `StereoCam3D` raises `RuntimeError` for stereo rigs.

## StereoCapturePipeline

`StereoCapturePipeline` resolves configured rigs, ensures stereo calibration image sets exist, builds `StereoCam3D` for stereo rigs, and returns a `FrameProvider`.

```python
from backend.src.camera.pipeline import StereoCapturePipeline

pipeline = StereoCapturePipeline(camera_config, calibration, stereo_matcher)
provider, stereo = pipeline.run()
provider, stereo = pipeline.run(rig_id="my_rig")
provider, stereo = pipeline.run(force_record=True, clear_existing=False)
```

Rig resolution:

| Inputs | Behavior |
| --- | --- |
| explicit `rig_id` | Use that enabled rig only. |
| `active_mode == "rig"` | Use `active_rig_id`. |
| `active_mode == "auto"` | Probe all enabled rigs and use available rigs. |

RGB-D rigs are never sent to stereo calibration or `StereoCam3D`. Stereo rig config order is preserved and matches `FrameProvider.get_stereo_rig_index()`.

## Calibration Paths

Each rig owns isolated capture/calibration artifacts through `calibration_paths.base_dir`:

```text
calibration/
├── webcam_pair_1/
│   ├── left/
│   ├── right/
│   └── stereoMap.xml
├── stereo_cam_1/
│   ├── left/
│   ├── right/
│   └── stereoMap.xml
└── rgbd_1/
    ├── color/
    ├── depth/
    └── intrinsics.json
```

## Runtime Workflow

```python
provider, stereo = StereoCapturePipeline(camera_config, calibration, matcher).run()

with provider:
    for rig_id in provider.rig_ids:
        if provider.is_stereo(rig_id):
            frame = provider.grab_rectified(rig_id)
            rig_index = provider.get_stereo_rig_index(rig_id)
            depth_mm = stereo.compute_depth_map(frame.left, frame.right, unit="mm", rig=rig_index)
        else:
            frame = provider.grab(rig_id)
            color, depth_mm = frame.color, frame.depth
```

## Migration Map

| Old path | Current path |
| --- | --- |
| `backend.src.pipelines.camera.orchestration.frame_provider` | `backend.src.camera.orchestration.frame_provider` |
| `backend.src.pipelines.camera.pipeline.stereo_capture` | `backend.src.camera.pipeline.stereo_capture` |
| `backend.src.camera.images.image_taking.*` | `backend.src.camera.setup.image_taking.*` |
| `backend.src.camera.images.devices.*` | `backend.src.camera.setup.devices.*` |
| `backend.src.camera.calibration.stereo.*` | `backend.src.calibration.stereo.*` |
| `backend.src.camera.calibration.eye_to_hand.*` | `backend.src.calibration.eye_hand.eye_to_hand.*` |

The root modules `backend.src.camera.frame_provider` and `backend.src.camera.stereo_capture` remain compatibility re-exports for now.

## Tests

Hardware-free tests live in `tests/test_camera_boundaries.py`. They cover public imports, canonical frame dataclasses, frame validation, rig ID registration, lifecycle errors, rectification requirements, stereo rig index mapping, explicit/active/auto rig resolution, RGB-D skipping, force recording, and per-rig calibration image cleanup.