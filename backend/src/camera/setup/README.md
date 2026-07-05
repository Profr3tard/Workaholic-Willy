# Camera Setup and Image Taking

Path: `backend/src/camera/setup/`

This package owns lower-level camera setup and acquisition. It is used by `FrameProvider` and `StereoCapturePipeline`, but it does not own stereo calibration math, hand-eye calibration, geometry transforms, model inference, or robot APIs.

## Structure

```text
camera/setup/
├── quality.py
├── devices/
│   ├── webcam.py          # interactive calibration image capture for webcam pairs
│   └── stereocamera.py    # interactive calibration image capture for single stereo devices
└── image_taking/
    ├── frames.py          # StereoFrame and RGBDFrame
    ├── webcam.py          # WebcamPairStreamer
    ├── single.py          # SingleDeviceStreamer
    └── rgbd.py            # OpenCvRGBDStreamer, RealSenseRGBDStreamer, RGBDStreamerProtocol
```

## Quality Settings

`configure_camera_for_quality()` in `quality.py` is the single source of truth for camera quality setup. Calibration capture classes and runtime streamers both call it.

It configures pixel format, resolution, FPS, optional manual exposure/gain/white balance, auto-feature disabling, buffer size, and warmup frames. Not every backend supports every property, so it returns the values read back from OpenCV for validation and logging.

## Canonical Frame Dataclasses

```python
from backend.src.camera.setup.image_taking import RGBDFrame, StereoFrame
```

`StereoFrame` contains `left` and `right` image arrays. Both must be non-empty numeric arrays with matching height/width.

`RGBDFrame` contains `color` and `depth`. `color` must be non-empty. `depth` may be an empty array only when the backend cannot expose a depth stream; otherwise it must match the color height/width.

These dataclasses are the only frame dataclass definitions in the camera package.

## Calibration Capture Classes

Calibration capture classes live in `camera/setup/devices/` and are interactive OpenCV sessions.

| Class | Config | Purpose |
| --- | --- | --- |
| `StereoVisionCalibrationWebcams` | `WebcamPairRigConfig` | Capture synced left/right PNG pairs from two USB cameras. |
| `StereoVisionCalibrationSingleDevice` | `SingleDeviceRigConfig` | Capture and split PNG pairs from one side-by-side/top-bottom stereo device. |

Both save raw images without preview overlays, enforce `min_pairs` and `max_pairs`, and use `configure_camera_for_quality()`.

## Runtime Streamers

Runtime streamers live in `camera/setup/image_taking/` and return frames programmatically without GUI ownership.

```python
from backend.src.camera.setup.image_taking import WebcamPairStreamer

with WebcamPairStreamer(config) as streamer:
    frame = streamer.grab()
```

| Streamer | Config | Return type |
| --- | --- | --- |
| `WebcamPairStreamer` | `WebcamPairRigConfig` | `StereoFrame` |
| `SingleDeviceStreamer` | `SingleDeviceRigConfig` | `StereoFrame` |
| `OpenCvRGBDStreamer` | `RGBDDeviceRigConfig` (`rgbd_backend: opencv`) | `RGBDFrame` |
| `RealSenseRGBDStreamer` | `RGBDDeviceRigConfig` (`rgbd_backend: realsense`) | `RGBDFrame` |

`WebcamPairStreamer` uses `grab()`/`retrieve()` on both cameras to reduce temporal offset. `SingleDeviceStreamer` owns crop/split/resize behavior for combined stereo frames.

Both RGB-D streamers implement `RGBDStreamerProtocol` (`open` / `release` / `is_opened` / `grab`), and `FrameProvider` picks between them from `RGBDDeviceRigConfig.rgbd_backend`:

* `OpenCvRGBDStreamer` retrieves color and depth channels from a generic `cv2.VideoCapture` (OpenNI `CAP_OPENNI_*` flags) and returns empty depth explicitly when the backend cannot supply it. No vendor SDK.
* `RealSenseRGBDStreamer` drives an Intel RealSense through `pyrealsense2`: hardware-aligned depth, device depth-scale → uint16 mm, emitter / laser-power / visual-preset control, a spatial/temporal/hole-filling/decimation post-processing chain, and a colour-intrinsics accessor. The `pyrealsense2` import is deferred, so the module stays importable without the SDK (`pip install -r requirements/camera-realsense.txt` to enable it). The real device round-trip runs on-box; the driver logic is unit-tested with an injected fake `rs` module (`tests/test_realsense_streamer.py`).

## Unit and Precision Notes

Image arrays keep the dtype returned by OpenCV. Robotics geometry and millimeter transforms are not represented here; they belong to `backend.src.geometry` and `backend.src.calibration`. RGB-D depth values are documented as millimeters when the backend supplies depth.

## Tests

`tests/test_camera_boundaries.py` covers canonical frame imports, frame validation, `FrameProvider` lifecycle and rig behavior, and `StereoCapturePipeline` rig resolution/path behavior without real hardware.