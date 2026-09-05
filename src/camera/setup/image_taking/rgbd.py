"""RGB-D runtime streamers: a generic OpenCV backend and an Intel RealSense backend.

Both return the canonical :class:`RGBDFrame` (BGR colour + uint16 millimetre depth),
so :class:`~src.camera.orchestration.frame_provider.FrameProvider` can select
either via ``RGBDDeviceRigConfig.rgbd_backend`` without the rest of the pipeline caring.

* :class:`OpenCvRGBDStreamer`, a generic ``cv2.VideoCapture`` + OpenNI depth channel.
  No vendor SDK; depth is only available when the OpenCV build exposes it.
* :class:`RealSenseRGBDStreamer`, Intel RealSense via ``pyrealsense2`` (hardware-aligned
  depth, device depth-scale, emitter/laser/preset control, a post-processing filter
  chain, intrinsics). The SDK import is deferred so this module stays importable without
  it; the real librealsense round-trip runs only on a physical device (validated on-box).
  The driver *logic* is unit-tested with an injected fake ``rs`` module.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Protocol, runtime_checkable

import cv2 as cv
import numpy as np

from config.schema.camera import RGBDDeviceRigConfig
from src.camera.setup.image_taking.frames import RGBDFrame
from src.camera.setup.quality import configure_camera_for_quality


@runtime_checkable
class RGBDStreamerProtocol(Protocol):
    """Runtime contract every RGB-D streamer backend fulfils."""

    def open(self) -> None: ...
    def release(self) -> None: ...
    def is_opened(self) -> bool: ...
    def grab(self) -> RGBDFrame: ...


# ------------------------------------------------------------------
# Generic OpenCV / OpenNI RGB-D streamer
# ------------------------------------------------------------------

class OpenCvRGBDStreamer:
    """Generic RGB-D streamer over ``cv2.VideoCapture`` (OpenNI depth channels).

    Returns colour and depth frames when the OpenCV backend exposes depth via the
    ``CAP_OPENNI_*`` retrieve flags. It does NOT talk to any vendor SDK, so on a
    device whose depth is only reachable through its native SDK (e.g. a RealSense
    over librealsense) the depth channel comes back empty, so use
    :class:`RealSenseRGBDStreamer` there. Quality settings from ``quality.py`` are
    applied to the colour stream.
    """

    def __init__(self, config: RGBDDeviceRigConfig):
        self.cfg = config
        self.logger = logging.getLogger(f"{__name__}.{config.rig_id}")

        self.device_index = int(config.device_index)
        self.color_res = tuple(config.color_resolution)
        self.depth_res = tuple(config.depth_resolution)
        self.align = config.align_depth_to_color
        self.fps = config.fps
        self.backend = config.backend

        self._cap: cv.VideoCapture | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the RGB-D device and apply quality settings to the colour stream."""
        self._cap = cv.VideoCapture(self.device_index, self.backend)

        if not self._cap.isOpened():
            raise OSError(f"Could not open RGB-D device index {self.device_index}")

        q = self.cfg.quality
        settings = configure_camera_for_quality(
            cap=self._cap,
            width=self.color_res[0],
            height=self.color_res[1],
            fps=self.fps,
            prefer_uncompressed=q.prefer_uncompressed,
            manual_exposure=q.manual_exposure,
            manual_gain=q.manual_gain,
            manual_wb=q.manual_wb,
            disable_auto_features=q.disable_auto_features,
            warmup_frames=q.warmup_frames,
        )
        # Fail closed if the device did not honour the requested colour
        # resolution (mirrors SingleDeviceStreamer._validate_settings). Depth
        # resolution is NOT verifiable through this single-VideoCapture path, so
        # it is intentionally left unchecked here.
        if (settings["width"], settings["height"]) != self.color_res:
            raise ValueError(
                f"RGB-D colour resolution mismatch: "
                f"{settings['width']}x{settings['height']} != "
                f"{self.color_res[0]}x{self.color_res[1]}"
            )
        if abs(settings["fps"] - self.fps) > q.fps_tolerance:
            self.logger.warning(
                "RGB-D FPS differs: actual=%.1f, requested=%d, tol=%.1f",
                settings["fps"], self.fps, q.fps_tolerance,
            )
        self.logger.info(
            "RGB-D device %d opened: FOURCC=%s %dx%d @ %.1f fps",
            self.device_index,
            settings["fourcc_str"],
            settings["width"],
            settings["height"],
            settings["fps"],
        )

    def release(self) -> None:
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
        self._cap = None

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------

    def grab(self) -> RGBDFrame:
        """
        Grab a colour frame and its associated depth map.

        The colour image is retrieved from channel 0, and the depth map
        from channel 1 (CAP_OPENNI_DEPTH_MAP).  If depth is not available,
        the depth array will be empty (shape ``(0,)``).

        Returns:
            RGBDFrame with .color (BGR uint8) and .depth (uint16 mm).
        """
        if not self.is_opened():
            raise RuntimeError("Device is not open. Call open() first.")

        assert self._cap is not None  # guaranteed by is_opened() above
        ok = self._cap.grab()
        if not ok:
            raise RuntimeError("Failed to grab frame from RGB-D device.")

        # Colour channel
        _, color = self._cap.retrieve(flag=cv.CAP_OPENNI_BGR_IMAGE)
        if color is None:
            raise RuntimeError("Failed to retrieve colour frame from RGB-D device.")

        # Depth channel, which may not be available for all backends or devices
        _, depth = self._cap.retrieve(flag=cv.CAP_OPENNI_DEPTH_MAP)
        if depth is None:
            self.logger.debug("Depth channel not available, returning empty array.")
            depth = np.empty(0, dtype=np.uint16)

        return RGBDFrame(color=color, depth=depth)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> OpenCvRGBDStreamer:
        self.open()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> Literal[False]:
        self.release()
        return False


# ------------------------------------------------------------------
# Intel RealSense RGB-D streamer (pyrealsense2)
# ------------------------------------------------------------------

class RealSenseRGBDStreamer:
    """Intel RealSense RGB-D streamer over ``pyrealsense2``.

    Streams hardware time-synced BGR colour + Z16 depth, optionally aligned to the
    colour frame, converts depth to uint16 millimetres via the device depth-scale,
    and applies the configured post-processing filter chain. The ``pyrealsense2``
    import is deferred (see :meth:`_import_rs`) so importing this module never
    requires the SDK; unit tests inject a fake ``rs`` module via ``rs_module`` to
    exercise the driver logic without hardware.
    """

    def __init__(self, config: RGBDDeviceRigConfig, *, rs_module: Any | None = None) -> None:
        self.cfg = config
        self.rs_cfg = config.realsense
        self.logger = logging.getLogger(f"{__name__}.{config.rig_id}")

        self.color_res = tuple(config.color_resolution)
        self.depth_res = tuple(config.depth_resolution)
        self.fps = int(config.fps)
        self.serial = config.serial_number
        self.align_to_color = config.align_depth_to_color

        self._rs = rs_module
        self._pipeline: Any | None = None
        self._align: Any | None = None
        self._filters: list[Any] = []
        self._depth_scale_m: float | None = None
        self._intrinsics: np.ndarray | None = None
        self._distortion: np.ndarray | None = None

    # ------------------------------------------------------------------
    # SDK import seam
    # ------------------------------------------------------------------

    def _import_rs(self) -> Any:
        """Return the ``pyrealsense2`` module (or the injected fake), importing lazily."""
        if self._rs is not None:
            return self._rs
        try:
            import pyrealsense2 as rs  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover (exercised only on bare envs)
            raise ImportError(
                "pyrealsense2 is not installed. Install it with "
                "`pip install -r requirements/camera-realsense.txt`, or pass a "
                "custom rs_module."
            ) from exc
        self._rs = rs
        return rs

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Start the RealSense pipeline and configure depth sensor + filters."""
        rs = self._import_rs()

        rs_config = rs.config()
        if self.serial:
            rs_config.enable_device(self.serial)
        rs_config.enable_stream(
            rs.stream.color, self.color_res[0], self.color_res[1], rs.format.bgr8, self.fps
        )
        rs_config.enable_stream(
            rs.stream.depth, self.depth_res[0], self.depth_res[1], rs.format.z16, self.fps
        )

        self._pipeline = rs.pipeline()
        profile = self._pipeline.start(rs_config)

        depth_sensor = profile.get_device().first_depth_sensor()
        self._configure_depth_sensor(rs, depth_sensor)
        self._depth_scale_m = self.rs_cfg.depth_units_m or float(depth_sensor.get_depth_scale())

        self._align = rs.align(rs.stream.color) if self.align_to_color else None
        self._filters = self._build_filters(rs)

        for _ in range(self.cfg.quality.warmup_frames):
            self._pipeline.wait_for_frames()

        self._intrinsics = self._read_intrinsics(rs, profile)
        if self.rs_cfg.export_intrinsics and self._intrinsics is not None:
            self._export_intrinsics(self._intrinsics)

        self.logger.info(
            "RealSense opened: colour %dx%d + depth %dx%d @ %d fps | depth_scale=%.6g m/unit | "
            "align=%s | filters=%d",
            self.color_res[0], self.color_res[1], self.depth_res[0], self.depth_res[1],
            self.fps, self._depth_scale_m, self.align_to_color, len(self._filters),
        )

    def release(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception as exc:  # pragma: no cover (defensive logging)
                self.logger.warning("RealSense pipeline stop failed: %s", exc)
        self._pipeline = None
        self._align = None
        self._filters = []

    def is_opened(self) -> bool:
        return self._pipeline is not None

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------

    def grab(self) -> RGBDFrame:
        """Grab an (optionally aligned + filtered) colour/depth pair as an RGBDFrame."""
        if self._pipeline is None:
            raise RuntimeError("Device is not open. Call open() first.")

        frameset = self._pipeline.wait_for_frames()
        if self._align is not None:
            frameset = self._align.process(frameset)

        depth_frame = frameset.get_depth_frame()
        color_frame = frameset.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("RealSense returned an incomplete frameset (missing colour or depth).")

        for filt in self._filters:
            depth_frame = filt.process(depth_frame)

        color = np.ascontiguousarray(np.asanyarray(color_frame.get_data()))
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_mm = self._match_color_size(self._to_millimetres(depth_raw), color)
        return RGBDFrame(color=color, depth=depth_mm)

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_intrinsics(self) -> np.ndarray | None:
        """Return the 3x3 colour-stream camera matrix K (available after ``open()``)."""
        return None if self._intrinsics is None else self._intrinsics.copy()

    def get_distortion(self) -> np.ndarray | None:
        """Return the colour-stream distortion coefficients (Brown-Conrady, available after ``open()``).

        ``None`` if the device did not report them; a PnP solve should then pass ``np.zeros(5)``. On the
        aligned colour stream these are usually ~0, but that is measured, not assumed.
        """
        return None if self._distortion is None else self._distortion.copy()

    @property
    def depth_scale_m(self) -> float | None:
        """Metres per raw depth unit reported by the device (available after ``open()``)."""
        return self._depth_scale_m

    # ------------------------------------------------------------------
    # Internal: depth conversion
    # ------------------------------------------------------------------

    def _to_millimetres(self, depth_raw: np.ndarray) -> np.ndarray:
        """Convert raw device depth units to a uint16 millimetre map."""
        scale_m = float(self._depth_scale_m or 0.0)
        depth_mm = depth_raw.astype(np.float64) * scale_m * 1000.0
        return np.clip(depth_mm, 0.0, 65535.0).astype(np.uint16)

    @staticmethod
    def _match_color_size(depth: np.ndarray, color: np.ndarray) -> np.ndarray:
        """Nearest-resize depth to the colour grid so RGBDFrame's size invariant holds.

        Only bites when depth and colour differ (e.g. decimation shrank depth, or an
        un-aligned native depth resolution), and nearest keeps depth values intact.
        """
        if depth.shape[:2] != color.shape[:2]:
            h, w = color.shape[:2]
            depth = cv.resize(depth, (w, h), interpolation=cv.INTER_NEAREST)
        return depth

    # ------------------------------------------------------------------
    # Internal: device and filter configuration
    # ------------------------------------------------------------------

    def _configure_depth_sensor(self, rs: Any, sensor: Any) -> None:
        """Apply emitter, laser power, depth-units override and visual preset."""
        if sensor.supports(rs.option.emitter_enabled):
            sensor.set_option(rs.option.emitter_enabled, 1.0 if self.rs_cfg.enable_emitter else 0.0)
        if self.rs_cfg.laser_power_mw is not None and sensor.supports(rs.option.laser_power):
            sensor.set_option(rs.option.laser_power, float(self.rs_cfg.laser_power_mw))
        if self.rs_cfg.depth_units_m is not None and sensor.supports(rs.option.depth_units):
            sensor.set_option(rs.option.depth_units, float(self.rs_cfg.depth_units_m))
        if self.rs_cfg.visual_preset:
            self._apply_visual_preset(rs, sensor, self.rs_cfg.visual_preset)

    def _apply_visual_preset(self, rs: Any, sensor: Any, preset: str) -> None:
        """Best-effort depth visual-preset selection by (case-insensitive) name.

        The preset enum is device-family specific, so a miss is logged and skipped
        rather than raised, because a bad preset name must not take the whole stream down.
        """
        if not sensor.supports(rs.option.visual_preset):
            self.logger.warning("Device does not support visual_preset; ignoring %r", preset)
            return
        try:  # pragma: no cover (enum shape is device/SDK specific (on-box only))
            enum = rs.rs400_visual_preset
            match = next(
                (m for m in enum.__members__.values() if m.name.lower() == preset.lower()),
                None,
            )
            if match is None:
                self.logger.warning("Unknown visual_preset %r; leaving device default", preset)
                return
            sensor.set_option(rs.option.visual_preset, float(int(match)))
        except Exception as exc:  # pragma: no cover (defensive, on-box only)
            self.logger.warning("Failed to apply visual_preset %r: %s", preset, exc)

    def _build_filters(self, rs: Any) -> list[Any]:
        """Build the depth post-processing chain in librealsense's recommended order."""
        pp = self.rs_cfg.post_processing
        filters: list[Any] = []
        if pp.decimation:
            dec = rs.decimation_filter()
            dec.set_option(rs.option.filter_magnitude, float(pp.decimation_magnitude))
            filters.append(dec)
        if pp.spatial:
            filters.append(rs.spatial_filter())
        if pp.temporal:
            filters.append(rs.temporal_filter())
        if pp.hole_filling:
            hole = rs.hole_filling_filter()
            hole.set_option(rs.option.holes_fill, float(pp.hole_filling_mode))
            filters.append(hole)
        return filters

    def _read_intrinsics(self, rs: Any, profile: Any) -> np.ndarray | None:
        """Read the colour-stream 3x3 K matrix from the active profile, and store its distortion.

        The D435 ships factory-calibrated: the SDK hands over ``intr.coeffs`` (the 5-term
        Brown-Conrady distortion) right next to fx/fy/ppx/ppy. It used to be dropped here, so every
        downstream ArUco/PnP solve ran with an implicit zero-distortion assumption. We keep it now (see
        :meth:`get_distortion` and decision D1); the aligned colour stream is typically near-zero, but
        that is a fact to read off the device, not to assume.
        """
        try:
            stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intr = stream.get_intrinsics()
        except Exception as exc:  # pragma: no cover (defensive, on-box only)
            self.logger.warning("Could not read RealSense intrinsics: %s", exc)
            return None
        coeffs = getattr(intr, "coeffs", None)
        self._distortion = (
            np.asarray(coeffs, dtype=np.float64).reshape(-1) if coeffs is not None else None
        )
        return np.array(
            [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def _export_intrinsics(self, k: np.ndarray) -> None:
        """Persist the colour intrinsics to the rig's ``intrinsics.json``."""
        from src.utility import dump_json, ensure_dir

        path = self.cfg.calibration_paths.intrinsics_file
        ensure_dir(self.cfg.calibration_paths.base_dir)
        dump_json(
            {
                "fx": float(k[0, 0]),
                "fy": float(k[1, 1]),
                "cx": float(k[0, 2]),
                "cy": float(k[1, 2]),
                "width": int(self.color_res[0]),
                "height": int(self.color_res[1]),
                # The factory distortion, persisted so `load_intrinsics` can hand a PnP solve the real
                # coefficients rather than an assumed zero vector. Empty list if the device gave none.
                "dist": [] if self._distortion is None else [float(c) for c in self._distortion],
            },
            path,
        )
        self.logger.info("Exported RealSense colour intrinsics (+distortion) to %s", path)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> RealSenseRGBDStreamer:
        self.open()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> Literal[False]:
        self.release()
        return False


AnyRGBDStreamer = OpenCvRGBDStreamer | RealSenseRGBDStreamer
