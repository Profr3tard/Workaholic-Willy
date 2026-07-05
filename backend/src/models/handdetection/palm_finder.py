"""
Hand-detection helpers for the handover pipeline.

Provides:

* **PalmDetector** — lightweight MediaPipe HandLandmarker wrapper that
  uses ``VIDEO`` running mode for *synchronous* per-frame results and
  only loads the hand-landmark model (no gesture recogniser).
* **HandFinder** — searches one or more camera rigs for **exactly one**
  hand and computes its 3-D position in the robot base frame.

These building blocks are consumed by :class:`HandoverPipeline` in
``handover.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import cv2 as cv
import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only (frames.py is numpy-only; no runtime/torch cost)
    from backend.src.camera.setup.image_taking.frames import RGBDFrame, StereoFrame

# mediapipe is OPTIONAL (hand detection is not part of the grasp pipeline; see requirements/voice.txt).
# Guarded so the module imports cleanly without it; constructing a detector raises a clear error.
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python.vision import (
        HandLandmarker,
        HandLandmarkerOptions,
        RunningMode,
    )

    _MEDIAPIPE_AVAILABLE = True
except ImportError:  # pragma: no cover - only on installs without the optional voice extra
    mp = None  # type: ignore[assignment]
    python = None  # type: ignore[assignment]
    HandLandmarker = HandLandmarkerOptions = RunningMode = None  # type: ignore[assignment,misc]
    _MEDIAPIPE_AVAILABLE = False

from backend.src.calibration.stereo.manager import StereoCam3D
from backend.src.models.constants import HAND_FINDER_LOG_FILE, MODELS_LOG_DIR
from backend.src.camera.orchestration.frame_provider import FrameProvider
from backend.src.utility.log_cfg import create_logger
from backend.src.utility.vision import bgr_to_rgb

# ── Constants ───────────────────────────────────────────────────────

# MediaPipe landmark indices used for palm-centre estimation.
# 0 = wrist, 1 = thumb_cmc, 5 = index_mcp,
# 9 = middle_mcp, 13 = ring_mcp, 17 = pinky_mcp
_PALM_INDICES = [0, 5, 9, 13, 17, 1]


# ── Helpers ─────────────────────────────────────────────────────────


def calculate_palm_center(
    landmarks: list[tuple[int, int]],
) -> tuple[float, float]:
    """Return the mean (x, y) of the six key palm landmarks."""
    pts = [landmarks[i] for i in _PALM_INDICES]
    x = sum(p[0] for p in pts) / len(pts)
    y = sum(p[1] for p in pts) / len(pts)
    return x, y


def draw_hand_landmarks(
    image: np.ndarray,
    palm: PalmDetection,
) -> np.ndarray:
    """Draw landmarks and palm centre on a BGR image (returns a copy)."""
    out = image.copy()
    for x, y in palm.landmarks:
        cv.circle(out, (x, y), 4, (0, 255, 0), -1)
    cx, cy = palm.palm_center_xy
    cv.circle(out, (int(cx), int(cy)), 8, (0, 0, 255), -1)
    return out


# ── Data classes ────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PalmDetection:
    """Single-hand detection result from one camera frame.

    ``landmarks`` must contain exactly 21 ``(int, int)`` pixel
    coordinates --- the standard MediaPipe hand-landmark layout.
    ``palm_center_xy`` and ``hand_index`` are validated for finiteness
    and non-negativity in :meth:`__post_init__`.
    """

    palm_center_xy: tuple[float, float]
    landmarks: list[tuple[int, int]]  # 21 MediaPipe hand landmarks
    hand_index: int = 0

    def __post_init__(self) -> None:
        if len(self.landmarks) != 21:
            raise ValueError(
                f"PalmDetection.landmarks must have 21 entries (MediaPipe layout), "
                f"got {len(self.landmarks)}"
            )
        for i, lm in enumerate(self.landmarks):
            if not (isinstance(lm, tuple) and len(lm) == 2):
                raise ValueError(
                    f"PalmDetection.landmarks[{i}] must be a (x, y) tuple, got {lm!r}"
                )
            x, y = lm
            if not (isinstance(x, (int, np.integer)) and isinstance(y, (int, np.integer))):
                raise ValueError(
                    f"PalmDetection.landmarks[{i}] entries must be ints, got {(type(x).__name__, type(y).__name__)}"
                )
        if len(self.palm_center_xy) != 2:
            raise ValueError(
                f"PalmDetection.palm_center_xy must be (x, y), got {self.palm_center_xy!r}"
            )
        cx, cy = self.palm_center_xy
        if not (np.isfinite(cx) and np.isfinite(cy)):
            raise ValueError(
                f"PalmDetection.palm_center_xy must be finite, got {self.palm_center_xy!r}"
            )
        if self.hand_index < 0:
            raise ValueError(
                f"PalmDetection.hand_index must be >= 0, got {self.hand_index}"
            )


@dataclass(frozen=True, slots=True)
class HandPosition3D:
    """3-D position of a detected hand in the robot base frame.

    ``position_base`` and ``position_cam`` are coerced to immutable
    ``float64`` arrays of shape ``(3,)`` in :meth:`__post_init__`. The
    arrays' ``writeable`` flag is set to ``False`` --- mutating them
    raises ``ValueError`` at numpy level.
    """

    position_base: np.ndarray  # (3,) XYZ in base frame (mm), read-only
    position_cam: np.ndarray   # (3,) XYZ in camera frame (mm), read-only
    palm_center_xy: tuple[float, float]
    depth_mm: float
    rig_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("position_base", "position_cam"):
            arr = np.asarray(getattr(self, name), dtype=np.float64).reshape(-1)
            if arr.shape != (3,):
                raise ValueError(
                    f"HandPosition3D.{name} must have shape (3,), got {arr.shape}"
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"HandPosition3D.{name} must be finite, got {arr!r}"
                )
            arr.flags.writeable = False
            object.__setattr__(self, name, arr)
        if len(self.palm_center_xy) != 2:
            raise ValueError(
                f"HandPosition3D.palm_center_xy must be (x, y), got {self.palm_center_xy!r}"
            )
        if not np.isfinite(self.depth_mm):
            raise ValueError(
                f"HandPosition3D.depth_mm must be finite, got {self.depth_mm!r}"
            )


# ── Palm detector ───────────────────────────────────────────────────


class PalmDetector:
    """
    Synchronous hand-landmark detector (MediaPipe VIDEO mode).

    Unlike :class:`BaseHandDetector` this class:

    * Only loads the **hand-landmark** model (no gesture recogniser).
    * Uses ``VIDEO`` running mode for **synchronous** per-frame results
      (no callback latency).
    * Detects up to ``max_hands`` (default 2) so the caller can
      distinguish single-hand vs. multi-hand scenarios.

    Parameters
    ----------
    model_path : str
        File path to the MediaPipe hand-landmark ``.task`` model.
    max_hands : int
        Maximum number of hands to detect per frame.
    threshold : float
        Minimum detection confidence in ``[0, 1]``.
    """

    def __init__(
        self,
        model_path: str,
        max_hands: int = 2,
        threshold: float = 0.5,
    ):
        if not _MEDIAPIPE_AVAILABLE:
            raise ImportError(
                "mediapipe is required for palm/hand detection but is not installed. "
                "Install the optional voice/gesture extra: pip install -r requirements/voice.txt"
            )
        opts = HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=threshold,
        )
        self._detector = HandLandmarker.create_from_options(opts)
        self._timestamp_ms: int = 0

    def detect(self, frame_bgr: np.ndarray) -> list[PalmDetection]:
        """
        Detect hands in a BGR image.

        Returns one :class:`PalmDetection` per detected hand
        (may be 0, 1, or more).
        """
        rgb = bgr_to_rgb(frame_bgr)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._detector.detect_for_video(img, self._timestamp_ms)
        self._timestamp_ms += 33  # ≈ 30 fps spacing

        if not result.hand_landmarks:
            return []

        h, w = frame_bgr.shape[:2]
        detections: list[PalmDetection] = []

        for idx, hand in enumerate(result.hand_landmarks):
            landmarks = [(int(p.x * w), int(p.y * h)) for p in hand]
            palm_xy = calculate_palm_center(landmarks)
            detections.append(
                PalmDetection(
                    palm_center_xy=palm_xy,
                    landmarks=landmarks,
                    hand_index=idx,
                )
            )

        return detections


# ── Multi-rig hand finder ───────────────────────────────────────────


class HandFinder:
    """
    Searches for **exactly one** hand across one or more camera rigs
    and computes its 3-D position in the robot base frame.

    Parameters
    ----------
    model_path : str
        Path to the MediaPipe hand-landmark ``.task`` model.
    threshold : float
        Detection confidence threshold.
    provider : FrameProvider
        Already-opened frame provider.
    stereo : StereoCam3D or None
        Stereo engine (``None`` when all rigs are RGB-D).
    rig_ids : list[str]
        Ordered list of rig IDs to search (primary rig first).
    T_cam_to_base : np.ndarray
        4x4 camera-to-robot-base transform.
    camera_matrix : np.ndarray or None
        3x3 intrinsics for RGB-D back-projection.
    palm_patch_radius_px : int
        Radius of the circular region around the palm centre used for
        robust (median) depth estimation.
    """

    def __init__(
        self,
        model_path: str,
        threshold: float,
        provider: FrameProvider,
        stereo: StereoCam3D | None,
        rig_ids: list[str],
        T_cam_to_base: np.ndarray,
        camera_matrix: np.ndarray | None = None,
        palm_patch_radius_px: int = 15,
    ):
        self.logger = create_logger("HandFinder", HAND_FINDER_LOG_FILE, log_dir=MODELS_LOG_DIR)

        self.palm_detector = PalmDetector(
            model_path=model_path,
            max_hands=2,
            threshold=threshold,
        )
        self.provider = provider
        self.stereo = stereo
        self.rig_ids = list(rig_ids)
        self.T_cam_to_base = np.asarray(T_cam_to_base, dtype=np.float64)
        self.camera_matrix = camera_matrix
        self.palm_patch_radius_px = palm_patch_radius_px

    # ── Public ──────────────────────────────────────────────────────

    def detect_palms(self, frame_bgr: np.ndarray) -> list[PalmDetection]:
        """Detect palms in a given BGR image (thin wrapper)."""
        return self.palm_detector.detect(frame_bgr)

    def compute_hand_3d(
        self,
        palm: PalmDetection,
        frame: StereoFrame | RGBDFrame,
        rig_id: str,
    ) -> HandPosition3D | None:
        """
        Compute the 3-D base-frame position for a detected palm.

        Parameters
        ----------
        palm : PalmDetection
            Detection result from :meth:`detect_palms`.
        frame : StereoFrame or RGBDFrame
            The raw frame that produced the detection.
        rig_id : str
            Which camera rig the frame came from.

        Returns
        -------
        HandPosition3D or None
            ``None`` when depth is unavailable or invalid.
        """
        if self.provider.is_rgbd(rig_id):
            return self._hand_3d_rgbd(palm, cast("RGBDFrame", frame), rig_id)
        return self._hand_3d_stereo(palm, cast("StereoFrame", frame), rig_id)

    def find_hand(
        self,
    ) -> tuple[HandPosition3D | None, np.ndarray | None]:
        """
        Search **all** rigs (in order) for exactly one hand.

        Returns
        -------
        (HandPosition3D, annotated_image)
            On success — the 3-D position and an annotated BGR frame.
        (None, None)
            When no rig yields exactly one hand with valid depth.
        """
        for rig_id in self.rig_ids:
            frame = self.provider.grab(rig_id)
            work = self._extract_work_image(frame, rig_id)
            palms = self.palm_detector.detect(work)

            if len(palms) != 1:
                self.logger.debug(
                    "Rig %s: %d hands detected — skipping.", rig_id, len(palms),
                )
                continue

            hand_3d = self.compute_hand_3d(palms[0], frame, rig_id)
            if hand_3d is None:
                self.logger.debug(
                    "Rig %s: depth invalid for palm — skipping.", rig_id,
                )
                continue

            annotated = draw_hand_landmarks(work, palms[0])
            return hand_3d, annotated

        return None, None

    # ── RGB-D depth path ────────────────────────────────────────────

    def _hand_3d_rgbd(
        self,
        palm: PalmDetection,
        frame: RGBDFrame,
        rig_id: str,
    ) -> HandPosition3D | None:
        cx, cy = palm.palm_center_xy
        depth = frame.depth
        h, w = depth.shape[:2]

        # Median depth inside a small patch around the palm centre.
        mask = np.zeros((h, w), dtype=np.uint8)
        cv.circle(
            mask,
            (int(round(cx)), int(round(cy))),
            self.palm_patch_radius_px,
            1,
            -1,
        )
        patch_depths = depth[mask > 0].astype(np.float64)
        patch_depths = patch_depths[patch_depths > 0]
        if len(patch_depths) == 0:
            return None
        depth_mm = float(np.median(patch_depths))

        # Back-project to camera frame.
        if self.camera_matrix is not None:
            fx = float(self.camera_matrix[0, 0])
            fy = float(self.camera_matrix[1, 1])
            cx_i = float(self.camera_matrix[0, 2])
            cy_i = float(self.camera_matrix[1, 2])
        else:
            fx = fy = w / 2.0
            cx_i, cy_i = w / 2.0, h / 2.0

        X = (cx - cx_i) * depth_mm / fx
        Y = (cy - cy_i) * depth_mm / fy
        Z = depth_mm
        pos_cam = np.array([X, Y, Z])

        R = self.T_cam_to_base[:3, :3]
        t = self.T_cam_to_base[:3, 3]
        pos_base = R @ pos_cam + t

        return HandPosition3D(
            position_base=pos_base,
            position_cam=pos_cam,
            palm_center_xy=(cx, cy),
            depth_mm=depth_mm,
            rig_id=rig_id,
        )

    # ── Stereo depth path ───────────────────────────────────────────

    def _hand_3d_stereo(
        self,
        palm: PalmDetection,
        frame: StereoFrame,
        rig_id: str,
    ) -> HandPosition3D | None:
        if self.stereo is None:
            return None

        cx, cy = palm.palm_center_xy
        h, w = frame.left.shape[:2]

        # Small circular mask for robust 3-D estimation.
        mask = np.zeros((h, w), dtype=np.uint8)
        cv.circle(
            mask,
            (int(round(cx)), int(round(cy))),
            self.palm_patch_radius_px,
            1,
            -1,
        )

        rig_idx = self.provider.get_stereo_rig_index(rig_id)
        rect_l, rect_r = self.stereo.rectify(
            frame.left, frame.right, rig=rig_idx,
        )
        pos_cam = self.stereo.compute_3D_point(
            rect_l, rect_r,
            mask=mask,
            reducer="median",
            unit="mm",
            rig=rig_idx,
        )
        if pos_cam is None:
            return None

        depth_mm = float(pos_cam[2])
        if depth_mm <= 0:
            return None

        R = self.T_cam_to_base[:3, :3]
        t = self.T_cam_to_base[:3, 3]
        pos_base = R @ pos_cam + t

        return HandPosition3D(
            position_base=pos_base,
            position_cam=pos_cam,
            palm_center_xy=(cx, cy),
            depth_mm=depth_mm,
            rig_id=rig_id,
        )

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _extract_work_image(frame: StereoFrame | RGBDFrame, rig_id: str) -> np.ndarray:
        """Return the BGR image suitable for hand detection."""
        if hasattr(frame, "color"):  # RGBDFrame
            return cast("RGBDFrame", frame).color
        return cast("StereoFrame", frame).left  # StereoFrame
