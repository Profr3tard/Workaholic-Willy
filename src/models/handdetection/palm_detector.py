"""Palm-centre detection: MediaPipe's hand landmarker, wrapped so the rest of the stack sees types."""

from __future__ import annotations

import threading
import time
from typing import Any, Literal, Optional

import numpy as np
import mediapipe as mp


from mediapipe.tasks import python
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

from src.models.handdetection.constants import (
    HAND_LANDMARK_MODEL_URL,
    MODELS_LOG_DIR,
    PALM_DETECTOR_LOG_FILE,
)
from src.models.handdetection.landmarks import calculate_palm_center
from src.models.handdetection.model_files import require_mediapipe, resolve_model_file
from src.models.handdetection.types import (
    GestureReading,
    Handedness,
    HandGesture,
    HandObservation,
    PalmDetection,
    as_landmark_tuple,
)
from src.utility.log_cfg import create_logger
from src.utility.vision import bgr_to_rgb

__all__ = ["PalmDetector", "handedness_of"]


def handedness_of(categories: Any) -> tuple[Handedness, float]:
    """Map MediaPipe's handedness categories onto the enum, with its score."""
    if not categories:
        return Handedness.UNKNOWN, 0.0
    top = categories[0]
    name = str(getattr(top, "category_name", "") or "").strip().lower()
    score = float(getattr(top, "score", 0.0) or 0.0)
    if name == "left":
        return Handedness.LEFT, score
    if name == "right":
        return Handedness.RIGHT, score
    return Handedness.UNKNOWN, score


class PalmDetector:
    """Detect hands in a BGR frame and report each palm centre in pixels."""

    def __init__(
        self,
        model_path: str,
        *,
        max_hands: int = 2,
        threshold: float = 0.5,
        tracking_threshold: float = 0.5,
        presence_threshold: float = 0.5,
        config_key: str = "models.handdetect.model_path",
    ) -> None:
        require_mediapipe("palm/hand detection")
        resolved = resolve_model_file(
            model_path, config_key=config_key, download_url=HAND_LANDMARK_MODEL_URL
        )

        self.logger = create_logger(
            "PalmDetector", PALM_DETECTOR_LOG_FILE, log_dir=MODELS_LOG_DIR
        )
        self._detector = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=resolved),
                running_mode=RunningMode.VIDEO,
                num_hands=int(max_hands),
                min_hand_detection_confidence=float(threshold),
                min_hand_presence_confidence=float(presence_threshold),
                min_tracking_confidence=float(tracking_threshold),
            )
        )
        self._lock = threading.Lock()
        self._last_timestamp_ms = -1
        self.logger.info(
            "PalmDetector ready: model=%s max_hands=%d detect>=%.2f presence>=%.2f track>=%.2f",
            resolved, max_hands, threshold, presence_threshold, tracking_threshold,
        )

    # ── Timestamps ──────────────────────────────────────────────────────────────────────────────

    def _next_timestamp_ms(self, timestamp_ms: Optional[int]) -> int:
        """A strictly increasing millisecond stamp, which is what VIDEO mode requires."""
        with self._lock:
            proposed = (
                int(timestamp_ms)
                if timestamp_ms is not None
                else int(time.monotonic_ns() // 1_000_000)
            )
            if proposed <= self._last_timestamp_ms:
                proposed = self._last_timestamp_ms + 1
            self._last_timestamp_ms = proposed
            return proposed

    # ── Detection ───────────────────────────────────────────────────────────────────────────────

    def detect(
        self, frame_bgr: np.ndarray, *, timestamp_ms: Optional[int] = None
    ) -> list[PalmDetection]:
        """Detect hands in a BGR image; one `PalmDetection` per hand, possibly none.

        `timestamp_ms` lets a caller that already has real frame timing pass it in
        """
        if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            raise ValueError("PalmDetector.detect needs a non-empty BGR image")

        rgb = bgr_to_rgb(frame_bgr)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect_for_video(image, self._next_timestamp_ms(timestamp_ms))
        return self._to_detections(result, frame_bgr.shape[:2])

    def observe(self, frame_bgr: np.ndarray) -> list[HandObservation]:
        """`HandObserver`: the same detections, wrapped with an explicit "no gesture" reading."""
        return [
            HandObservation(
                palm=palm,
                gesture=GestureReading(
                    gesture=HandGesture.NONE,
                    confidence=0.0,
                    raw_label="",
                    hand_index=palm.hand_index,
                ),
            )
            for palm in self.detect(frame_bgr)
        ]

    @staticmethod
    def _to_detections(result: Any, shape_hw: tuple[int, int]) -> list[PalmDetection]:
        """Convert a MediaPipe result into `PalmDetection`s, denormalising to pixels."""
        hands = getattr(result, "hand_landmarks", None) or []
        if not hands:
            return []
        height, width = shape_hw
        all_handedness = getattr(result, "handedness", None) or []

        detections: list[PalmDetection] = []
        for index, hand in enumerate(hands):
            landmarks = as_landmark_tuple(
                [(int(point.x * width), int(point.y * height)) for point in hand]
            )
            side, score = handedness_of(
                all_handedness[index] if index < len(all_handedness) else None
            )
            detections.append(
                PalmDetection(
                    palm_center_xy=calculate_palm_center(landmarks),
                    landmarks=landmarks,
                    hand_index=index,
                    handedness=side,
                    handedness_score=score,
                )
            )
        return detections

    # ── Lifetime ────────────────────────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Release the MediaPipe graph."""
        closer = getattr(self._detector, "close", None)
        if callable(closer):
            closer()

    def __enter__(self) -> PalmDetector:
        return self

    def __exit__(
        self, exc_type: object, exc_val: object, exc_tb: object
    ) -> Literal[False]:
        self.close()
        return False
