"""Shared constants for the hand-detection package."""

from __future__ import annotations

from typing import Final

from src.models.constants import MODELS_LOG_DIR

__all__ = [
    "GESTURE_MODEL_URL",
    "GESTURE_RECOGNIZER_LOG_FILE",
    "HAND_FINDER_LOG_FILE",
    "HAND_LANDMARK_MODEL_URL",
    "MODELS_LOG_DIR",
    "PALM_DETECTOR_LOG_FILE",
]

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
HAND_FINDER_LOG_FILE: Final[str] = "hand_finder.log"
PALM_DETECTOR_LOG_FILE: Final[str] = "palm_detector.log"
GESTURE_RECOGNIZER_LOG_FILE: Final[str] = "gesture_recognizer.log"


# URL constants for the MediaPipe hand-landmarker and gesture-recognizer `.task` bundles.
_MEDIAPIPE_MODEL_BASE: Final[str] = "https://storage.googleapis.com/mediapipe-models"
HAND_LANDMARK_MODEL_URL: Final[str] = (
    f"{_MEDIAPIPE_MODEL_BASE}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
GESTURE_MODEL_URL: Final[str] = (
    f"{_MEDIAPIPE_MODEL_BASE}/gesture_recognizer/gesture_recognizer/float16/latest"
    "/gesture_recognizer.task"
)
