"""Shared constants for the hand-detection package.

Two kinds of constant live here, for the same reason: both are strings that would otherwise be
re-typed at every call site and drift apart silently.

* Log files, following the `models/constants.py` pattern (one directory, one file per subsystem)
  rather than the robot package's aggregate file. Hand detection runs independently of the grasp
  pipeline, so an operator debugging it wants its lines on their own.
* The model-download URLs, which appear in an error message, not in a fetch. Nothing here downloads
  anything: the operator installs the `.task` bundles and points the config at them. The URL is in
  the message because "file not found" without it sends the reader to a search engine.
"""

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
#: The 3-D search across camera rigs. Its own file because its interesting lines are about geometry
#: (which rig saw the hand, what depth came back, why a detection was discarded), and those
#: questions outlive any single frame.
HAND_FINDER_LOG_FILE: Final[str] = "hand_finder.log"

#: The 2-D landmark detector. Separate from the finder above: "MediaPipe saw no hand" and "the hand
#: was seen but had no usable depth" are different failures with different fixes, and burying one in
#: the other's file is how an operator ends up re-aiming a camera that was already aimed correctly.
PALM_DETECTOR_LOG_FILE: Final[str] = "palm_detector.log"

#: The canned gesture classifier. Its own file because gesture lines are a confirmation trail: an
#: operator asking "did the cell see my thumbs-up?" wants one grep, not a scroll through landmarks.
GESTURE_RECOGNIZER_LOG_FILE: Final[str] = "gesture_recognizer.log"

# ----------------------------------------------------------------------
# Where the operator gets the models (quoted in errors; never fetched here)
# ----------------------------------------------------------------------
_MEDIAPIPE_MODEL_BASE: Final[str] = "https://storage.googleapis.com/mediapipe-models"

#: Hand-landmark bundle: 21 landmarks per hand, no gesture head. The smaller, faster model, and all
#: that is needed when only the palm centre is wanted.
HAND_LANDMARK_MODEL_URL: Final[str] = (
    f"{_MEDIAPIPE_MODEL_BASE}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

#: Canned gesture bundle. It embeds a hand landmarker, so its results carry `hand_landmarks` too,
#: which is why `ThumbGestureRecognizer` returns palm detections alongside the gesture instead of
#: making the caller run a second model over the same frame.
GESTURE_MODEL_URL: Final[str] = (
    f"{_MEDIAPIPE_MODEL_BASE}/gesture_recognizer/gesture_recognizer/float16/latest"
    "/gesture_recognizer.task"
)
