"""Build the detectors from config: the readers of `models.handdetect` and `models.gesturedetect`.

Each function consumes its block field by field, and these are the only readers of those two
blocks, so a key dropped here is a key nobody reads. The detector keys reach MediaPipe;
`palm_patch_radius_px` and `min_depth_samples` reach `HandFinder`, which samples depth around the
palm.

The 3-D builder does not source its transforms from config. A CAMERA->BASE transform is a
calibration artefact with an existing loader (`grasping.fusion` extrinsics, or an eye-in-hand
resolver composed from the live TCP), and a second spelling of it in `models.handdetect` would be a
second source of truth for the most safety-relevant number here. The caller passes the transforms it
already has.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

from src.config.schema.models import GestureDetectConfig, HandDetectConfig
from src.calibration.stereo.manager import StereoCam3D
from src.camera.orchestration.frame_provider import FrameProvider
from src.models.handdetection.gestures import ThumbGestureRecognizer
from src.models.handdetection.hand_finder import HandFinder, HandObserver
from src.models.handdetection.palm_detector import PalmDetector

__all__ = [
    "build_gesture_recognizer",
    "build_hand_finder",
    "build_palm_detector",
]


def build_palm_detector(config: HandDetectConfig) -> PalmDetector:
    """`models.handdetect` -> a landmark detector. Raises if mediapipe or the bundle is missing."""
    return PalmDetector(
        config.model_path,
        max_hands=config.max_hands,
        threshold=config.threshold,
        tracking_threshold=config.tracking_threshold,
        presence_threshold=config.presence_threshold,
        config_key="models.handdetect.model_path",
    )


def build_gesture_recognizer(config: GestureDetectConfig) -> ThumbGestureRecognizer:
    """`models.gesturedetect` -> a thumbs-up/down recogniser that also reports palm centres."""
    return ThumbGestureRecognizer(
        config.model_path,
        max_hands=config.max_hands,
        threshold=config.threshold,
        tracking_threshold=config.tracking_threshold,
        presence_threshold=config.presence_threshold,
        min_gesture_confidence=config.min_gesture_confidence,
        config_key="models.gesturedetect.model_path",
    )


def build_hand_finder(
    config: HandDetectConfig,
    *,
    provider: FrameProvider,
    transforms: Mapping[str, np.ndarray],
    observer: Optional[HandObserver] = None,
    stereo: Optional[StereoCam3D] = None,
    camera_matrices: Optional[Mapping[str, np.ndarray]] = None,
    rig_ids: Optional[Sequence[str]] = None,
) -> HandFinder:
    """`models.handdetect` + the caller's calibration -> a 3-D hand search.

    `observer` defaults to a landmark-only detector built from the same block. Pass a
    `ThumbGestureRecognizer` from `build_gesture_recognizer` when the located hand should also carry
    a gesture: that model returns landmarks too, so nothing is detected twice.
    """
    return HandFinder(
        observer if observer is not None else build_palm_detector(config),
        provider=provider,
        transforms=transforms,
        stereo=stereo,
        camera_matrices=camera_matrices,
        rig_ids=rig_ids,
        palm_patch_radius_px=config.palm_patch_radius_px,
        min_depth_samples=config.min_depth_samples,
    )
