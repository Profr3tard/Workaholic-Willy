"""Build the detectors from config the readers that make `models.handdetect` mean something."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

from config.schema.models import GestureDetectConfig, HandDetectConfig
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
    """`models.handdetect` + the caller's calibration -> a 3-D hand search."""
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
