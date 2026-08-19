"""Root :class:`AppConfig` plus the camera/models composition.

Hierarchy::

    AppConfig
    ├── camera   : CameraConfig         (cameras + matcher + eye-to-hand)
    ├── models   : ModelsConfig         (ML/CV model configs)
    ├── robot    : RobotConfig | None   (optional robot tree)
    └── runtime  : RuntimeConfig        (app-service tuning, see runtime.py)

All schemas inherit from :class:`StrictModel` (immutable, ``extra="forbid"``).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from ._base import StrictModel
from .camera import (
    CameraSystemConfig,
    HandEyeConfig,
    StereoMatcherConfig,
)
from .models import (
    GestureDetectConfig,
    HandDetectConfig,
    ObjectDetectorConfig,
    OneFormerConfig,
    PipelineConfig,
    SegmenterConfig,
    SpeechToTextConfig,
)
from .robot import RobotConfig
from .runtime import RuntimeConfig


class CameraConfig(StrictModel):
    """Camera section: rigs, stereo matcher and hand-eye calibration."""

    cameras: CameraSystemConfig
    stereomatcher: StereoMatcherConfig
    hand_eye: HandEyeConfig = Field(default_factory=HandEyeConfig)


class ModelsConfig(StrictModel):
    """ML / CV model configurations."""

    handdetect: HandDetectConfig
    gesturedetect: GestureDetectConfig
    objectdetector: ObjectDetectorConfig
    segmenter: SegmenterConfig
    stt: SpeechToTextConfig
    detector: Literal["groundingdino", "rtdetr"] = "groundingdino"
    segmenter_backend: Literal["sam2", "oneformer"] = "sam2"
    rtdetr: ObjectDetectorConfig | None = None
    oneformer: OneFormerConfig | None = None
    pipeline: PipelineConfig | None = None


class AppConfig(StrictModel):
    """Root configuration object returned by :func:`backend.config.load_config`."""

    camera: CameraConfig
    models: ModelsConfig
    robot: RobotConfig | None = None
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
