"""Root :class:`AppConfig` plus the camera/models composition.

Hierarchy::

    AppConfig
    +-- camera   : CameraConfig         (cameras + matcher + eye-to-hand)
    +-- models   : ModelsConfig         (ML/CV model configs)
    +-- robot    : RobotConfig | None   (optional robot tree)
    `-- runtime  : RuntimeConfig        (app-service tuning, see runtime.py)

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

    objectdetector: ObjectDetectorConfig
    segmenter: SegmenterConfig
    stt: SpeechToTextConfig
    #: Optional MediaPipe hand/gesture surface. Standalone: nothing in the grasp pipeline reads
    #: these. They are read when a caller builds a detector via `src.models.handdetection.factory`,
    #: or by that package's `python -m` CLI.
    handdetect: HandDetectConfig = Field(default_factory=HandDetectConfig)
    gesturedetect: GestureDetectConfig = Field(default_factory=GestureDetectConfig)
    # Perception-backend selection by hand: two independently-settable keys with no cross-check, so
    # every detector x segmenter combination builds, including ones where the prompt means something
    # different to each half. `pipeline` below is the cross-checked way to choose a stack.
    detector: Literal["groundingdino", "rtdetr"] = "groundingdino"
    segmenter_backend: Literal["sam2", "oneformer"] = "sam2"
    rtdetr: ObjectDetectorConfig | None = None
    oneformer: OneFormerConfig | None = None
    #: A perception stack chosen in one line, with the combinations validated fail-closed. ``None``,
    #: the default, means the two keys above are in force and behaviour is byte-identical to a build
    #: without this block.
    pipeline: PipelineConfig | None = None


class AppConfig(StrictModel):
    """Root configuration object returned by :func:`config.load_config`."""

    camera: CameraConfig
    models: ModelsConfig
    robot: RobotConfig | None = None
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
