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
    """The ML and CV model blocks, and the two ways of choosing a perception stack."""

    objectdetector: ObjectDetectorConfig
    segmenter: SegmenterConfig
    stt: SpeechToTextConfig
    #: Optional MediaPipe hand and gesture surface. Standalone: nothing in the grasp pipeline
    #: reads these. Their readers are `src.models.handdetection.factory` and that package's
    #: `python -m` CLI.
    handdetect: HandDetectConfig = Field(default_factory=HandDetectConfig)
    gesturedetect: GestureDetectConfig = Field(default_factory=GestureDetectConfig)
    # Perception backends chosen by hand: two independently settable keys with no cross-check, so
    # every detector x segmenter combination builds, including ones where the prompt means
    # something different to each half. Kept because assembling a stack by hand is legitimate;
    # `pipeline` below is the cross-checked way to choose for everyday use.
    detector: Literal["groundingdino", "rtdetr"] = "groundingdino"
    segmenter_backend: Literal["sam2", "oneformer"] = "sam2"
    #: The closed-set detector block, needed when ``detector`` is ``rtdetr``. Asking for that
    #: backend without this block refuses the build; there is no fallback to the default backend.
    rtdetr: ObjectDetectorConfig | None = None
    #: The OneFormer block, needed when ``segmenter_backend`` is ``oneformer``. Asking for that
    #: backend without this block refuses the build; there is no fallback to the default backend.
    oneformer: OneFormerConfig | None = None
    #: A perception stack chosen in one line, with the combinations validated fail-closed. Set,
    #: ``build_perception`` decides the stack from it and the two keys above go unread; a build
    #: through ``build_object_detector`` or ``build_segmenter`` reads those two keys whatever this
    #: block says. ``None``, the default, leaves them in force and behaviour is byte-identical to
    #: a build without this block.
    pipeline: PipelineConfig | None = None


class AppConfig(StrictModel):
    """Root configuration object returned by :func:`src.config.load_config`."""

    camera: CameraConfig
    models: ModelsConfig
    robot: RobotConfig | None = None
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
