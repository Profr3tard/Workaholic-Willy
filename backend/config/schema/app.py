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

from pydantic import Field, model_validator

from ._base import StrictModel
from .camera import (
    CameraSystemConfig,
    HandEyeConfig,
    EyeToHandConfig,
    StereoMatcherConfig,
)
from .models import (
    GestureDetectConfig,
    HandDetectConfig,
    ObjectDetectorConfig,
    SegmenterConfig,
    SimplifierConfig,
    SpeechToTextConfig,
)
from .robot import RobotConfig
from .runtime import RuntimeConfig


class CameraCalibrationQualityBandsPx(StrictModel):
    """RMSE thresholds (px) for stereo camera calibration quality."""

    excellent: float = Field(default=0.5, gt=0.0)
    good: float = Field(default=1.0, gt=0.0)
    marginal: float = Field(default=2.0, gt=0.0)

    @model_validator(mode="after")
    def _check_ordering(self) -> CameraCalibrationQualityBandsPx:
        if not (self.excellent < self.good < self.marginal):
            raise ValueError(
                f"quality bands must be strictly ordered: "
                f"{self.excellent} < {self.good} < {self.marginal}"
            )
        return self


class CameraConfig(StrictModel):
    """Camera section: rigs, stereo matcher and hand-eye calibration."""

    cameras: CameraSystemConfig
    stereomatcher: StereoMatcherConfig
    eye_to_hand: EyeToHandConfig
    hand_eye: HandEyeConfig = Field(default_factory=HandEyeConfig)
    quality_bands_px: CameraCalibrationQualityBandsPx = Field(
        default_factory=CameraCalibrationQualityBandsPx
    )


class ModelsConfig(StrictModel):
    """ML / CV model configurations."""

    handdetect: HandDetectConfig
    gesturedetect: GestureDetectConfig
    objectdetector: ObjectDetectorConfig
    segmenter: SegmenterConfig
    simplifier: SimplifierConfig
    stt: SpeechToTextConfig


class AppConfig(StrictModel):
    """Root configuration object returned by :func:`backend.config.load_config`."""

    camera: CameraConfig
    models: ModelsConfig
    robot: RobotConfig | None = None
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)