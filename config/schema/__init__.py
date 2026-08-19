"""Public schema surface for the configuration package."""

from ._base import StrictModel, validate_aruco_dict_name
from .app import (
    AppConfig,
    CameraConfig,
    ModelsConfig,
)
from .camera import (
    ArucoDictName,
    BaseRigConfig,
    CalibrationConfig,
    CameraSystemConfig,
    EyeHandRoutineConfig,
    EyeInHandWorkflowConfig,
    EyeToHandWorkflowConfig,
    HandEyeConfig,
    QualityConfig,
    RGBDCalibPaths,
    RGBDDeviceRigConfig,
    SingleDeviceRigConfig,
    StereoCalibPaths,
    StereoMatcherConfig,
    WebcamPairRigConfig,
    WlsFilterConfig,
)
from .models import (
    GestureDetectConfig,
    HandDetectConfig,
    InferenceOptimization,
    ObjectDetectorConfig,
    SegmenterConfig,
    SpeechToTextConfig,
)
from .robot import (
    DummyConfig,
    GripperConfig,
    MotionLimitsConfig,
    RobotCalibrationConfig,
    RobotCalibrationQualityBandsMm,
    RobotConfig,
    RobotSafetyConfig,
    SafePoseConfig,
    SimCameraSchema,
    SimConfig,
    URConfig,
    WorkspaceLimitsConfig,
)
from .runtime import (
    DecisionImagesConfig,
    EventHubConfig,
    ImageEncodingConfig,
    InteractionConfig,
    RunRegistryConfig,
    RuntimeConfig,
)

__all__ = [
    # Root + runtime
    "AppConfig",
    # Camera
    "ArucoDictName",
    "BaseRigConfig",
    "CalibrationConfig",
    "CameraConfig",
    "CameraSystemConfig",
    "DecisionImagesConfig",
    "DummyConfig",
    "EventHubConfig",
    "EyeHandRoutineConfig",
    "EyeInHandWorkflowConfig",
    "EyeToHandWorkflowConfig",
    "HandEyeConfig",
    # Models
    "GestureDetectConfig",
    # Robot
    "GripperConfig",
    "HandDetectConfig",
    "ImageEncodingConfig",
    "InferenceOptimization",
    "InteractionConfig",
    "ModelsConfig",
    "MotionLimitsConfig",
    "ObjectDetectorConfig",
    "QualityConfig",
    "RGBDCalibPaths",
    "RGBDDeviceRigConfig",
    "RobotCalibrationConfig",
    "RobotCalibrationQualityBandsMm",
    "RobotConfig",
    "RobotSafetyConfig",
    "RunRegistryConfig",
    "RuntimeConfig",
    "SafePoseConfig",
    "SegmenterConfig",
    "SimCameraSchema",
    "SimConfig",
    "SingleDeviceRigConfig",
    "SpeechToTextConfig",
    "StereoCalibPaths",
    "StereoMatcherConfig",
    # Base
    "StrictModel",
    "URConfig",
    "WebcamPairRigConfig",
    "WlsFilterConfig",
    "WorkspaceLimitsConfig",
    "validate_aruco_dict_name",
]
