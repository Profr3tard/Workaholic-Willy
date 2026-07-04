"""Calibration APIs for stereo vision, eye-hand solving, and persistence.

The package owns calibration domain logic only. Camera device I/O, robot
vendor drivers, FastAPI routes, and application pipelines stay outside this
boundary. OpenCV usage is isolated to the stereo subpackage; hand-eye solvers
and typed persistence stay pure NumPy / geometry.
"""

from __future__ import annotations

from .exceptions import (
    CalibrationDataError,
    CalibrationError,
    CalibrationSolveError,
    ExtrinsicsError,
    StereoCalibrationError,
)
from .extrinsics import EXTRINSICS_SCHEMA, Extrinsics
from .eye_hand import (
    EYE_HAND_DATASET_SCHEMA,
    EyeHandCalibrationResult,
    EyeHandCalibrationSettings,
    EyeHandDataset,
    EyeHandSample,
    EyeInHandCalibrator,
    EyeToHandCalibrator,
    MountingMode,
)
from .helpers import proj_to_K, unit_scaling, validate_image_pair_shapes
from .quality import (
    DEFAULT_BANDS_MM,
    DEFAULT_BANDS_PX,
    QUALITY_LABELS,
    QualityBandsMm,
    QualityBandsPx,
    QualityLabel,
    classify_rmse,
)
from .serialization import (
    extrinsics_from_dict,
    extrinsics_to_dict,
    load_extrinsics,
    save_extrinsics,
)
from .solver import HandEyeAXXB, UmeyamaRigid

__all__ = [
    "DEFAULT_BANDS_MM",
    "DEFAULT_BANDS_PX",
    "EXTRINSICS_SCHEMA",
    "EYE_HAND_DATASET_SCHEMA",
    "QUALITY_LABELS",
    # Exceptions
    "CalibrationDataError",
    "CalibrationError",
    "CalibrationSolveError",
    # Core type
    "Extrinsics",
    "ExtrinsicsError",
    "EyeHandCalibrationResult",
    "EyeHandCalibrationSettings",
    "EyeHandDataset",
    "EyeHandSample",
    "EyeInHandCalibrator",
    "EyeToHandCalibrator",
    "MountingMode",
    "proj_to_K",
    "QualityBandsMm",
    "QualityBandsPx",
    "StereoCalibrationError",
    # Quality bands
    "QualityLabel",
    "classify_rmse",
    "extrinsics_from_dict",
    # Serialisation
    "extrinsics_to_dict",
    "load_extrinsics",
    "save_extrinsics",
    "unit_scaling",
    "validate_image_pair_shapes",
    # Solvers
    "HandEyeAXXB",
    "UmeyamaRigid",
]
