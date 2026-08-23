"""Shared constants for the calibration package."""

from __future__ import annotations

from typing import Final

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------

# Main calibration log directory (relative to the root of the repo)
CALIBRATION_LOG_DIR: Final[str] = "logs/backend/calibration"


EYE_HAND_CALIBRATOR_LOG_FILE: Final[str] = "eye_hand_calibrator.log"
EYE_HAND_DATASET_LOG_FILE: Final[str] = "eye_hand_dataset.log"
SERIALIZATION_LOG_FILE: Final[str] = "serialization.log"
HAND_EYE_AXXB_LOG_FILE: Final[str] = "hand_eye_axxb.log"
UMEYAMA_RIGID_LOG_FILE: Final[str] = "umeyama_rigid.log"
STEREO_FACTORY_LOG_FILE: Final[str] = "stereo_factory.log"
STEREO_CALIBRATOR_LOG_FILE: Final[str] = "stereo_calibrator.log"