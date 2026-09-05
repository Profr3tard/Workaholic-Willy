"""Log directory and per-module log file names shared across calibration."""

from __future__ import annotations

from typing import Final

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
#: Directory every calibration module writes its rotating log file into.
#: :func:`src.utility.log_cfg.create_logger` resolves it against
#: ``WILLY_LOG_DIR`` and makes it absolute.
CALIBRATION_LOG_DIR: Final[str] = "logs/calibration"

#: One log file per module, not the single aggregate file the robot package
#: uses: a calibration session runs one workflow, eye-hand or stereo, and a
#: run leaves a handful of lines per file.
EYE_HAND_CALIBRATOR_LOG_FILE: Final[str] = "eye_hand_calibrator.log"
EYE_HAND_DATASET_LOG_FILE: Final[str] = "eye_hand_dataset.log"
SERIALIZATION_LOG_FILE: Final[str] = "serialization.log"
HAND_EYE_AXXB_LOG_FILE: Final[str] = "hand_eye_axxb.log"
UMEYAMA_RIGID_LOG_FILE: Final[str] = "umeyama_rigid.log"
STEREO_FACTORY_LOG_FILE: Final[str] = "stereo_factory.log"
STEREO_CALIBRATOR_LOG_FILE: Final[str] = "stereo_calibrator.log"
