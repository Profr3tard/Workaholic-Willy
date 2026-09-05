"""Shared constants for the calibration package: the log directory and the
per-module log file names every solver and persistence helper writes to.
"""

from __future__ import annotations

from typing import Final

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
#: Single directory under which every calibration module drops its rotating
#: log file. Resolved to an absolute path by
#: :func:`src.utility.log_cfg.create_logger`.
CALIBRATION_LOG_DIR: Final[str] = "logs/calibration"

#: Per-module log-file names, one file per module rather than the aggregate
#: single file the robot package uses: a session touches one workflow at a
#: time, eye-hand or stereo, and a run is a handful of lines per file.
EYE_HAND_CALIBRATOR_LOG_FILE: Final[str] = "eye_hand_calibrator.log"
EYE_HAND_DATASET_LOG_FILE: Final[str] = "eye_hand_dataset.log"
SERIALIZATION_LOG_FILE: Final[str] = "serialization.log"
HAND_EYE_AXXB_LOG_FILE: Final[str] = "hand_eye_axxb.log"
UMEYAMA_RIGID_LOG_FILE: Final[str] = "umeyama_rigid.log"
STEREO_FACTORY_LOG_FILE: Final[str] = "stereo_factory.log"
STEREO_CALIBRATOR_LOG_FILE: Final[str] = "stereo_calibrator.log"
