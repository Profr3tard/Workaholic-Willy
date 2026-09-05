"""Shared constants for the calibration package.

Centralising log paths here avoids the magic-string drift that creeps in
when the same directory is referenced by every solver and persistence
helper.
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

#: Per-module log-file names. Kept separate (rather than the aggregate
#: single-file pattern used by the robot package) because a calibration
#: session touches exactly one of these workflows at a time, eye-hand or
#: stereo, and the question afterwards is always "what did that solve see?".
#: A run is a handful of lines per file, so one file per module stays greppable.
EYE_HAND_CALIBRATOR_LOG_FILE: Final[str] = "eye_hand_calibrator.log"
EYE_HAND_DATASET_LOG_FILE: Final[str] = "eye_hand_dataset.log"
SERIALIZATION_LOG_FILE: Final[str] = "serialization.log"
HAND_EYE_AXXB_LOG_FILE: Final[str] = "hand_eye_axxb.log"
UMEYAMA_RIGID_LOG_FILE: Final[str] = "umeyama_rigid.log"
STEREO_FACTORY_LOG_FILE: Final[str] = "stereo_factory.log"
STEREO_CALIBRATOR_LOG_FILE: Final[str] = "stereo_calibrator.log"
