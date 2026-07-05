"""Shared constants for the models package.

Centralising log paths here avoids the magic-string drift that creeps
in when the same directory is referenced by every model class.
"""

from __future__ import annotations

from typing import Final

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
#: Single directory under which every model class drops its rotating
#: log file. Resolved relative to the process working directory by
#: :func:`backend.src.utility.log_cfg.create_logger`.
MODELS_LOG_DIR: Final[str] = "logs/backend/models"

#: Per-model log-file names. Kept separate (rather than the consolidated
#: single-file pattern used by the robot package) because the model
#: subsystems run independently and operators usually want to grep one
#: at a time without sifting through the others.
ZEROSHOT_DETECTOR_LOG_FILE: Final[str] = "GroundingDINOObjectDetector.log"
DETECTOR_LOG_FILE: Final[str] = "RT-DETR-Detector.log"
SAM2_SEGMENTER_LOG_FILE: Final[str] = "SAM2_segmenter.log"
ONE_FORMER_SEGMENTER_LOG_FILE: Final[str] = "One-Former_segmenter.log"
HAND_FINDER_LOG_FILE: Final[str] = "Hand-Finder.log"
WHISPER_LOG_FILE: Final[str] = "Whisper-Speechregognition.log"
SIMPLIFIER_LOG_FILE: Final[str] = "Simplifier_model.log"