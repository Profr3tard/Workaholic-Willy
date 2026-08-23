"""Shared constants for the models package."""

from __future__ import annotations

from typing import Final

# ----------------------------------------------------------------------
# Main log directory
# ----------------------------------------------------------------------
MODELS_LOG_DIR: Final[str] = "logs/backend/models"


# ----------------------------------------------------------------------
# Models and training logs
# ----------------------------------------------------------------------
DETECTOR_LOG_FILE: Final[str] = "GroundingDINOObjectDetector.log"
RTDETR_LOG_FILE: Final[str] = "rtdetr_detector.log"
SEGMENTER_LOG_FILE: Final[str] = "sam2_segmenter.log"
ONEFORMER_LOG_FILE: Final[str] = "oneformer_segmenter.log"
HAND_FINDER_LOG_FILE: Final[str] = "hand_finder.log"
WHISPER_LOG_FILE: Final[str] = "whisper_model.log"
PERCEPTION_BACKEND_LOG_FILE: Final[str] = "perception_backend.log"
VLM_PARSING_LOG_FILE: Final[str] = "vlm_parsing.log"
RTDETR_TRAIN_LOG_FILE: Final[str] = "rtdetr_train.log"

# ----------------------------------------------------------------------
# The VLM route + the routing seam
# ----------------------------------------------------------------------

#: The grounding VLM wrapper (``vlm/qwen.py``): which checkpoint loaded,
#: what it cost, and how many boxes each answer produced.
VLM_GROUNDER_LOG_FILE: Final[str] = "vlm_grounder.log"

#: The refuse-vs-degrade guard (``vlm/availability.py``).
VLM_AVAILABILITY_LOG_FILE: Final[str] = "vlm_availability.log"

#: Which route each prompt took (``routed_backend.py``). Read to answer "why was that pick slow" 
#: the answer is usually a word in the prompt that sent it to the VLM.
PERCEPTION_ROUTING_LOG_FILE: Final[str] = "perception_routing.log"

#: What :func:`~src.models.factory.build_perception` actually assembled.
MODELS_FACTORY_LOG_FILE: Final[str] = "models_factory.log"