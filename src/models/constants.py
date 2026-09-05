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
#: :func:`src.utility.log_cfg.create_logger`.
MODELS_LOG_DIR: Final[str] = "logs/models"

#: Per-model log-file names. Kept separate (rather than the consolidated
#: single-file pattern used by the robot package) because the model
#: subsystems run independently and operators usually want to grep one
#: at a time without sifting through the others.
DETECTOR_LOG_FILE: Final[str] = "GroundingDINOObjectDetector.log"
RTDETR_LOG_FILE: Final[str] = "rtdetr_detector.log"
SEGMENTER_LOG_FILE: Final[str] = "sam2_segmenter.log"
ONEFORMER_LOG_FILE: Final[str] = "oneformer_segmenter.log"
HAND_FINDER_LOG_FILE: Final[str] = "hand_finder.log"
WHISPER_LOG_FILE: Final[str] = "whisper_model.log"

#: The perception seam (:class:`~src.models.perception_backend.TwoStageBackend`). Its own
#: file because it is the only place that sees detector and segmenter together: a run where every
#: detection lost its mask is invisible in either model's log and obvious in this one.
PERCEPTION_BACKEND_LOG_FILE: Final[str] = "perception_backend.log"

#: The VLM grounding parser. Separate from the model wrapper's log on purpose: these lines record
#: which boxes the parser dropped and why, which is a different question from what the model said.
VLM_PARSING_LOG_FILE: Final[str] = "vlm_parsing.log"

#: RT-DETR fine-tuning. An offline, long-running job rather than a runtime component, so mixing it
#: into the detector's log would bury a pick-time inference line under a training run's history.
RTDETR_TRAIN_LOG_FILE: Final[str] = "rtdetr_train.log"

# ----------------------------------------------------------------------
# The VLM route + the routing seam
# ----------------------------------------------------------------------
# A bare ``logging.getLogger(__name__)`` formats its lines and then drops them on the floor in a repo
# whose CLIs never call ``basicConfig``. The constants below give these four modules the same rotating
# file every other model writes to, without moving their call sites.

#: The grounding VLM wrapper (``vlm/qwen.py``): which checkpoint loaded, what it cost, and how many
#: boxes each answer produced. Separate from the parser's file below: "the model said nothing" and
#: "the model said something this parser could not use" are different diagnoses.
VLM_GROUNDER_LOG_FILE: Final[str] = "vlm_grounder.log"

#: The refuse-vs-degrade guard (``vlm/availability.py``). Its own file because a degraded run is an
#: operational fact about the whole cell, not a detail of one model: every pick after that warning was
#: grounded by a detector that returns a confident box for the wrong object.
VLM_AVAILABILITY_LOG_FILE: Final[str] = "vlm_availability.log"

#: Which route each prompt took (``routed_backend.py``). Read to answer "why was that pick slow":
#: the answer is usually a word in the prompt that sent it to the VLM.
PERCEPTION_ROUTING_LOG_FILE: Final[str] = "perception_routing.log"

#: What :func:`~src.models.factory.build_perception` actually assembled. The models mirror of
#: the robot package's ``grasp_builders.log``: "what is this cell actually running" is the most-asked
#: question here too, and config alone does not answer it (presets and legacy keys both feed in).
MODELS_FACTORY_LOG_FILE: Final[str] = "models_factory.log"

# NB: ``_inference.py`` deliberately has no entry here and keeps its bare ``getLogger``. It is a shared
# load-time helper, so its three warnings describe whichever model called it, and it is imported
# inside ``ProcessPoolExecutor`` workers by the datagen evaluator, where a module-scope handler means
# one rotating file opened per worker process on a single path. ``log_cfg`` serialises writers within a
# process; it cannot across processes. Not worth that for three lines that belong in a model's log.
