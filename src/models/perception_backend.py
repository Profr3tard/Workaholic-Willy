"""One seam between "a prompt" and "masks", so a pipeline is a choice rather than an assembly."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from src.models.constants import MODELS_LOG_DIR, PERCEPTION_BACKEND_LOG_FILE
from src.utility.log_cfg import create_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.models.detection.types import Detection
    from src.models.segmentation.types import SegmentationResult

__all__ = ["PerceivedObject", "PerceptionBackend", "TwoStageBackend"]


@dataclass(frozen=True, slots=True)
class PerceivedObject:
    """One grounded object: where the model said it is, and the mask that was cut for it."""

    detection: "Detection"
    segmentation: "SegmentationResult"


@runtime_checkable
class PerceptionBackend(Protocol):
    """Prompt in, grounded objects out. The whole perception contract.

    Implementations are constructed from config by
    :func:`src.models.factory.build_perception` and hold whatever models they need. They are
    stateful only in the sense that they own loaded weights ``perceive`` itself is a pure function
    of its arguments.
    """

    def perceive(self, image_bgr: Any, prompt: str) -> tuple[PerceivedObject, ...]:
        """Ground ``prompt`` in ``image_bgr``. Returns ``()`` when nothing matched.

        An empty result is a legitimate answer, not an error: downstream the pick loop reports
        ``no_perception`` and moves on, which is the honest outcome when the scene does not contain
        what was asked for.
        """
        ...


class TwoStageBackend:
    """Detector then segmenter the chain this repo has always run, extracted verbatim.

    Carried across unchanged from the perception sources, including the two behaviours that look like
    omissions and are not:

    * a detector failure yields NO objects rather than raising, so a model error surfaces downstream as
      an honest ``no_valid_grasp`` instead of a traceback in the middle of a pick;
    * a *single* segmentation failure skips that object and keeps the rest, because one bad mask in a
      cluttered bin should not discard the objects that segmented fine.
    """

    def __init__(self, detector: Any, segmenter: Any) -> None:
        self.detector = detector
        self.segmenter = segmenter
        # The seam's own log.
        self.logger = create_logger(
            "TwoStageBackend", log_file=PERCEPTION_BACKEND_LOG_FILE, log_dir=MODELS_LOG_DIR,
        )

    def perceive(self, image_bgr: Any, prompt: str) -> tuple[PerceivedObject, ...]:
        started = time.perf_counter()
        try:
            detections = self.detector.detect_all(image_bgr, prompt)
        except Exception:  # noqa: BLE001 - a model error emits no object (honest no_valid_grasp)
            # Returned as `()`, never raised, so this is the ONLY record the exception leaves --
            # hence the traceback rather than a one-line message.
            self.logger.exception("detector failed for prompt %r -- perceiving nothing", prompt)
            return ()

        objects: list[PerceivedObject] = []
        # Counted, not logged per detection: a cluttered multi-phrase prompt grounds a dozen-plus
        # boxes and one line each would drown the file. One aggregate line after the loop instead.
        seg_failures = 0
        first_seg_error = ""
        # Counted rather than `len(detections)`: the detector is duck-typed, and a backend that
        # yields its boxes would turn a length check into a crash on a path that must not have one.
        n_detections = 0
        for detection in detections:
            n_detections += 1
            try:
                segmentation = self.segmenter.segment_detection(image_bgr, detection)
            except Exception as exc:  # noqa: BLE001 - skip one segmentation failure, keep the rest
                seg_failures += 1
                if not first_seg_error:
                    first_seg_error = repr(exc)
                continue
            objects.append(PerceivedObject(detection=detection, segmentation=segmentation))

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if seg_failures:
            self.logger.warning(
                "segmentation dropped %d of %d detection(s) for prompt %r (first error: %s)",
                seg_failures, n_detections, prompt, first_seg_error,
            )
        self.logger.info(
            "perceived %d object(s) from %d detection(s) for prompt %r in %.1f ms",
            len(objects), n_detections, prompt, elapsed_ms,
        )
        return tuple(objects)
