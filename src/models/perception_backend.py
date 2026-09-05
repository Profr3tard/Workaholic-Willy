"""One seam between "a prompt" and "masks", so a pipeline is a choice rather than an assembly.

The two-stage chain, ``detect_all(bgr, prompt)`` followed by ``segment_detection(bgr, det)`` per
detection, lives here instead of in each caller. A :class:`PerceptionBackend` is that loop, named
and owned:

    backend = build_perception(models_cfg)
    objects = backend.perceive(image_bgr, "a green cube")

``perceive`` returns pairs, ``(detection, segmentation)``, because callers consume both: the real
cell fills an incomplete mask back out to the detection box when SAM2 drops one end of a long object
(a measured ~22 mm centroid shift, an off-centre grasp, no lift). A backend whose model produces
masks natively, with no box stage, synthesises the detection from the mask's own bounding box.

This module does not import torch. It is the contract, not the models; the wrappers stay behind the
lazy imports in :mod:`src.models.factory`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from src.models.constants import MODELS_LOG_DIR, PERCEPTION_BACKEND_LOG_FILE
from src.utility.log_cfg import create_logger

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.models.detection.types import Detection
    from src.models.segmentation.types import SegmentationResult

__all__ = ["PerceivedObject", "PerceptionBackend", "TwoStageBackend"]


@dataclass(frozen=True, slots=True)
class PerceivedObject:
    """One grounded object: the detection box and the mask cut for it.

    Both halves travel together because both are consumed. The grasp calculator reads the mask; a
    caller compares the mask against the box to decide whether the mask is trustworthy.
    """

    detection: "Detection"
    segmentation: "SegmentationResult"


@runtime_checkable
class PerceptionBackend(Protocol):
    """Prompt in, grounded objects out. The whole perception contract.

    :func:`src.models.factory.build_perception` constructs an implementation from config, and the
    implementation holds whatever models it needs. Loaded weights are its only state; ``perceive``
    is a pure function of its arguments.
    """

    def perceive(self, image_bgr: Any, prompt: str) -> tuple[PerceivedObject, ...]:
        """Ground ``prompt`` in ``image_bgr``. Returns ``()`` when nothing matched.

        An empty result is an answer, not an error: the pick loop reports ``no_perception`` and
        moves on, the honest outcome when the scene does not contain what was asked for.
        """
        ...


class TwoStageBackend:
    """Detector then segmenter, the two-stage chain.

    Two failures are swallowed rather than raised:

    * a detector failure yields no objects, so a model error surfaces downstream as
      ``no_valid_grasp`` instead of a traceback in the middle of a pick;
    * a single segmentation failure skips that object and keeps the rest, so one bad mask in a
      cluttered bin does not discard the objects that segmented fine.

    Detections are not de-duplicated. A multi-phrase prompt grounds neighbour clutter on purpose:
    the dense sampler needs to know what else is in the bin.
    """

    def __init__(self, detector: Any, segmenter: Any) -> None:
        self.detector = detector
        self.segmenter = segmenter
        # The seam's own log. Both swallow-and-continue paths below are silent downstream: a pick
        # that dropped every mask and a pick over an empty bin both arrive as `no_perception`, and
        # these lines are the only place that difference is recorded.
        self.logger = create_logger(
            "TwoStageBackend", log_file=PERCEPTION_BACKEND_LOG_FILE, log_dir=MODELS_LOG_DIR,
        )

    def perceive(self, image_bgr: Any, prompt: str) -> tuple[PerceivedObject, ...]:
        started = time.perf_counter()
        try:
            detections = self.detector.detect_all(image_bgr, prompt)
        except Exception:  # noqa: BLE001 (a model error emits no object (honest no_valid_grasp))
            # The failure returns `()` and never raises, so this line is the only record of the
            # exception; hence a full traceback rather than a one-line message.
            self.logger.exception("detector failed for prompt %r, perceiving nothing", prompt)
            return ()

        objects: list[PerceivedObject] = []
        # Counted, not logged per detection: a cluttered multi-phrase prompt grounds a dozen-plus
        # boxes, and one line each would drown the file. One aggregate line follows the loop.
        seg_failures = 0
        first_seg_error = ""
        # Counted rather than `len(detections)`: the detector is duck-typed, and a backend that
        # yields its boxes has no length to take on a path that must not crash.
        n_detections = 0
        for detection in detections:
            n_detections += 1
            try:
                segmentation = self.segmenter.segment_detection(image_bgr, detection)
            except Exception as exc:  # noqa: BLE001 (skip one segmentation failure, keep the rest)
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
