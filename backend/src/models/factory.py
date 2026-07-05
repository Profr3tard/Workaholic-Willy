"""Config-driven perception-backend factory.

Resolves ``models.detector`` / ``models.segmenter_backend`` to the concrete
wrapper. The torch-heavy wrappers are imported lazily inside each branch, so this
module stays importable without torch (e.g. on CI / macOS).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from backend.config.schema.app import ModelsConfig

__all__ = ["build_object_detector", "build_segmenter"]


def build_object_detector(models: ModelsConfig, *, debug_images: bool = False) -> Any:
    """Build the configured object detector (``groundingdino`` | ``rtdetr``).

    Both backends expose ``detect(image, prompt)`` / ``detect_all(image, prompt)``
    returning :class:`~backend.src.models.detection.types.Detection`; RT-DETR
    ignores an absent prompt and treats a present one as a class-name filter.
    """
    if models.detector == "rtdetr":
        if models.rtdetr is None:
            raise ValueError("models.detector='rtdetr' requires a models.rtdetr config block")
        from backend.src.models.detection.closed_set.detector import RtDetrObjectDetector

        return RtDetrObjectDetector(models.rtdetr, debug_images=debug_images)

    from backend.src.models.detection.zeroshot.detector import GroundingDinoObjectDetector

    return GroundingDinoObjectDetector(models.objectdetector, debug_images=debug_images)


def build_segmenter(models: ModelsConfig, *, save_debug: bool = False) -> Any:
    """Build the configured segmenter (``sam2`` | ``oneformer``).

    Both expose ``segment_detection(image_bgr, detection)`` returning
    :class:`~backend.src.models.segmentation.types.SegmentationResult`.
    """
    if models.segmenter_backend == "oneformer":
        if models.oneformer is None:
            raise ValueError("models.segmenter_backend='oneformer' requires a models.oneformer config block")
        from backend.src.models.segmentation.research.segmenter import OneFormerSegmenter

        return OneFormerSegmenter(models.oneformer, save_debug=save_debug)

    from backend.src.models.segmentation.realtime.segmenter import Sam2Segmenter

    return Sam2Segmenter(models.segmenter, save_debug=save_debug)
