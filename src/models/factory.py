"""Config-driven perception-backend factory.

Resolves ``models.detector`` / ``models.segmenter_backend`` to the concrete
wrapper.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.models.constants import MODELS_FACTORY_LOG_FILE, MODELS_LOG_DIR
from src.utility.log_cfg import create_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from config.schema.app import ModelsConfig

__all__ = ["build_object_detector", "build_perception", "build_segmenter"]

#: One line per assembled stack.
_LOG = create_logger(__name__, log_file=MODELS_FACTORY_LOG_FILE, log_dir=MODELS_LOG_DIR)


def build_object_detector(models: ModelsConfig, *, debug_images: bool = False) -> Any:
    """Build the configured object detector (``groundingdino`` | ``rtdetr``).

    Both backends expose ``detect(image, prompt)`` / ``detect_all(image, prompt)``
    returning :class:`~src.models.detection.types.Detection`; RT-DETR
    ignores an absent prompt and treats a present one as a class-name filter.
    """
    if models.detector == "rtdetr":
        if models.rtdetr is None:
            raise ValueError("models.detector='rtdetr' requires a models.rtdetr config block")
        from src.models.detection.closed_set.detector import RtDetrObjectDetector

        return RtDetrObjectDetector(models.rtdetr, debug_images=debug_images)

    from src.models.detection.zero_shot.detector import GroundingDinoObjectDetector

    return GroundingDinoObjectDetector(models.objectdetector, debug_images=debug_images)


def build_segmenter(models: ModelsConfig, *, save_debug: bool = False) -> Any:
    """Build the configured segmenter (``sam2`` | ``oneformer``).

    Both expose ``segment_detection(image_bgr, detection)`` returning
    :class:`~src.models.segmentation.types.SegmentationResult`.
    """
    if models.segmenter_backend == "oneformer":
        if models.oneformer is None:
            raise ValueError("models.segmenter_backend='oneformer' requires a models.oneformer config block")
        from src.models.segmentation.research.segmenter import OneFormerSegmenter

        return OneFormerSegmenter(models.oneformer, save_debug=save_debug)

    from src.models.segmentation.realtime.segmenter import Sam2Segmenter

    return Sam2Segmenter(models.segmenter, save_debug=save_debug)


def build_perception(models: ModelsConfig, *, debug_images: bool = False) -> Any:
    """Build the whole perception stack in one call, from ``models.pipeline``."""
    from src.models.perception_backend import TwoStageBackend

    pipeline = getattr(models, "pipeline", None)
    if pipeline is None:
        # Logged BEFORE the build, not after: the two constructors below load weights, and a stack that
        # dies during that load is exactly the one whose composition you want on record.
        _LOG.info(
            "building perception: two-stage %s + %s (legacy keys, no models.pipeline)",
            models.detector, models.segmenter_backend,
        )
        return TwoStageBackend(
            detector=build_object_detector(models, debug_images=debug_images),
            segmenter=build_segmenter(models, save_debug=debug_images),
        )

    # EVERY refusal is decided before a single weight is loaded.
    if pipeline.kind == "closed_set" and models.rtdetr is None:
        raise ValueError(
            "models.pipeline.kind='closed_set' needs a models.rtdetr config block -- the closed-set "
            "stack is RT-DETR, and nothing else here answers with a fixed class list."
        )
    if pipeline.kind == "zero_shot" and pipeline.zero_shot.backend == "vlm":
        # The schema guarantees router.enabled implies backend='vlm', so this is the only place a
        # routed stack can be built and the un-routed VLM stack is the same call without the split.
        if pipeline.router.enabled:
            return _build_routed_backend(models, debug_images=debug_images)
        return _build_vlm_backend(models, debug_images=debug_images)

    if pipeline.kind == "closed_set":
        from src.models.detection.closed_set.detector import RtDetrObjectDetector

        assert models.rtdetr is not None  # refused above
        _LOG.info("building perception: closed-set rtdetr + %s", pipeline.closed_set.segmenter)
        detector: Any = RtDetrObjectDetector(models.rtdetr, debug_images=debug_images)
        segmenter_backend = pipeline.closed_set.segmenter
    else:
        from src.models.detection.zero_shot.detector import GroundingDinoObjectDetector

        _LOG.info("building perception: zero-shot groundingdino + %s", pipeline.zero_shot.segmenter)
        detector = GroundingDinoObjectDetector(models.objectdetector, debug_images=debug_images)
        segmenter_backend = pipeline.zero_shot.segmenter

    return TwoStageBackend(
        detector=detector,
        segmenter=_build_named_segmenter(models, segmenter_backend, save_debug=debug_images),
    )


def _build_routed_backend(models: ModelsConfig, *, debug_images: bool) -> Any:
    """The full pipeline: route each prompt, then run the route it chose.

    Both routes are **factories**, so neither model is loaded until a prompt actually selects it. They
    deliberately share ONE segmenter instance.
    """
    from src.models.detection.zero_shot.detector import GroundingDinoObjectDetector
    from src.models.perception_backend import TwoStageBackend
    from src.models.routed_backend import RoutedPerceptionBackend

    assert models.pipeline is not None  # only reached from build_perception's pipeline branch
    if models.objectdetector is None:
        raise ValueError(
            "models.pipeline.router.enabled=true routes simple prompts to the phrase grounder, so a "
            "models.objectdetector block is required. Either add one, or set router.enabled: false."
        )

    # Both routes are lazy, so this line is the only place the routed SHAPE appears, the per-route
    # "building the ... route (first use)" lines arrive later, in the routing log, and possibly never.
    _LOG.info(
        "building perception: routed (simple=groundingdino | vlm=%s), shared segmenter=%s, "
        "both routes lazy",
        models.pipeline.zero_shot.vlm.model_id if models.pipeline.zero_shot.vlm else "unconfigured",
        models.pipeline.zero_shot.segmenter,
    )
    # Built once, up front, and shared: it is needed by whichever route runs first, and both need it.
    # One segmenter choice covers both routes, masks should not change shape depending on which
    # grounding model happened to answer, or a pick rate would move for reasons nobody could see.
    segmenter = _build_named_segmenter(
        models, models.pipeline.zero_shot.segmenter, save_debug=debug_images,
    )

    def _simple() -> Any:
        return TwoStageBackend(
            detector=GroundingDinoObjectDetector(models.objectdetector, debug_images=debug_images),
            segmenter=segmenter,
        )

    def _vlm() -> Any:
        return _build_vlm_backend(models, debug_images=debug_images, segmenter=segmenter)

    return RoutedPerceptionBackend(simple_factory=_simple, vlm_factory=_vlm)


def _build_vlm_backend(models: ModelsConfig, *, debug_images: bool, segmenter: Any = None) -> Any:
    """The VLM grounding route: Qwen as the detector, SAM2 for masks, guarded by ``on_unavailable``.

    The VLM is wired in as a **detector**, so the same :class:`TwoStageBackend` that runs the phrase
    route composes it with the same segmenter, there is no second pipeline to keep in step.

    Two things are deliberately lazy. The grounder loads weights on first use unless ``preload`` says
    otherwise, and the degrade fallback is a factory rather than an instance, so a cell that never
    degrades never pays for GroundingDINO..
    """
    from src.models.perception_backend import TwoStageBackend
    from src.models.vlm import GuardedVlmBackend, Qwen3VLGrounder

    assert models.pipeline is not None  # only reached from build_perception's pipeline branch
    vlm_cfg = models.pipeline.zero_shot.vlm
    if vlm_cfg is None:
        raise ValueError(
            "models.pipeline.zero_shot.backend='vlm' needs a models.pipeline.zero_shot.vlm block "
            "naming the checkpoint -- refusing to guess which model was meant."
        )
    degrade = vlm_cfg.on_unavailable == "degrade"
    if degrade and models.objectdetector is None:
        # Caught here rather than at the moment of degradation, which would be mid-pick and far too
        # late to do anything about.
        raise ValueError(
            "on_unavailable='degrade' falls back to the phrase detector, so a models.objectdetector "
            "block is required. Either add one, or use on_unavailable='refuse'."
        )

    # `on_unavailable` is the config knob that decides whether a missing VLM refuses the prompt or
    # silently grounds it with the wrong model, so it belongs on the record next to the checkpoint.
    _LOG.info(
        "building perception: vlm %s (on_unavailable=%s, preload=%s) + %s",
        vlm_cfg.model_id, vlm_cfg.on_unavailable, vlm_cfg.preload,
        models.pipeline.zero_shot.segmenter,
    )
    grounder = Qwen3VLGrounder(
        model_id=vlm_cfg.model_id,
        model_path=vlm_cfg.model_path,
        local=vlm_cfg.local,
        preload=vlm_cfg.preload,
    )
    if segmenter is None:
        segmenter = _build_named_segmenter(
            models, models.pipeline.zero_shot.segmenter, save_debug=debug_images,
        )

    def _fallback() -> Any:
        from src.models.detection.zero_shot.detector import GroundingDinoObjectDetector

        return TwoStageBackend(
            detector=GroundingDinoObjectDetector(models.objectdetector, debug_images=debug_images),
            segmenter=segmenter,
        )

    return GuardedVlmBackend(
        vlm=TwoStageBackend(detector=grounder, segmenter=segmenter),
        model_id=vlm_cfg.model_id,
        degrade=degrade,
        fallback_factory=_fallback if degrade else None,
    )


def _build_named_segmenter(models: ModelsConfig, backend: str, *, save_debug: bool) -> Any:
    """Build ``backend`` by name."""
    if backend == "oneformer":
        if models.oneformer is None:
            raise ValueError("segmenter 'oneformer' requires a models.oneformer config block")
        from src.models.segmentation.research.segmenter import OneFormerSegmenter

        return OneFormerSegmenter(models.oneformer, save_debug=save_debug)

    from src.models.segmentation.realtime.segmenter import Sam2Segmenter

    return Sam2Segmenter(models.segmenter, save_debug=save_debug)
