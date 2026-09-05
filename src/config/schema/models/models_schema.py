"""Schemas for ML / CV model configurations (vision + speech)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .._base import StrictModel


class InferenceOptimization(StrictModel):
    """Optional inference-time optimizations for torch-based models.

    Defaults are conservative so behaviour matches the pre-optimization
    code exactly until a config explicitly opts in. Production configs
    should typically enable ``torch_dtype="auto"`` and
    ``attn_implementation="sdpa"`` on CUDA hosts.

    Fields:
        torch_dtype:
            ``"auto"`` (fp16 on CUDA, fp32 elsewhere), ``"float16"``,
            ``"bfloat16"``, ``"float32"`` or ``None`` (= fp32, legacy).
        attn_implementation:
            Forwarded to ``from_pretrained``. ``"sdpa"`` is a safe fast
            default on CUDA. ``None`` keeps the HF default.
        channels_last:
            Use channels-last memory format for vision models. Faster on
            Ampere+ GPUs, no-op on CPU.
        compile:
            Wrap the model with ``torch.compile``. Significant speed-up
            on stable input shapes but adds warm-up time and can fail on
            some HF architectures, hence opt-in.
        compile_mode:
            Mode forwarded to ``torch.compile``.
    """

    torch_dtype: str | None = None
    attn_implementation: str | None = None
    channels_last: bool = False
    compile: bool = False
    compile_mode: str = "reduce-overhead"






class ObjectDetectorConfig(StrictModel):
    """Object-detection model configuration."""

    model_path: str
    model_id: str | None = None
    threshold: float
    local: bool
    optim: InferenceOptimization = Field(default_factory=InferenceOptimization)


class SegmenterConfig(StrictModel):
    """Image segmentation (e.g. Sam2) configuration."""

    model_path: str
    model_id: str | None = None
    local: bool
    optim: InferenceOptimization = Field(default_factory=InferenceOptimization)


class OneFormerConfig(StrictModel):
    """OneFormer universal-segmentation configuration (research/high-accuracy backend)."""

    model_path: str
    model_id: str | None = None
    local: bool
    #: Narrowed to ``instance`` because that is the only mode the wrapper implements: it always calls
    #: ``post_process_instance_segmentation``, so a config asking for ``semantic`` or ``panoptic`` used
    #: to validate cleanly and then silently run instance segmentation anyway. Advertising a capability
    #: the code does not have is worse than not offering it; widen this the day the post-processing
    #: exists, not before.
    task: Literal["instance"] = "instance"
    optim: InferenceOptimization = Field(default_factory=InferenceOptimization)


class SpeechToTextConfig(StrictModel):
    """Speech-to-text (Whisper) configuration."""

    model_id: str
    model_path: str
    samplerate: int
    # Read wholesale by `WhisperSpeechToText.__init__`, which takes the block, not a path:
    # the reason the dead-key trace first mis-read these four as unread.
    blocksize: int
    channels: int
    dtype: str
    chunk_duration: int
    language: str
    task: str
    local: bool
    optim: InferenceOptimization = Field(default_factory=InferenceOptimization)


# ---------------------------------------------------------------------------
# Hand + gesture (MediaPipe): optional, standalone, not part of the grasp pipeline
# ---------------------------------------------------------------------------


class HandDetectConfig(StrictModel):
    """MediaPipe hand-landmark detection: where the palm centre is, in pixels.

    Every field below is passed to MediaPipe by
    :class:`~src.models.handdetection.palm_detector.PalmDetector`; none is stored and ignored.

    The ``.task`` bundle is not in this repository. It is an operator download, and a missing file
    raises with the URL rather than failing inside MediaPipe's graph.
    """

    #: Path to the MediaPipe hand-landmark ``.task`` bundle. Relative paths resolve against the
    #: process working directory. Download:
    #: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
    model_path: str = "assets/models/mediapipe/hand_landmarker.task"

    #: Upper bound on simultaneously tracked hands -> ``num_hands``. Note that the 3-D
    #: :class:`~src.models.handdetection.hand_finder.HandFinder` refuses a frame with more
    #: than one hand regardless: two hands in a workspace is a reason to stop, not to choose one.
    #: Keeping this at 2 is what lets it see the second hand in order to refuse.
    max_hands: int = Field(default=2, ge=1, le=4)

    #: Minimum hand-detection confidence -> ``min_hand_detection_confidence``.
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    #: Minimum tracking confidence between frames -> ``min_tracking_confidence``. Only meaningful
    #: because the detector runs in video mode; in image mode there is no previous frame and this
    #: would be inert.
    tracking_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    #: Minimum hand-presence confidence -> ``min_hand_presence_confidence``. Distinct from
    #: ``threshold``: detection asks "is there a hand here", presence asks "is the hand I was
    #: tracking still in frame".
    presence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    #: Radius, in pixels, of the disc around the palm centre whose median depth becomes the hand's
    #: distance. Ours, not MediaPipe's. Too small and a depth dropout on skin loses the hand; too
    #: large and the disc starts averaging in whatever is behind the hand.
    palm_patch_radius_px: int = Field(default=15, ge=1, le=200)

    #: How many valid depth pixels that disc must contain before its median is believed. A median
    #: over three surviving pixels is a number, not a measurement.
    min_depth_samples: int = Field(default=20, ge=1, le=100_000)


class GestureDetectConfig(StrictModel):
    """MediaPipe canned gesture recognition, narrowed to thumbs-up / thumbs-down.

    The canned bundle embeds a hand landmarker, so this model returns landmarks as well as
    gestures; a cell that wants both should configure this one alone rather than running two
    models over the same frame.

    The classifier knows seven shapes;
    :class:`~src.models.handdetection.gestures.ThumbGestureRecognizer` maps two of them and
    reports the rest as ``OTHER``, keeping the raw label.
    """

    #: Path to the MediaPipe gesture-recognizer ``.task`` bundle. Download:
    #: https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task
    model_path: str = "assets/models/mediapipe/gesture_recognizer.task"

    #: -> ``num_hands``.
    max_hands: int = Field(default=2, ge=1, le=4)

    #: -> ``min_hand_detection_confidence``.
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    #: -> ``min_tracking_confidence``.
    tracking_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    #: -> ``min_hand_presence_confidence``.
    presence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    #: The score a thumbs-up must reach to be reported as one. Passed to MediaPipe as the canned
    #: classifier's ``score_threshold`` and re-checked on our side, so a below-threshold thumbs-up
    #: becomes ``NONE`` rather than a low-confidence confirmation. Raise it where a false
    #: confirmation is expensive.
    min_gesture_confidence: float = Field(default=0.5, ge=0.0, le=1.0)

# ---------------------------------------------------------------------------
# Pipeline presets: choosing a perception stack in one line instead of assembling it
# ---------------------------------------------------------------------------


class VlmConfig(StrictModel):
    """The vision-language model that grounds a complex prompt.

    **VRAM is the constraint that decides this block**, and the numbers belong in the README next to
    the choice, not in someone's head. The model runs alongside sam2 and (in simulation) Isaac's own
    renderer on the same card.
    """

    #: A released Qwen3-VL-Instruct checkpoint. The paper's Qwen3-VL-Seg (arXiv 2605.07141) publishes
    #: neither weights nor code, so this stack grounds with the released instruct model and cuts masks
    #: with sam2 from the same bbox_2d JSON the paper's own decoder consumes. A native-mask
    #: checkpoint, if one is ever released, becomes a different backend behind the same seam.
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct-FP8"
    model_path: str | None = None
    local: bool = False
    #: Load at cell build (predictable latency, VRAM held from the start) rather than on the first
    #: complex prompt (nothing paid until something needs it, one-off load pause mid-session).
    preload: bool = False
    #: What to do when the VLM cannot be loaded on this cell: weights missing, dependency absent, or
    #: not enough VRAM.
    #:
    #: ``refuse``  the pick is rejected with a typed reason. The default, because the failure this
    #:             whole route exists to prevent is exactly what falling back would produce:
    #:             GroundingDINO does not fail on a complex prompt, it confidently grounds the wrong
    #:             object, and a wrong grasp is worse than a refused one.
    #: ``degrade`` fall back to the simple route, with a warning stamped into the event stream and the
    #:             grasp record so the weaker grounding is visible afterwards.
    on_unavailable: Literal["refuse", "degrade"] = "refuse"


class PromptRouterConfig(StrictModel):
    """Deciding which route a prompt takes.

    Rule-based and deterministic: a prompt's complexity is read off its structure (attribute count,
    relative clauses, negation, quantifiers) and its language. A cheap-first cascade is deliberately
    Not the design: it only works when the cheap stage fails loudly, and this one does not.
    """

    enabled: bool = True
    #: A non-English prompt routes to the VLM regardless of complexity. GroundingDINO is effectively
    #: English-only, and Whisper's German->English translation covers the spoken path only: a typed
    #: German prompt reaches the detector untranslated today.
    route_non_english_to_vlm: bool = True


class ZeroShotPipelineConfig(StrictModel):
    """Open-vocabulary perception: any prompt, no fixed class list."""

    #: ``grounded_sam`` = GroundingDINO + a segmenter. ``vlm`` = the VLM grounds, sam2 cuts.
    backend: Literal["grounded_sam", "vlm"] = "grounded_sam"
    #: Only consulted for ``grounded_sam``; the VLM route always cuts with sam2, because there is no
    #: second mask source to choose between yet. Offering a knob with one legal value would be
    #: pretending otherwise.
    segmenter: Literal["sam2", "oneformer"] = "sam2"
    vlm: VlmConfig = Field(default_factory=VlmConfig)


class ClosedSetPipelineConfig(StrictModel):
    """Fixed-vocabulary perception: RT-DETR's trained classes, no free text."""

    segmenter: Literal["sam2", "oneformer"] = "sam2"


class PipelineConfig(StrictModel):
    """A perception stack chosen in one line, instead of assembled from parts.

    Leaving this block out keeps the legacy ``models.detector`` / ``models.segmenter_backend`` keys in
    force, byte-identically, so an existing cell is unaffected until someone opts in. Those keys
    also remain the do-it-yourself path for anyone who wants a combination the presets do not bless.

    The presets exist because the individual keys are independently settable and nothing checked them
    against each other: every detector x segmenter combination builds today, including ones where the
    prompt means something different to each half.
    """

    kind: Literal["zero_shot", "closed_set"] = "zero_shot"
    zero_shot: ZeroShotPipelineConfig = Field(default_factory=ZeroShotPipelineConfig)
    closed_set: ClosedSetPipelineConfig = Field(default_factory=ClosedSetPipelineConfig)
    router: PromptRouterConfig = Field(default_factory=PromptRouterConfig)

    @model_validator(mode="after")
    def _refuse_combinations_that_cannot_mean_anything(self) -> "PipelineConfig":
        """Reject a stack whose halves disagree, naming what IS legal.

        Fail-closed rather than warn-and-continue: the failure mode this pipeline work exists to remove
        is a stack that runs confidently and grounds the wrong thing, and a warning in a log has never
        stopped a grasp.
        """
        # NOTE (2026-08-11): the VLM route used to pin segmenter='sam2' here. That was policy, not a
        # technical limit, and it was wrong: OneFormer implements the same box-prompted contract
        # (`segment_detection` -> `_pick_segment_in_box`), so it works with any box source, the
        # phrase grounder and the VLM alike. Both routes now honour the choice.
        #
        # Which is better is unmeasured. Sam2 is prompted per box and cuts one object; OneFormer
        # segments the whole image with its own trained vocabulary and the box selects among the
        # segments. Those fail differently, and this repo has no comparison yet, so this is a knob,
        # not a recommendation.
        if self.kind == "closed_set" and self.router.enabled:
            # A closed-set stack has no free-text route to send anything to; routing a prompt there
            # would produce a decision nothing can act on.
            #
            # Only an explicit `router.enabled: true` is an error. Left unset it is just the field's
            # default, and failing someone for a default they never wrote, when the correct value is
            # unambiguous, is a validator being pedantic rather than protective.
            if "router" in self.model_fields_set and "enabled" in self.router.model_fields_set:
                raise ValueError(
                    "models.pipeline: kind='closed_set' cannot use the prompt router. Routing "
                    "decides between open-vocabulary backends, and a closed-set detector answers only "
                    "with its trained class list. Set router.enabled: false, or kind: 'zero_shot'."
                )
            object.__setattr__(self, "router", self.router.model_copy(update={"enabled": False}))
        if self.kind == "zero_shot" and self.router.enabled and self.zero_shot.backend != "vlm":
            # Routing exists to send hard prompts somewhere better. With only the phrase grounder
            # configured there is nowhere better to send them, so every VLM decision the router made
            # would either be ignored or would fail at pick time, after the operator pressed go.
            #
            # Same rule as the closed_set case above: only an explicit `router.enabled: true` is an
            # error. The field defaults to true because routing is the point of this block, and the
            # Default backend is the phrase grounder, so a bare `pipeline: {}` would otherwise be
            # rejected for a combination nobody wrote.
            if "router" in self.model_fields_set and "enabled" in self.router.model_fields_set:
                raise ValueError(
                    f"models.pipeline: router.enabled=true needs zero_shot.backend='vlm'. The "
                    f"router chooses BETWEEN the phrase grounder and the VLM, and backend="
                    f"{self.zero_shot.backend!r} configures no VLM for it to choose. Set backend: "
                    f"'vlm' (the phrase grounder stays the simple route), or router.enabled: false."
                )
            object.__setattr__(self, "router", self.router.model_copy(update={"enabled": False}))
        return self
