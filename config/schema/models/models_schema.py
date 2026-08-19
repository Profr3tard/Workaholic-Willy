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


class HandDetectConfig(StrictModel):
    """MediaPipe hand-landmark detection configuration."""

    model_path: str
    max_hands: int
    threshold: float
    tracking_threshold: float


class GestureDetectConfig(StrictModel):
    """MediaPipe gesture-recognition configuration."""

    model_path: str
    max_hands: int
    threshold: float
    tracking_threshold: float


class ObjectDetectorConfig(StrictModel):
    """Object-detection model configuration."""

    model_path: str
    model_id: str | None = None
    threshold: float
    local: bool
    optim: InferenceOptimization = Field(default_factory=InferenceOptimization)


class SegmenterConfig(StrictModel):
    """Image segmentation (e.g. SAM2) configuration."""

    model_path: str
    model_id: str | None = None
    local: bool
    optim: InferenceOptimization = Field(default_factory=InferenceOptimization)


class OneFormerConfig(StrictModel):
    """OneFormer universal-segmentation configuration (research/high-accuracy backend)."""

    model_path: str
    model_id: str | None = None
    local: bool
    task: Literal["instance"] = "instance"
    optim: InferenceOptimization = Field(default_factory=InferenceOptimization)


class SpeechToTextConfig(StrictModel):
    """Speech-to-text (Whisper) configuration."""

    model_id: str
    model_path: str
    samplerate: int
    blocksize: int
    channels: int
    dtype: str
    chunk_duration: int
    language: str
    task: str
    local: bool
    optim: InferenceOptimization = Field(default_factory=InferenceOptimization)


# ---------------------------------------------------------------------------
# Pipeline presets
# ---------------------------------------------------------------------------


class VlmConfig(StrictModel):
    """The vision-language model that grounds a complex prompt."""

    model_id: str = "Qwen/Qwen3-VL-4B-Instruct-FP8"
    model_path: str | None = None
    local: bool = False
    preload: bool = False
    on_unavailable: Literal["refuse", "degrade"] = "refuse"
    optim: InferenceOptimization = Field(default_factory=InferenceOptimization)


class PromptRouterConfig(StrictModel):
    """Deciding which route a prompt takes."""

    enabled: bool = True
    route_non_english_to_vlm: bool = True


class ZeroShotPipelineConfig(StrictModel):
    """Open-vocabulary perception: any prompt, no fixed class list."""

    backend: Literal["grounded_sam", "vlm"] = "grounded_sam"
    segmenter: Literal["sam2", "oneformer"] = "sam2"
    vlm: VlmConfig = Field(default_factory=VlmConfig)


class ClosedSetPipelineConfig(StrictModel):
    """Fixed-vocabulary perception: RT-DETR's trained classes, no free text."""

    segmenter: Literal["sam2", "oneformer"] = "sam2"


class PipelineConfig(StrictModel):
    """A perception stack chosen in one line, instead of assembled from parts.

    Leaving this block out keeps the legacy ``models.detector`` / ``models.segmenter_backend`` keys in
    force, byte-identically so an existing cell is unaffected until someone opts in. Those keys also
    remain the do-it-yourself path for anyone who wants a combination the presets do not bless.

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
        
        if self.kind == "closed_set" and self.router.enabled:
            if "router" in self.model_fields_set and "enabled" in self.router.model_fields_set:
                raise ValueError(
                    "models.pipeline: kind='closed_set' cannot use the prompt router routing "
                    "decides between open-vocabulary backends, and a closed-set detector answers only "
                    "with its trained class list. Set router.enabled: false, or kind: 'zero_shot'."
                )
            object.__setattr__(self, "router", self.router.model_copy(update={"enabled": False}))
        if self.kind == "zero_shot" and self.router.enabled and self.zero_shot.backend != "vlm":
            if "router" in self.model_fields_set and "enabled" in self.router.model_fields_set:
                raise ValueError(
                    f"models.pipeline: router.enabled=true needs zero_shot.backend='vlm' the router "
                    f"chooses BETWEEN the phrase grounder and the VLM, and backend="
                    f"{self.zero_shot.backend!r} configures no VLM for it to choose. Set backend: "
                    f"'vlm' (the phrase grounder stays the simple route), or router.enabled: false."
                )
            object.__setattr__(self, "router", self.router.model_copy(update={"enabled": False}))
        return self
