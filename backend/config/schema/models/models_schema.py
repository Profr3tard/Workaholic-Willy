"""Schemas for ML / CV model configurations (vision + speech)."""

from __future__ import annotations

from pydantic import Field

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
            ``"bfloat16"``, ``"float32"`` or ``None``.
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


class SimplifierConfig(StrictModel):
    """Text-simplification (seq2seq) model configuration."""

    model_path: str
    model_id: str | None = None
    max_length: int
    num_beams: int
    no_repeat_ngram_size: int
    early_stopping: bool
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
