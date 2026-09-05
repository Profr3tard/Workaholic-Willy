"""Shared inference-time helpers for torch-based model wrappers.

The CUDA-versus-CPU handling lives here, so a wrapper carries only the
logic of its own model.

``build_load_kwargs`` produces the kwargs for ``from_pretrained``, adding
``torch_dtype`` and ``attn_implementation`` as the optim config asks.
``finalize_model`` applies the post-load steps, ``.eval()``,
channels-last and optional ``torch.compile``, and returns the possibly
wrapped model. ``autocast_ctx`` is a no-op context on CPU and
``torch.autocast`` on CUDA.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

import torch

from src.config.schema.models.models_schema import InferenceOptimization
from src.utility.device import resolve_torch_dtype

_log = logging.getLogger(__name__)


def build_load_kwargs(
    optim: InferenceOptimization | None,
    device: torch.device,
    base_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return ``from_pretrained`` kwargs honoring the optim config."""
    kwargs: dict[str, Any] = dict(base_kwargs or {})
    if optim is None:
        return kwargs

    # Request a non-default torch_dtype only when the config names one
    # ("auto" or a concrete dtype). Left unset, the HF default of fp32
    # weights applies, which is what a config with no `optim` section
    # gets.
    if optim.torch_dtype is not None:
        resolved = resolve_torch_dtype(optim.torch_dtype, device)
        if resolved == torch.float16:
            # fp16 drops small-object detection recall. The default torch_dtype=None keeps HF fp32
            # weights, so this fires only for a config that asks for "auto" or fp16.
            _log.warning(
                "Loading model under float16 (torch_dtype=%r on %s): fp16 can drop small-object recall; "
                "set torch_dtype=null/fp32 for recall-sensitive detectors (GroundingDINO/SAM2).",
                optim.torch_dtype, device.type,
            )
        kwargs["torch_dtype"] = resolved

    if optim.attn_implementation:
        kwargs["attn_implementation"] = optim.attn_implementation

    return kwargs


def finalize_model(
    model: torch.nn.Module,
    device: torch.device,
    optim: InferenceOptimization | None,
    vision: bool = False,
) -> torch.nn.Module:
    """Apply post-load optimizations and return the (possibly wrapped) model."""
    model.eval()

    if optim is None:
        return model

    if vision and optim.channels_last and device.type == "cuda":
        try:
            model = model.to(memory_format=torch.channels_last)  # type: ignore[call-overload]  # torch stub lacks the memory_format overload
        except (RuntimeError, ValueError) as exc:
            # Not every submodule supports channels_last; skip it and log.
            _log.warning("channels_last conversion skipped: %s", exc)

    if optim.compile and device.type == "cuda":
        try:
            model = torch.compile(model, mode=optim.compile_mode)  # type: ignore[assignment]  # torch.compile returns an OptimizedModule (callable) reassigned to the Module-typed var
        except Exception as exc:
            # Compile is best-effort: on a failure such as dynamic shapes
            # or an unsupported op, stay eager and record the reason.
            _log.warning("torch.compile failed (mode=%s): %s", optim.compile_mode, exc)

    return model


def autocast_ctx(device: torch.device, dtype: torch.dtype | None = None):
    """Return an autocast context, or a no-op context on non-CUDA devices."""
    if device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


__all__ = [
    "autocast_ctx",
    "build_load_kwargs",
    "finalize_model",
]
