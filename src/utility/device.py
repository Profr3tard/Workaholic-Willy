"""Device + dtype selection helpers for Workaholic-Willy.

    The selection can be forced via the ``WILLY_DEVICE``
    environment variable, which is useful in CI and the test suite:

    WILLY_DEVICE=cpu     pytest -q       # force CPU
    WILLY_DEVICE=cuda    python ...      # error if CUDA missing
    WILLY_DEVICE=auto    (default)       # auto-select (cuda > mps > cpu)

Helpers are provided for resolving inference dtypes
(``resolve_torch_dtype``) and for moving Hugging Face style input dicts
to the active device with pinned-memory + ``non_blocking`` copies
(``move_inputs_to_device``).
"""

from __future__ import annotations

import os
from typing import Any

import torch

_VALID_PREFS = {"auto", "cuda", "cpu", "mps"}


def _normalize_pref(pref: str | None) -> str:
    if pref is None:
        pref = os.environ.get("WILLY_DEVICE", "auto")
    pref = pref.strip().lower()
    if pref not in _VALID_PREFS:
        raise ValueError(
            f"Invalid device preference '{pref}'. "
            f"Expected one of {sorted(_VALID_PREFS)}."
        )
    return pref


def get_device(prefer: str | None = None) -> torch.device:
    """Return the active torch device.

    Selection order:
        1. The ``prefer`` argument, if provided.
        2. The ``WILLY_DEVICE`` environment variable.
        3. Auto-detect: cuda > mps > cpu.

    Raises:
        RuntimeError: if a specific backend is requested but unavailable.
    """
    pref = _normalize_pref(prefer)

    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "WILLY_DEVICE=cuda requested but CUDA is not available."
            )
        return torch.device("cuda")
    if pref == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "WILLY_DEVICE=mps requested but MPS is not available."
            )
        return torch.device("mps")
    if pref == "cpu":
        return torch.device("cpu")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def is_cuda(device: torch.device | None = None) -> bool:
    return (device or get_device()).type == "cuda"


def resolve_torch_dtype(
    name: str | None,
    device: torch.device,
) -> torch.dtype:
    """Map a string dtype (or ``None`` / ``"auto"``) to a torch dtype.

    On CUDA, ``None``/``auto`` resolves to ``float16``.  
    On CPU/MPS it resolves to ``float32``.
    """
    if name is None or name == "" or str(name).lower() == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32

    key = str(name).strip().lower()
    mapping: dict[str, torch.dtype] = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if key not in mapping:
        raise ValueError(f"Unknown torch dtype '{name}'.")

    dtype = mapping[key]
    if dtype != torch.float32 and device.type != "cuda":
        return torch.float32
    return dtype


def move_inputs_to_device(
    inputs: dict[str, Any],
    device: torch.device,
    non_blocking: bool = True,
) -> dict[str, Any]:
    """Move a dict of tensors / processor outputs to ``device``.

    Uses ``non_blocking=True`` plus pinned memory for CPU->CUDA copies.
    Non-tensor values are passed through unchanged.
    """
    out: dict[str, Any] = {}
    is_cuda_dev = device.type == "cuda"
    for k, v in inputs.items():
        if torch.is_tensor(v):
            if is_cuda_dev and v.device.type == "cpu" and not v.is_pinned():
                try:
                    v = v.pin_memory()
                except (RuntimeError, ValueError):
                    pass
            out[k] = v.to(device, non_blocking=non_blocking and is_cuda_dev)
        else:
            out[k] = v
    return out


__all__ = [
    "get_device",
    "is_cuda",
    "move_inputs_to_device",
    "resolve_torch_dtype",
]
