"""Device and dtype selection helpers.

Production runs on CUDA whenever a GPU is present, while tests and developer
machines fall back to CPU, or to MPS on Apple silicon. ``WILLY_DEVICE`` forces
the choice:

    WILLY_DEVICE=cpu     pytest -q       # force CPU
    WILLY_DEVICE=cuda    python ...      # error if CUDA is missing
    WILLY_DEVICE=auto    (default)       # auto-select, cuda then mps then cpu

:func:`resolve_torch_dtype` picks the inference dtype and
:func:`move_inputs_to_device` moves HuggingFace style input dicts to the active
device with pinned memory and non-blocking copies.

Neither selection is visible to the caller and both change how fast, and
sometimes whether, a model runs, so both reach ``logs/utility/device.log``: the
chosen device at info, the dtype at debug, and the two downgrades the caller did
not ask for, auto to CPU and half precision to fp32, at warning.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch

from src.utility.constants import DEVICE_LOG_FILE, utility_logger

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from logging import Logger

_VALID_PREFS = {"auto", "cuda", "cpu", "mps"}


def _log() -> Logger:
    """Logger for this module, built on first use. See :func:`utility_logger`."""
    return utility_logger("UtilityDevice", DEVICE_LOG_FILE)


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
        1. The ``prefer`` argument, if given.
        2. The ``WILLY_DEVICE`` environment variable.
        3. Auto-detection: cuda, then mps, then cpu.

    Raises:
        RuntimeError: if a named backend is requested but is not available.
    """
    pref = _normalize_pref(prefer)

    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "WILLY_DEVICE=cuda requested but CUDA is not available."
            )
        _log().info("device: pref=cuda (forced) -> cuda")
        return torch.device("cuda")
    if pref == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "WILLY_DEVICE=mps requested but MPS is not available."
            )
        _log().info("device: pref=mps (forced) -> mps")
        return torch.device("mps")
    if pref == "cpu":
        _log().info("device: pref=cpu (forced) -> cpu")
        return torch.device("cpu")

    # auto
    if torch.cuda.is_available():
        _log().info("device: pref=auto -> cuda")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        _log().info("device: pref=auto -> mps")
        return torch.device("mps")
    # Landing on CPU is neither an error nor a preference the operator
    # expressed, but a silent order-of-magnitude slowdown, hence warning.
    _log().warning(
        "device: pref=auto found neither CUDA nor MPS -> cpu; model inference will be slow"
    )
    return torch.device("cpu")


def is_cuda(device: torch.device | None = None) -> bool:
    return (device or get_device()).type == "cuda"


def resolve_torch_dtype(
    name: str | None,
    device: torch.device,
) -> torch.dtype:
    """Map a string dtype, or ``None`` or ``"auto"``, to a torch dtype.

    On CUDA, ``None`` and ``auto`` give ``float16``, the fastest broadly
    supported path on consumer GPUs. On CPU and MPS they give ``float32``, and a
    configured half-precision dtype is downgraded to it, because half-precision
    matmul on those backends is usually slower or unsupported. An unknown name
    raises ``ValueError``.
    """
    if name is None or name == "" or str(name).lower() == "auto":
        resolved = torch.float16 if device.type == "cuda" else torch.float32
        _log().debug("dtype: auto on %s -> %s", device.type, resolved)
        return resolved

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
        # The override is invisible to the caller, so it is logged: a model
        # configured fp16 that ran fp32 explains later accuracy and latency
        # differences.
        _log().warning(
            "dtype: configured %r downgraded to float32; the %s backend cannot use "
            "half precision",
            name,
            device.type,
        )
        return torch.float32
    _log().debug("dtype: configured %r -> %s", name, dtype)
    return dtype


def move_inputs_to_device(
    inputs: dict[str, Any],
    device: torch.device,
    non_blocking: bool = True,
) -> dict[str, Any]:
    """Move a dict of tensors or processor outputs to ``device``.

    Host tensors bound for CUDA are pinned and copied with ``non_blocking``; for
    every other device that flag is ignored. Non-tensor values pass through
    unchanged.
    """
    out: dict[str, Any] = {}
    is_cuda_dev = device.type == "cuda"
    pin_failures = 0
    for k, v in inputs.items():
        if torch.is_tensor(v):
            if is_cuda_dev and v.device.type == "cpu" and not v.is_pinned():
                try:
                    v = v.pin_memory()
                except (RuntimeError, ValueError):
                    pin_failures += 1
            out[k] = v.to(device, non_blocking=non_blocking and is_cuda_dev)
        else:
            out[k] = v
    # Counted rather than logged per tensor: this runs once per inference, and
    # what matters is that pinning failed at all, silently leaving the copy
    # blocking.
    if pin_failures:
        _log().debug(
            "move_inputs_to_device: %d of %d entries could not be pinned; "
            "host->device copies fall back to blocking",
            pin_failures,
            len(inputs),
        )
    return out


__all__ = [
    "get_device",
    "is_cuda",
    "move_inputs_to_device",
    "resolve_torch_dtype",
]
