"""Recursive mapping-merge used by the config loader to combine a base file with a profile overlay.
"""

from __future__ import annotations

from typing import Any


def _deep_merge(base: Any, overlay: Any) -> Any:
    """Recursively merge ``overlay`` into ``base`` (returns a new value)."""
    if overlay is None:
        return base
    if isinstance(base, dict) and isinstance(overlay, dict):
        out: dict[str, Any] = dict(base)
        for key, value in overlay.items():
            if key in out:
                out[key] = _deep_merge(out[key], value)
            else:
                out[key] = value
        return out
    return overlay
