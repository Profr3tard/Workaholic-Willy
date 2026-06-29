"""Recursive mapping-merge shared by the config loader + editor.

``backend.src.robot.grasping.replay.presets`` keeps its OWN ``_deep_merge`` deliberately — its
None/Mapping/deepcopy semantics are a genuinely different contract (it does NOT keep-base on a ``None``
overlay leaf), so it is intentionally not folded here.
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
