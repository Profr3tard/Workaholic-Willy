"""Recursive mapping-merge shared by the config loader + editor (R7.2 dedup).

``backend.src.robot.grasping.replay.presets`` keeps its OWN ``_deep_merge`` deliberately — its
None/Mapping/deepcopy semantics are a genuinely different contract (it does NOT keep-base on a ``None``
overlay leaf), so it is intentionally not folded here.

Contract: a ``None`` overlay leaf KEEPS the base value (a partial overlay can omit-by-null without wiping
fields; locked by ``test_config_loader``). Profile overlays that need to *reset* a base field to ``None``
use the loader's explicit reset sentinel instead (see ``backend/config/loader.py``); it never reaches this
merge as ``None``.
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
