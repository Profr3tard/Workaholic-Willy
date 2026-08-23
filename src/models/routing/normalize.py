"""The simple path's prompt normalizer.

GroundingDINO's documented input convention is lowercase phrases terminated by a period, and it is not
cosmetic: the text encoder was trained that way, and the caption is split on ``.`` to separate phrases.
A prompt that arrives as ``"A Red Cube"`` is a different caption from ``"a red cube."``.

Applies only to the SIMPLE route. The VLM route gets the operator's words untouched, because the
phrasing is exactly what it is being asked to reason about.
"""

from __future__ import annotations

import re

__all__ = ["normalize_simple_prompt"]

_WHITESPACE = re.compile(r"\s+")

#: Sentence-final marks that already terminate a phrase. A prompt ending in one is left alone rather
#: than acquiring a second terminator.
_TERMINATORS = (".", "?", "!")


def normalize_simple_prompt(prompt: str) -> str:
    """Lowercase, collapse whitespace, ensure a single trailing period. Idempotent."""
    collapsed = _WHITESPACE.sub(" ", prompt).strip().lower()
    if not collapsed:
        return ""
    if collapsed.endswith(_TERMINATORS):
        return collapsed
    return f"{collapsed}."
