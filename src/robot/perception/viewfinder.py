"""One colour image, on demand, from whatever perception source a cell was built with."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = ["COLOUR_SOURCE_KINDS", "ColourPeekable", "colour_source_kind", "peek_color_of"]

#: What a peekable source is a picture OF. This is not decoration: a synthetic scene rendered by the
#: rehearsal source and a frame off a D435 are both HxWx3 BGR arrays, and a console that labels the
#: first one "live from the camera" is making a claim about a room that has no camera in it. The
#: source says which; nothing downstream has to guess from the shape of the array.
#:
#:   "camera"     pixels that came off a physical device
#:   "synthetic"  pixels this process drew
#:   "unknown"    a source that can peek but does not say described neutrally, never as a camera
COLOUR_SOURCE_KINDS = ("camera", "synthetic", "unknown")


@runtime_checkable
class ColourPeekable(Protocol):
    """A perception source that can hand over a colour image without doing any perception."""

    #: One of :data:`COLOUR_SOURCE_KINDS`. Optional; a source that omits it is described as "unknown"
    #: rather than assumed to be a camera, because the assumption is the one that can mislead.
    colour_source_kind: str

    def peek_color(self) -> np.ndarray | None:
        """The scene as colour, right now: ``HxWx3`` BGR ``uint8``, or ``None`` if unavailable.

        ``None`` is a legitimate answer a camera that has not produced a frame yet, a source whose
        colour channel is disabled and it means "no picture", never "black picture".
        """
        ...


def colour_source_kind(source: Any) -> str:
    """What ``source``'s picture is OF -- ``"camera"``, ``"synthetic"`` or ``"unknown"``."""
    kind = str(getattr(source, "colour_source_kind", "") or "unknown")
    return kind if kind in COLOUR_SOURCE_KINDS else "unknown"


def peek_color_of(source: Any) -> np.ndarray | None:
    """The colour image a perception source can show, or ``None`` when it cannot show one."""
    peek = getattr(source, "peek_color", None)
    if peek is None or not callable(peek):
        return None
    try:
        image = peek()
    except Exception:  # noqa: BLE001 - a broken viewfinder must never take down the caller
        return None
    if image is None:
        return None
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or array.size == 0:
        return None
    return np.ascontiguousarray(array.astype(np.uint8, copy=False))
