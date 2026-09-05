"""The MediaPipe hand-landmark layout, and the palm centre computed from it.

Pure geometry: no MediaPipe import, no camera, no config. That is deliberate. The palm centre is the
one piece of this package that can be exercised exhaustively without a model file or a hand, and
keeping it importable on its own is what makes that possible.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Final, Sequence

import numpy as np

__all__ = [
    "HandLandmark",
    "PALM_LANDMARKS",
    "calculate_palm_center",
    "draw_hand_landmarks",
]


class HandLandmark(IntEnum):
    """MediaPipe's 21-point hand-landmark layout, by name.

    `IntEnum` rather than `StrEnum` because these are indices into the result list: the value has
    to stay usable as a subscript. Naming them removes the class of bug where `landmarks[17]` is
    read as "some finger" by the next person and quietly changed.
    """

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


#: The landmarks averaged into the palm centre: the wrist, the four finger knuckles, and the base of
#: the thumb.
#:
#: The wrist and the four MCP joints bound the palm; `THUMB_CMC` is the base of the thumb,
#: anatomically the thenar side of the palm, and including it pulls the centroid a little toward the
#: thumb. The commoner published definition uses the five without it. Both are defensible and the
#: difference is small; this set is kept for continuity with the positions this package already
#: reports, and changing it would move every 3-D hand position in a way nothing here has measured
#: against a real hand. Do not change it without a measurement, and if you do, say which one.
PALM_LANDMARKS: Final[tuple[HandLandmark, ...]] = (
    HandLandmark.WRIST,
    HandLandmark.INDEX_MCP,
    HandLandmark.MIDDLE_MCP,
    HandLandmark.RING_MCP,
    HandLandmark.PINKY_MCP,
    HandLandmark.THUMB_CMC,
)

#: How many landmarks MediaPipe returns per hand. A result with any other count is not a hand this
#: package can interpret, and is rejected rather than indexed into.
LANDMARK_COUNT: Final[int] = 21


def calculate_palm_center(
    landmarks: Sequence[tuple[int, int]],
) -> tuple[float, float]:
    """Return the mean `(x, y)` of the six palm landmarks, in the units the landmarks are in.

    Raises `ValueError` on anything that is not a full 21-point hand. The alternative is an
    `IndexError` from deep inside the averaging, several frames after the real problem.
    """
    if len(landmarks) != LANDMARK_COUNT:
        raise ValueError(
            f"calculate_palm_center expects {LANDMARK_COUNT} landmarks (MediaPipe layout), "
            f"got {len(landmarks)}"
        )
    points = [landmarks[int(index)] for index in PALM_LANDMARKS]
    x = sum(float(p[0]) for p in points) / len(points)
    y = sum(float(p[1]) for p in points) / len(points)
    return x, y


def draw_hand_landmarks(
    image: np.ndarray,
    landmarks: Sequence[tuple[int, int]],
    palm_center_xy: tuple[float, float],
    *,
    label: str | None = None,
) -> np.ndarray:
    """Draw the landmarks and the palm centre onto a copy of a BGR image.

    `cv2` is imported lazily: this module is otherwise numpy-only, and the geometry above stays
    usable without pulling OpenCV in.
    """
    import cv2 as cv

    out = image.copy()
    for x, y in landmarks:
        cv.circle(out, (int(x), int(y)), 4, (0, 255, 0), -1)
    cx, cy = palm_center_xy
    cv.circle(out, (int(round(cx)), int(round(cy))), 8, (0, 0, 255), -1)
    if label:
        cv.putText(
            out, label, (int(round(cx)) + 12, int(round(cy)) - 12),
            cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv.LINE_AA,
        )
    return out
