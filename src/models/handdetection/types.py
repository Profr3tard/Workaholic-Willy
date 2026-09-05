"""Value objects for hand detection: a 2-D detection, a 3-D position, and a gesture reading.

Frozen dataclasses that validate in `__post_init__`, per the repo's value-object convention. Every
one of these crosses a boundary between a C++ inference graph and geometry code that will happily
propagate a NaN into a robot pose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Sequence

import numpy as np

from src.models.handdetection.landmarks import LANDMARK_COUNT

__all__ = [
    "GestureReading",
    "HandGesture",
    "HandObservation",
    "HandPosition3D",
    "Handedness",
    "LocatedHand",
    "PalmDetection",
    "as_landmark_tuple",
]


class Handedness(StrEnum):
    """Which hand MediaPipe thinks it saw.

    MediaPipe reports this from the camera's point of view on a non-mirrored image. A selfie-style
    preview flips it, which is a display concern: this enum records what the model said, and the
    caller that mirrors the image is the one that has to reinterpret it.
    """

    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


class HandGesture(StrEnum):
    """The gestures this package commits to recognising.

    Four values, not the seven MediaPipe's canned classifier knows: this package promises exactly
    thumbs-up and thumbs-down, and everything else is `OTHER`, which is not the same as `NONE`. At
    an operator console "the cell saw your hand and it was not a thumbs-up" and "the cell saw
    nothing" call for different actions, so they stay two values.
    """

    THUMB_UP = "thumb_up"
    THUMB_DOWN = "thumb_down"
    OTHER = "other"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class PalmDetection:
    """One hand found in one frame, in image space.

    `landmarks` holds exactly 21 integer pixel coordinates in MediaPipe's layout, and
    `palm_center_xy` is their palm-subset mean, sub-pixel and hence float.
    """

    palm_center_xy: tuple[float, float]
    landmarks: tuple[tuple[int, int], ...]
    hand_index: int = 0
    handedness: Handedness = Handedness.UNKNOWN
    #: MediaPipe's confidence in the handedness call, not in the detection itself. Kept because a
    #: low value is the usual explanation for a left/right label that looks wrong.
    handedness_score: float = 0.0

    def __post_init__(self) -> None:
        if len(self.landmarks) != LANDMARK_COUNT:
            raise ValueError(
                f"PalmDetection.landmarks must have {LANDMARK_COUNT} entries (MediaPipe layout), "
                f"got {len(self.landmarks)}"
            )
        for i, landmark in enumerate(self.landmarks):
            if not (isinstance(landmark, tuple) and len(landmark) == 2):
                raise ValueError(
                    f"PalmDetection.landmarks[{i}] must be an (x, y) tuple, got {landmark!r}"
                )
            x, y = landmark
            if not (isinstance(x, (int, np.integer)) and isinstance(y, (int, np.integer))):
                raise ValueError(
                    f"PalmDetection.landmarks[{i}] entries must be ints, got "
                    f"{(type(x).__name__, type(y).__name__)}"
                )
        if len(self.palm_center_xy) != 2:
            raise ValueError(
                f"PalmDetection.palm_center_xy must be (x, y), got {self.palm_center_xy!r}"
            )
        cx, cy = self.palm_center_xy
        if not (np.isfinite(cx) and np.isfinite(cy)):
            raise ValueError(
                f"PalmDetection.palm_center_xy must be finite, got {self.palm_center_xy!r}"
            )
        if self.hand_index < 0:
            raise ValueError(f"PalmDetection.hand_index must be >= 0, got {self.hand_index}")


@dataclass(frozen=True, slots=True)
class GestureReading:
    """What the canned classifier said about one hand.

    `raw_label` is kept even when `gesture` is `OTHER` or `NONE`: it is the only way to answer "what
    did it think that was", and mapping it away would make every unexpected result look identical.
    """

    gesture: HandGesture
    confidence: float
    raw_label: str = ""
    hand_index: int = 0

    def __post_init__(self) -> None:
        if not np.isfinite(self.confidence):
            raise ValueError(f"GestureReading.confidence must be finite, got {self.confidence!r}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"GestureReading.confidence must be in [0, 1], got {self.confidence!r}"
            )
        if self.hand_index < 0:
            raise ValueError(f"GestureReading.hand_index must be >= 0, got {self.hand_index}")


@dataclass(frozen=True, slots=True)
class HandObservation:
    """One hand seen once: where it is in the image, and what it was doing.

    The pairing exists because the canned gesture model returns both: it embeds a hand landmarker.
    Handing back two parallel lists and leaving the caller to align them by index is how the two
    halves end up describing different hands.
    """

    palm: PalmDetection
    gesture: GestureReading


@dataclass(frozen=True, slots=True)
class HandPosition3D:
    """A hand's palm centre in metric space: camera frame and robot base frame, millimetres.

    Both arrays are coerced to read-only `float64` of shape `(3,)`. Read-only because these are
    handed to motion code: a caller that mutates one in place would change a pose another caller is
    still holding, and the numpy-level `ValueError` names the mistake at the moment it happens.
    """

    position_base: np.ndarray
    position_cam: np.ndarray
    palm_center_xy: tuple[float, float]
    depth_mm: float
    rig_id: str = ""
    handedness: Handedness = Handedness.UNKNOWN
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("position_base", "position_cam"):
            array = np.asarray(getattr(self, name), dtype=np.float64).reshape(-1)
            if array.shape != (3,):
                raise ValueError(
                    f"HandPosition3D.{name} must have shape (3,), got {array.shape}"
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(f"HandPosition3D.{name} must be finite, got {array!r}")
            array.flags.writeable = False
            object.__setattr__(self, name, array)
        if len(self.palm_center_xy) != 2:
            raise ValueError(
                f"HandPosition3D.palm_center_xy must be (x, y), got {self.palm_center_xy!r}"
            )
        if not np.isfinite(self.depth_mm):
            raise ValueError(f"HandPosition3D.depth_mm must be finite, got {self.depth_mm!r}")
        if self.depth_mm <= 0.0:
            raise ValueError(
                f"HandPosition3D.depth_mm must be positive: a hand behind the camera is a "
                f"back-projection bug, not a detection: got {self.depth_mm!r}"
            )


@dataclass(frozen=True, slots=True)
class LocatedHand:
    """The complete answer about one hand: where it is in metric space, and what it was doing.

    Returned as one object rather than a tuple because these three always travel together and a
    caller that unpacks them into loose variables is one refactor away from pairing a position with
    the wrong gesture.
    """

    position: HandPosition3D
    gesture: GestureReading
    palm: PalmDetection


def as_landmark_tuple(landmarks: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """Freeze a landmark sequence for `PalmDetection`.

    A frozen dataclass holding a `list` is frozen in name only; this is the one-line conversion that
    keeps every construction site honest.
    """
    return tuple((int(x), int(y)) for x, y in landmarks)
