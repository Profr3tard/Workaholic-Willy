from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _validate_frame_array(value: np.ndarray, *, name: str, allow_empty: bool = False) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim not in (2, 3) and not (allow_empty and array.shape == (0,)):
        raise ValueError(f"{name} must be a 2-D or 3-D image array, got shape {array.shape}")
    if not allow_empty and array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if array.size and not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must use a numeric dtype, got {array.dtype}")
    return array


@dataclass(frozen=True, slots=True)
class StereoFrame:
    """Left / right BGR frame pair from any stereo rig."""
    left: np.ndarray
    right: np.ndarray

    def __post_init__(self) -> None:
        left = _validate_frame_array(self.left, name="left")
        right = _validate_frame_array(self.right, name="right")
        if left.shape[:2] != right.shape[:2]:
            raise ValueError(f"left/right frame sizes differ: {left.shape[:2]} vs {right.shape[:2]}")
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)


@dataclass(frozen=True, slots=True)
class RGBDFrame:
    """Colour (BGR uint8) + depth (uint16, millimetres) from an RGB-D camera."""
    color: np.ndarray
    depth: np.ndarray

    def __post_init__(self) -> None:
        color = _validate_frame_array(self.color, name="color")
        depth = _validate_frame_array(self.depth, name="depth", allow_empty=True)
        if depth.size and depth.shape[:2] != color.shape[:2]:
            raise ValueError(f"color/depth frame sizes differ: {color.shape[:2]} vs {depth.shape[:2]}")
        object.__setattr__(self, "color", color)
        object.__setattr__(self, "depth", depth)


# The two frame kinds a rig-keyed provider/consumer fronts (R9.1): a StereoFrame (left/right) or an
# RGBDFrame (colour/depth). A readability alias for the heterogeneous return/param union.
AnyFrame = StereoFrame | RGBDFrame
