"""OpenCV colour-space helpers.

OpenCV returns BGR images while almost every ML model (Pillow, torch,
MediaPipe, transformers) expects RGB. The wrappers name that conversion, so a
call site reads ``bgr_to_rgb(frame)`` rather than
``cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)``.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["bgr_to_rgb", "rgb_to_bgr"]


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR frame to RGB."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert an RGB frame back to the BGR layout OpenCV expects."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
