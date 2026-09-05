"""Shared segmentation result type, emitted by every segmenter backend.

It reaches ``src/robot/grasping/`` as a public input through the read-only
``SegmentationLike`` protocol. The ``mask`` is a ``uint8`` array of shape
``(H, W)`` with values in ``{0, 1}``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["SegmentationResult"]


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    label: str
    score: float
    bbox_xyxy: tuple[int, int, int, int]
    mask: np.ndarray                     # uint8 mask, shape (H, W), values {0,1}
    mask_area_px: int
    centroid_xy: tuple[float, float]
    derived_bbox_xyxy: tuple[int, int, int, int]
    inference_time_s: float
    timestamp_utc: str
    frame_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
