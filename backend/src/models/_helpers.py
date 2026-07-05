"""
Shared detection & segmentation result type, emitted by every detector & segmenter backend.

Segmenter only: 
    Crosses into ``backend/src/robot/grasping/`` as a public input via the read-only
    ``SegmentationLike`` protocol. The ``mask`` is a ``uint8`` array of shape
    ``(H, W)`` with values in ``{0, 1}``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["Detection", "SegmentationResult"]


@dataclass(frozen=True, slots=True)
class Detection:
    """Immutable detection result emitted by the zero-shot and closed-set detectors.

    All numeric fields are validated in :meth:`__post_init__`:

    * ``box`` must have length 4 and be in ``(x0, y0, x1, y1)`` order with
      ``x1 > x0`` and ``y1 > y0`` and all values finite.
    * ``x_center`` / ``y_center`` must be finite.
    * ``score`` must lie in ``[0, 1]``.
    * ``label`` must be a non-empty string.
    """

    box: list[float]
    x_center: float
    y_center: float
    label: str
    score: float

    def __post_init__(self) -> None:
        if len(self.box) != 4:
            raise ValueError(
                f"Detection.box must have length 4 (x0, y0, x1, y1), got {len(self.box)}"
            )
        x0, y0, x1, y1 = self.box
        for name, v in (("x0", x0), ("y0", y0), ("x1", x1), ("y1", y1),
                        ("x_center", self.x_center), ("y_center", self.y_center),
                        ("score", self.score)):
            if not np.isfinite(v):
                raise ValueError(f"Detection.{name} must be finite, got {v!r}")
        if not (x1 > x0 and y1 > y0):
            raise ValueError(
                f"Detection.box must satisfy x1>x0 and y1>y0, got {self.box!r}"
            )
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"Detection.score must be in [0, 1], got {self.score!r}")
        if not isinstance(self.label, str) or not self.label:
            raise ValueError(f"Detection.label must be a non-empty string, got {self.label!r}")
        

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