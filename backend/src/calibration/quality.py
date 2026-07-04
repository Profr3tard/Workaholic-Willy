"""
Quality bands for calibration results.

A *band* maps a scalar error metric (RMSE) onto one of five labels:

    ``excellent`` < ``good`` < ``marginal`` < ``poor``  (+ ``unknown``)

Bands are vendor-neutral: the eye-to-hand routine reports translational
RMSE in millimetres, whereas stereo intrinsics report reprojection RMSE
in pixels. Each callsite chooses the appropriate :class:`QualityBands`
subclass.

The dataclasses are immutable so they can be safely shared across
threads. ``__post_init__`` enforces strict monotonicity, which is the
only guarantee callers rely on when classifying.
"""


from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "DEFAULT_BANDS_MM",
    "DEFAULT_BANDS_PX",
    "QUALITY_LABELS",
    "QualityBandsMm",
    "QualityBandsPx",
    "QualityLabel",
    "classify_rmse",
]


QualityLabel = Literal["excellent", "good", "marginal", "poor", "unknown"]

QUALITY_LABELS: tuple[QualityLabel, ...] = (
    "excellent",
    "good",
    "marginal",
    "poor",
    "unknown",
)


def _validate_bands(excellent: float, good: float, marginal: float) -> None:
    for name, value in (("excellent", excellent), ("good", good), ("marginal", marginal)):
        if not math.isfinite(value):
            raise ValueError(f"{name} band must be finite, got {value!r}")
        if value <= 0:
            raise ValueError(f"{name} band must be > 0, got {value!r}")
    if not (excellent < good < marginal):
        raise ValueError(
            "quality bands must satisfy excellent < good < marginal "
            f"(got {excellent}, {good}, {marginal})"
        )


@dataclass(frozen=True, slots=True)
class QualityBandsMm:
    """RMSE thresholds in millimetres (hand-eye / extrinsics)."""

    excellent: float = 1.0
    good: float = 2.5
    marginal: float = 5.0

    def __post_init__(self) -> None:
        _validate_bands(self.excellent, self.good, self.marginal)


@dataclass(frozen=True, slots=True)
class QualityBandsPx:
    """RMSE thresholds in pixels (stereo / intrinsics)."""

    excellent: float = 0.5
    good: float = 1.0
    marginal: float = 2.0

    def __post_init__(self) -> None:
        _validate_bands(self.excellent, self.good, self.marginal)


DEFAULT_BANDS_MM = QualityBandsMm()
DEFAULT_BANDS_PX = QualityBandsPx()


def classify_rmse(
    value: float | None,
    bands: QualityBandsMm | QualityBandsPx = DEFAULT_BANDS_MM,
) -> QualityLabel:
    """Map an RMSE value onto a :data:`QualityLabel`.

    ``None`` and non-finite inputs map to ``"unknown"``. Negative values
    are also treated as ``"unknown"`` rather than silently classifying
    as ``"excellent"``.
    """
    if value is None:
        return "unknown"
    if not math.isfinite(value) or value < 0:
        return "unknown"
    if value <= bands.excellent:
        return "excellent"
    if value <= bands.good:
        return "good"
    if value <= bands.marginal:
        return "marginal"
    return "poor"