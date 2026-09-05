"""Unit scaling helpers shared across camera, calibration and depth code."""

from __future__ import annotations

__all__ = ["SUPPORTED_DISTANCE_UNITS", "unit_scaling"]


SUPPORTED_DISTANCE_UNITS: dict[str, float] = {
    "mm": 1.0,
    "cm": 0.1,
    "m": 0.001,
}


def unit_scaling(out: str) -> float:
    """Return the multiplier that converts millimetres into ``out``.

    A pure scaling lookup. It never rounds and never changes dtype for array
    callers that multiply by the returned factor.
    """
    key = str(out).strip().lower()
    try:
        return SUPPORTED_DISTANCE_UNITS[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(SUPPORTED_DISTANCE_UNITS))
        raise ValueError(f"unsupported distance unit {out!r}; expected one of: {allowed}") from exc
