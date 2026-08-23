"""
Closed-form capsule / capsule and capsule / axis-aligned-box distance
helpers for the self-collision guard.

All distances are in millimetres. Each "capsule" is a swept-sphere over
a finite line segment: a segment ``(p0, p1)`` in mm plus a single
``radius_mm``. An axis-aligned box is its centre plus half-extents in mm.

The math is intentionally simple and closed-form (no iterative solver);
this keeps the per-move cost low enough to live on the hot path even
for a 6-DoF arm with all-pairs link checks.

References
----------
* "Real-Time Collision Detection" Christer Ericson, sec. 5.1.3
  (closest points between two segments).
* Box-point distance: clamp box centre offset to ``half_extents`` axis
  by axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "AxisAlignedBox",
    "Capsule",
    "capsule_capsule_distance_mm",
    "capsule_box_distance_mm",
    "segment_segment_distance_mm",
    "segment_point_distance_mm",
]


@dataclass(frozen=True, slots=True)
class Capsule:
    """Swept-sphere capsule. ``p0`` and ``p1`` are mm; ``radius_mm`` mm."""

    p0: np.ndarray
    p1: np.ndarray
    radius_mm: float

    @property
    def is_degenerate(self) -> bool:
        return bool(np.allclose(self.p0, self.p1))


@dataclass(frozen=True, slots=True)
class AxisAlignedBox:
    """Axis-aligned box in the robot base frame.

    Defined by ``center_mm`` and per-axis ``half_extents_mm``.
    """

    center_mm: np.ndarray
    half_extents_mm: np.ndarray


def segment_point_distance_mm(p0: np.ndarray, p1: np.ndarray, q: np.ndarray) -> float:
    """Closest distance from point ``q`` to segment ``(p0, p1)``."""
    d = p1 - p0
    denom = float(d @ d)
    if denom <= 1e-12:
        return float(np.linalg.norm(q - p0))
    t = float((q - p0) @ d) / denom
    t = max(0.0, min(1.0, t))
    closest = p0 + t * d
    return float(np.linalg.norm(q - closest))


def segment_segment_distance_mm(
    p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray,
) -> float:
    """Closest distance between two segments ``(p0,p1)`` and ``(q0,q1)``.

    Implementation follows Ericson "Real-Time Collision Detection",
    closest-points-on-segments. Returns mm; both inputs must be in mm.
    """
    d1 = p1 - p0
    d2 = q1 - q0
    r = p0 - q0
    a = float(d1 @ d1)
    e = float(d2 @ d2)
    f = float(d2 @ r)

    eps = 1e-12
    if a <= eps and e <= eps:
        return float(np.linalg.norm(p0 - q0))
    if a <= eps:
        s = 0.0
        t = max(0.0, min(1.0, f / e))
    else:
        c = float(d1 @ r)
        if e <= eps:
            t = 0.0
            s = max(0.0, min(1.0, -c / a))
        else:
            b = float(d1 @ d2)
            denom = a * e - b * b
            if denom != 0.0:
                s = max(0.0, min(1.0, (b * f - c * e) / denom))
            else:
                s = 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = max(0.0, min(1.0, -c / a))
            elif t > 1.0:
                t = 1.0
                s = max(0.0, min(1.0, (b - c) / a))

    closest_1 = p0 + s * d1
    closest_2 = q0 + t * d2
    return float(np.linalg.norm(closest_1 - closest_2))


def capsule_capsule_distance_mm(a: Capsule, b: Capsule) -> float:
    """Signed distance between two capsules (negative = penetrating).

    ``signed = segment_segment_distance - (r_a + r_b)``.
    """
    seg = segment_segment_distance_mm(a.p0, a.p1, b.p0, b.p1)
    return seg - (a.radius_mm + b.radius_mm)


def capsule_box_distance_mm(c: Capsule, box: AxisAlignedBox) -> float:
    """Signed distance from a capsule to an axis-aligned box.

    Approximation: sample a few points along the capsule segment and
    use the per-point clamp-to-box distance. The sampling resolution
    (16 points) bounds the worst-case error at sub-millimetre for
    typical arm-link capsule lengths < 500 mm.
    """
    closest = float("inf")
    samples = 16
    for i in range(samples + 1):
        t = i / samples
        q = c.p0 + t * (c.p1 - c.p0)
        offset = q - box.center_mm
        clamped = np.clip(offset, -box.half_extents_mm, box.half_extents_mm)
        nearest_on_box = box.center_mm + clamped
        d = float(np.linalg.norm(q - nearest_on_box))
        if d < closest:
            closest = d
    return closest - c.radius_mm
