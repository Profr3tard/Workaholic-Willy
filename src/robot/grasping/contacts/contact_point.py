"""One surface contact: a position + its outward unit normal (external frame, millimetres).

Shared substrate composed by both finger-contact grasp types so they validate a contact the same
way: a finite 3D position and a normal within 1% of unit length, stored normalised. The parallel-jaw
:class:`~src.robot.grasping.contacts.ContactPair` holds two of these; the multi-finger
``MultiContactGrasp`` holds N. Suction is deliberately NOT built on this its ``approach`` is a
pressing direction, not an outward surface normal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.robot.grasping.geometry import as_vec3

__all__ = ["ContactPoint", "as_unit_normal"]

_UNIT_NORMAL_TOLERANCE = 0.01  # accept normals within 1% of unit length, then normalise


def as_unit_normal(value: Any, name: str) -> np.ndarray:
    """Validate a near-unit normal (length within 1% of 1) and return it normalised to exactly unit."""
    arr = as_vec3(value, name)
    norm = float(np.linalg.norm(arr))
    if not 1.0 - _UNIT_NORMAL_TOLERANCE <= norm <= 1.0 + _UNIT_NORMAL_TOLERANCE:
        raise ValueError(f"{name} must be unit length (within 1%); got {norm:.4f}")
    return arr / norm


@dataclass(frozen=True, slots=True)
class ContactPoint:
    """A surface contact: position (mm) + outward unit ``normal``, in a shared external frame."""

    point_mm: np.ndarray
    normal: np.ndarray

    def __post_init__(self) -> None:
        point = as_vec3(self.point_mm, "point_mm")
        normal = as_unit_normal(self.normal, "normal")
        point.setflags(write=False)
        normal.setflags(write=False)
        object.__setattr__(self, "point_mm", point)
        object.__setattr__(self, "normal", normal)

    def to_dict(self) -> dict[str, Any]:
        return {"point_mm": self.point_mm.tolist(), "normal": self.normal.tolist()}
