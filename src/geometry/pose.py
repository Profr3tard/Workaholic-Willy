"""
:class:`Pose` frame-tagged, immutable rigid pose in 3-D space.

Replaces ad-hoc tuples / numpy arrays / vendor-specific structs (such as
``URPose``) on every public API across Workaholic-Willy. The numerics contract
matches the directive:

* ``position_mm``        - (3,) float64, **millimetres**
* ``quaternion_xyzw``    - (4,) float64, unit length, canonical sign
* ``frame``              - :class:`~src.geometry.frame.Frame`
* ``label``              - optional human-readable tag

Both ndarray fields are made read-only at construction so consumers
cannot silently mutate them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .exceptions import FrameMismatchError
from .frame import Frame
from .matrix import matrix_to_position_quaternion, position_quaternion_to_matrix
from .quaternion import (
    IDENTITY_QUAT_XYZW,
    angle_between,
    to_axis_angle,
)
from .validation import (
    normalise_quaternion,
    validate_position_mm,
)

__all__ = ["Pose"]


@dataclass(frozen=True, slots=True)
class Pose:
    """A rigid 6-DoF pose tagged with its coordinate frame.

    Construction validates and normalises inputs:

    * the position is coerced to ``float64`` and checked for shape (3,)
      and finiteness;
    * the quaternion is normalised and put into canonical sign form
      (``w >= 0``);
    """

    position_mm: np.ndarray
    quaternion_xyzw: np.ndarray
    frame: Frame
    label: str | None = None

    # ------------------------------------------------------------------
    # Construction / validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        pos = validate_position_mm(self.position_mm)
        quat = normalise_quaternion(self.quaternion_xyzw, canonicalise=True)
        if not isinstance(self.frame, Frame):
            object.__setattr__(self, "frame", Frame(self.frame))
        pos.setflags(write=False)
        quat.setflags(write=False)
        object.__setattr__(self, "position_mm", pos)
        object.__setattr__(self, "quaternion_xyzw", quat)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def identity(cls, frame: Frame, *, label: str | None = None) -> Pose:
        """Identity pose at the origin of ``frame``."""
        return cls(
            position_mm=np.zeros(3, dtype=np.float64),
            quaternion_xyzw=IDENTITY_QUAT_XYZW.copy(),
            frame=frame,
            label=label,
        )

    @classmethod
    def from_matrix(
        cls,
        T: np.ndarray,
        *,
        frame: Frame,
        label: str | None = None,
    ) -> Pose:
        """Build a pose from a validated 4x4 homogeneous matrix in millimetres."""
        t, q = matrix_to_position_quaternion(T)
        return cls(position_mm=t, quaternion_xyzw=q, frame=frame, label=label)

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------

    def to_matrix(self) -> np.ndarray:
        """Return a fresh 4x4 homogeneous transform (mm) for this pose."""
        return position_quaternion_to_matrix(self.position_mm, self.quaternion_xyzw)

    def with_frame(self, frame: Frame) -> Pose:
        """Return a copy in a different frame label."""
        return Pose(
            position_mm=self.position_mm.copy(),
            quaternion_xyzw=self.quaternion_xyzw.copy(),
            frame=frame,
            label=self.label,
        )

    def with_label(self, label: str | None) -> Pose:
        """Return a copy with a different ``label`` attribute."""
        return Pose(
            position_mm=self.position_mm.copy(),
            quaternion_xyzw=self.quaternion_xyzw.copy(),
            frame=self.frame,
            label=label,
        )

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def distance_to(self, other: Pose) -> float:
        """Euclidean distance in mm between two poses (frames must match)."""
        if self.frame != other.frame:
            raise FrameMismatchError(
                f"distance_to: frame mismatch {self.frame!r} vs {other.frame!r}"
            )
        return float(np.linalg.norm(self.position_mm - other.position_mm))

    def angle_to(self, other: Pose) -> float:
        """Geodesic angle (rad, 0..π) between two pose orientations."""
        if self.frame != other.frame:
            raise FrameMismatchError(
                f"angle_to: frame mismatch {self.frame!r} vs {other.frame!r}"
            )
        return angle_between(self.quaternion_xyzw, other.quaternion_xyzw)

    def axis_angle_rad(self) -> np.ndarray:
        """Return the axis-angle vector (rad) equivalent to the quaternion."""
        return to_axis_angle(self.quaternion_xyzw)

    # ------------------------------------------------------------------
    # Equality / hashing
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Pose):
            return NotImplemented
        return (
            self.frame == other.frame
            and self.label == other.label
            and np.array_equal(self.position_mm, other.position_mm)
            and np.array_equal(self.quaternion_xyzw, other.quaternion_xyzw)
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.frame,
                self.label,
                self.position_mm.tobytes(),
                self.quaternion_xyzw.tobytes(),
            )
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        x, y, z = self.position_mm
        qx, qy, qz, qw = self.quaternion_xyzw
        lbl = f", label={self.label!r}" if self.label else ""
        return (
            f"Pose(frame={self.frame.value!r}, "
            f"pos_mm=[{x:.2f}, {y:.2f}, {z:.2f}], "
            f"quat_xyzw=[{qx:.4f}, {qy:.4f}, {qz:.4f}, {qw:.4f}]{lbl})"
        )
