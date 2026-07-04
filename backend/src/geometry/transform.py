"""
:class:`Transform` — a typed, frame-aware rigid transform.

A ``Transform`` carries an explicit *direction* in the form of a
``from_frame`` / ``to_frame`` pair. The semantics chosen across
Workaholic-Willy are:

    A point ``p_A`` expressed in ``from_frame`` is mapped to ``to_frame``
    via ``p_B = R · p_A + t``.

That is, ``Transform(from_frame=A, to_frame=B)`` is equivalent to the
homogeneous matrix usually written as ``T_B_A`` (or ``T_{B -> A}``).
This convention is enforced by :meth:`apply_pose` and :meth:`compose`.

The dataclass is frozen + slotted, all ndarrays are read-only, and the
quaternion is canonicalised on construction so equality is well defined.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .frame import Frame
from .matrix import (
    matrix_to_position_quaternion,
    position_quaternion_to_matrix,
)
from .quaternion import (
    IDENTITY_QUAT_XYZW,
    multiply,
    rotate_vector,
)
from .validation import (
    normalise_quaternion,
    validate_frames_compatible,
    validate_position_mm,
)

if TYPE_CHECKING:  # pragma: no cover - import only for typing
    from .pose import Pose


__all__ = ["Transform"]


@dataclass(frozen=True, slots=True)
class Transform:
    """Typed rigid transform between two frames.

    Attributes
    ----------
    translation_mm:
        (3,) float64 ndarray, millimetres.
    quaternion_xyzw:
        (4,) float64 unit XYZW quaternion (canonical, ``w >= 0``).
    from_frame, to_frame:
        Source and target :class:`Frame`. The transform takes data
        expressed in ``from_frame`` and returns it in ``to_frame``.
    """

    translation_mm: np.ndarray
    quaternion_xyzw: np.ndarray
    from_frame: Frame
    to_frame: Frame

    # ------------------------------------------------------------------
    # Construction / validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        t = validate_position_mm(self.translation_mm, name="translation_mm")
        q = normalise_quaternion(self.quaternion_xyzw, canonicalise=True)
        if not isinstance(self.from_frame, Frame):
            object.__setattr__(self, "from_frame", Frame(self.from_frame))
        if not isinstance(self.to_frame, Frame):
            object.__setattr__(self, "to_frame", Frame(self.to_frame))
        t.setflags(write=False)
        q.setflags(write=False)
        object.__setattr__(self, "translation_mm", t)
        object.__setattr__(self, "quaternion_xyzw", q)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def identity(cls, *, from_frame: Frame, to_frame: Frame) -> Transform:
        """Zero translation + identity rotation between two frames.

        ``from_frame`` may equal ``to_frame``; in that case the transform
        is the literal identity.
        """
        return cls(
            translation_mm=np.zeros(3, dtype=np.float64),
            quaternion_xyzw=IDENTITY_QUAT_XYZW.copy(),
            from_frame=from_frame,
            to_frame=to_frame,
        )

    @classmethod
    def from_matrix(
        cls,
        T: np.ndarray,
        *,
        from_frame: Frame,
        to_frame: Frame,
    ) -> Transform:
        """Build a transform from a validated 4x4 homogeneous matrix (mm)."""
        t, q = matrix_to_position_quaternion(T)
        return cls(
            translation_mm=t,
            quaternion_xyzw=q,
            from_frame=from_frame,
            to_frame=to_frame,
        )

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def to_matrix(self) -> np.ndarray:
        """Return a fresh 4x4 homogeneous transform (mm)."""
        return position_quaternion_to_matrix(self.translation_mm, self.quaternion_xyzw)

    def inverse(self) -> Transform:
        """Return the inverse transform with frames swapped."""
        # For a rigid transform, R_inv = R^T and t_inv = -R^T t.
        # In quaternion form, q_inv is the conjugate (because q is unit).
        from .quaternion import conjugate  # local import — avoid cycles

        q_inv = conjugate(self.quaternion_xyzw)
        t_inv = -rotate_vector(q_inv, self.translation_mm)
        return Transform(
            translation_mm=t_inv,
            quaternion_xyzw=q_inv,
            from_frame=self.to_frame,
            to_frame=self.from_frame,
        )

    def compose(self, other: Transform) -> Transform:
        """Return the composition ``self`` then ``other``.

        Convention: if ``self`` is "A→B" and ``other`` is "B→C", the
        result is "A→C". Requires
        ``self.to_frame == other.from_frame``. Equivalent to the matrix
        product ``other.to_matrix() @ self.to_matrix()``.
        """
        validate_frames_compatible(
            self.to_frame, other.from_frame, op="Transform.compose"
        )
        # Apply self first, then other:
        #   p_C = R_other (R_self p_A + t_self) + t_other
        #       = (R_other R_self) p_A + (R_other t_self + t_other)
        new_q = multiply(other.quaternion_xyzw, self.quaternion_xyzw)
        new_t = rotate_vector(other.quaternion_xyzw, self.translation_mm) + other.translation_mm
        return Transform(
            translation_mm=new_t,
            quaternion_xyzw=new_q,
            from_frame=self.from_frame,
            to_frame=other.to_frame,
        )

    def apply_point(self, point_mm: np.ndarray) -> np.ndarray:
        """Map a (3,) point from ``self.from_frame`` to ``self.to_frame``."""
        p = validate_position_mm(point_mm, name="point_mm")
        return rotate_vector(self.quaternion_xyzw, p) + self.translation_mm

    def apply_pose(self, pose: Pose) -> Pose:
        """Re-express ``pose`` from ``self.from_frame`` in ``self.to_frame``.

        Raises
        ------
        InvalidTransformError
            If ``pose.frame != self.from_frame``.
        """
        from .pose import Pose  # local import — avoid cycles

        validate_frames_compatible(
            self.from_frame, pose.frame, op="Transform.apply_pose"
        )
        new_pos = self.apply_point(pose.position_mm)
        new_quat = multiply(self.quaternion_xyzw, pose.quaternion_xyzw)
        return Pose(
            position_mm=new_pos,
            quaternion_xyzw=new_quat,
            frame=self.to_frame,
            label=pose.label,
        )

    # ------------------------------------------------------------------
    # Equality
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Transform):
            return NotImplemented
        return (
            self.from_frame == other.from_frame
            and self.to_frame == other.to_frame
            and np.array_equal(self.translation_mm, other.translation_mm)
            and np.array_equal(self.quaternion_xyzw, other.quaternion_xyzw)
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.from_frame,
                self.to_frame,
                self.translation_mm.tobytes(),
                self.quaternion_xyzw.tobytes(),
            )
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        x, y, z = self.translation_mm
        qx, qy, qz, qw = self.quaternion_xyzw
        return (
            f"Transform({self.from_frame.value!r} → {self.to_frame.value!r}, "
            f"t_mm=[{x:.2f}, {y:.2f}, {z:.2f}], "
            f"quat_xyzw=[{qx:.4f}, {qy:.4f}, {qz:.4f}, {qw:.4f}])"
        )