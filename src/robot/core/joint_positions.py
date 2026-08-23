"""
Typed joint-vector wrapper used across the vendor-neutral robot surface.

Different robots have different DoF counts (UR = 6, Franka = 7, KUKA iiwa = 7,
some humanoid arms = 8). Passing raw ``np.ndarray`` everywhere makes those
counts implicit and easy to mismatch. ``JointPositions`` makes the DoF an
explicit, validated property of the value.

Numerics
--------
* Values are radians (``float64``).
* Stored as a frozen, write-protected ``np.ndarray`` of shape ``(dof,)``.
* Equality is exact (bit-level) — useful as dict keys / cache keys.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

__all__ = ["JointPositions"]


def _validate_joint_array(values: np.ndarray) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError(
            f"JointPositions expects a 1-D array; got shape {values.shape}."
        )
    if values.size == 0:
        raise ValueError("JointPositions requires at least one joint value.")
    if not np.isfinite(values).all():
        raise ValueError("JointPositions values must all be finite (no NaN/inf).")
    return values


@dataclass(frozen=True, slots=True)
class JointPositions:
    """
    Immutable joint-space configuration.

    Construct from any 1-D iterable of floats; the array is copied,
    cast to ``float64``, validated, and write-locked.

    Parameters
    ----------
    values : array-like
        Joint angles in radians, one per DoF.
    """

    values: np.ndarray
    # frame: not stored, joint space is implicit per arm. The
    # arm's RobotCapabilities.dof field gives the expected length.

    def __init__(self, values: Iterable[float] | np.ndarray) -> None:
        arr = np.array(values, dtype=np.float64, copy=True)
        _validate_joint_array(arr)
        arr.setflags(write=False)
        # frozen dataclass: must use object.__setattr__
        object.__setattr__(self, "values", arr)

    # ---- structural -----------------------------------------------------

    @property
    def dof(self) -> int:
        """Degrees of freedom (length of the joint vector)."""
        return int(self.values.shape[0])

    def check_dof(self, expected: int) -> "JointPositions":
        """Assert this vector has ``expected`` joints; raise ``ValueError`` otherwise. Returns ``self``.

        Canonical DoF check so drivers/guards stop re-implementing ad-hoc joint-count
        validation. Joint space is implicit per arm; the arm's ``RobotCapabilities.dof`` gives the
        expected length.
        """
        if self.dof != int(expected):
            raise ValueError(f"JointPositions has {self.dof} joints; expected {int(expected)}.")
        return self

    def __len__(self) -> int:
        return self.dof

    def __iter__(self):
        return iter(self.values.tolist())

    def __getitem__(self, idx: int) -> float:
        return float(self.values[idx])

    def __array__(self, dtype=None):
        # Allow ``np.asarray(jp)`` to round-trip cheaply (still write-protected).
        return self.values if dtype is None else self.values.astype(dtype, copy=False)

    # ---- value semantics ------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, JointPositions):
            return NotImplemented
        if other.values.shape != self.values.shape:
            return False
        return bool(np.array_equal(self.values, other.values))

    def __hash__(self) -> int:
        return hash((self.values.shape, self.values.tobytes()))

    def __repr__(self) -> str:
        # Compact repr — full precision available via .values
        formatted = ", ".join(f"{v:.6f}" for v in self.values.tolist())
        return f"JointPositions(dof={self.dof}, values=[{formatted}])"

    # ---- conversions ----------------------------------------------------

    def tolist(self) -> list[float]:
        """Plain Python list of joint angles (rad)."""
        return self.values.tolist()

    @classmethod
    def from_list(cls, values: Iterable[float]) -> JointPositions:
        """Convenience alias for ``JointPositions(values)``."""
        return cls(values)
