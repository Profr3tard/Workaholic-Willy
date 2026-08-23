"""
KUKA pose / joint conversions.

Bridges between the Willy-internal representation
(:class:`src.geometry.Pose` millimetres + XYZW quaternion,
joints in radians) and the KUKA controller representation:

* **E6POS** ``{X, Y, Z, A, B, C}`` ``X/Y/Z`` in millimetres,
  ``A/B/C`` are intrinsic Z--Y--X Euler angles in **degrees** with the
  KUKA convention ``R = Rz(A) * Ry(B) * Rx(C)``.
* **E6AXIS** ``{A1, A2, A3, A4, A5, A6}`` joint angles in **degrees**.

Conversions never round-trip through axis-angle / Rodrigues to avoid
gimbal-style cancellations: KUKA is a strictly intrinsic ZYX Euler
system and we use scipy's ``Rotation`` directly.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from src.geometry import Frame, Pose
from src.geometry.quaternion import from_euler, to_euler

__all__ = [
    "KukaCartesian",
    "joints_deg_to_rad",
    "joints_rad_to_deg",
    "kuka_cartesian_to_pose",
    "pose_to_kuka_cartesian",
]


@dataclass(frozen=True, slots=True)
class KukaCartesian:
    """Lightweight container mirroring KUKA E6POS ``{X, Y, Z, A, B, C}``.

    ``x/y/z`` are millimetres; ``a/b/c`` are intrinsic Z-Y-X Euler
    angles in **degrees** following the KUKA convention
    (``R = Rz(A) * Ry(B) * Rx(C)``).
    """

    x: float
    y: float
    z: float
    a: float
    b: float
    c: float

    def to_dict(self) -> dict:
        return {"X": self.x, "Y": self.y, "Z": self.z, "A": self.a, "B": self.b, "C": self.c}

    @classmethod
    def from_dict(cls, d: dict) -> KukaCartesian:
        return cls(
            x=float(d["X"]),
            y=float(d["Y"]),
            z=float(d["Z"]),
            a=float(d["A"]),
            b=float(d["B"]),
            c=float(d["C"]),
        )

    def as_tuple(self) -> tuple:
        return (self.x, self.y, self.z, self.a, self.b, self.c)


def kuka_cartesian_to_pose(
    e6: KukaCartesian,
    *,
    label: str | None = None,
) -> Pose:
    """Convert a KUKA ``{X, Y, Z, A, B, C}`` reading to a :class:`Pose`.

    The returned pose is tagged :attr:`Frame.BASE` (KUKA's ``$BASE``
    frame, by convention the robot world frame after any tool/base
    chain has been resolved on the controller side).

    Numerics
    --------
    KUKA's convention is ``R = Rz(A) * Ry(B) * Rx(C)`` with angles in
    **degrees**. The geometry helper :func:`from_euler` (order ``"xyz"``)
    internally builds ``Rz(arr[2]) * Ry(arr[1]) * Rx(arr[0])``, so we
    pass ``[C, B, A]`` (in radians) to obtain the same rotation matrix.
    """
    abc_rad = np.radians(np.array([e6.c, e6.b, e6.a], dtype=np.float64))
    quat = from_euler(abc_rad, order="xyz")
    return Pose(
        position_mm=np.array([e6.x, e6.y, e6.z], dtype=np.float64),
        quaternion_xyzw=quat,
        frame=Frame.BASE,
        label=label,
    )


def pose_to_kuka_cartesian(pose: Pose) -> KukaCartesian:
    """Convert a :class:`Pose` (Frame.BASE) to a KUKA ``E6POS``-style record.

    See :func:`kuka_cartesian_to_pose` for the array-ordering rationale:
    the geometry helper returns ``[x_rad, y_rad, z_rad]`` corresponding
    to KUKA ``[C, B, A]`` in radians.
    """
    if pose.frame is not Frame.BASE:
        raise ValueError(
            f"pose_to_kuka_cartesian requires Frame.BASE; got {pose.frame!r}."
        )
    xyz_rad = to_euler(pose.quaternion_xyzw, order="xyz")
    xyz_deg = np.degrees(xyz_rad)
    return KukaCartesian(
        x=float(pose.position_mm[0]),
        y=float(pose.position_mm[1]),
        z=float(pose.position_mm[2]),
        a=float(xyz_deg[2]),
        b=float(xyz_deg[1]),
        c=float(xyz_deg[0]),
    )


def joints_rad_to_deg(joints_rad: Sequence[float]) -> list[float]:
    """Convert joint angles from **radians** to **degrees** (KUKA wire format)."""
    return [float(np.degrees(j)) for j in joints_rad]


def joints_deg_to_rad(joints_deg: Iterable[float]) -> list[float]:
    """Convert joint angles from **degrees** (KUKA wire format) to **radians**."""
    return [float(np.radians(j)) for j in joints_deg]
