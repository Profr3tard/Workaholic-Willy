"""
Vendor boundary adapters: :class:`URPose` <-> vendor-neutral :class:`Pose`.

This module is the **only** place in the codebase allowed to translate
between Universal Robots' axis-angle pose representation and the
canonical :class:`Pose` (mm + XYZW quaternion + :class:`Frame`).

Architectural rule
------------------
Outside ``src.robot.drivers.ur`` no module should import
``URPose``. Vendor-specific types stop here; everything above this
boundary speaks ``Pose``.

The geometry package must not host these adapters (geometry must not
know about UR types); they live here at the vendor boundary instead.
"""

from __future__ import annotations

import numpy as np

from src.geometry import Frame, Pose
from src.geometry.quaternion import from_axis_angle, to_axis_angle

from .pose import URPose

__all__ = ["urpose_to_pose", "pose_to_urpose"]


def urpose_to_pose(
    urpose: URPose,
    *,
    frame: Frame = Frame.BASE,
    label: str | None = None,
) -> Pose:
    """Convert a :class:`URPose` to a vendor-neutral :class:`Pose`.

    ``URPose`` stores ``(x, y, z)`` in **millimetres** and orientation as
    an axis-angle vector ``(rx, ry, rz)`` in **radians**. Both pieces map
    directly to the geometry numerics contract; ``frame`` defaults to
    :attr:`Frame.BASE` because the UR controller reports TCP poses in
    base frame.

    The ``label`` argument overrides the URPose's own label when given.
    """
    pos = np.array([urpose.x, urpose.y, urpose.z], dtype=np.float64)
    rvec = np.array([urpose.rx, urpose.ry, urpose.rz], dtype=np.float64)
    quat = from_axis_angle(rvec)
    return Pose(
        position_mm=pos,
        quaternion_xyzw=quat,
        frame=frame,
        label=label if label is not None else (urpose.label or None),
    )


def pose_to_urpose(pose: Pose) -> URPose:
    """Convert a :class:`Pose` back to a :class:`URPose`."""
    rvec = to_axis_angle(pose.quaternion_xyzw)
    x, y, z = pose.position_mm
    return URPose(
        x=float(x),
        y=float(y),
        z=float(z),
        rx=float(rvec[0]),
        ry=float(rvec[1]),
        rz=float(rvec[2]),
        label=pose.label or "",
    )
