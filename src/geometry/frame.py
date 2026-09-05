"""Coordinate frames for the geometry subsystem.

Every public :class:`~src.geometry.pose.Pose` and
:class:`~src.geometry.transform.Transform` names its :class:`Frame`; the
public API has no implicit or unnamed frames.

The set is small and covers the robot vision stack:

* ``WORLD``  application-defined fixed frame, a table corner for instance.
* ``BASE``   robot base frame, link 0 of the kinematic chain.
* ``CAMERA`` optical frame of a mono, stereo or depth camera.
* ``MARKER`` fiducial marker frame, origin at the marker centre.
* ``TCP``    tool centre point as reported by the robot controller.
* ``TOOL``   physical tool or flange frame, often equal to TCP but not always.
* ``OBJECT`` frame attached to a perceived object instance.
* ``GRASP``  frame of a parallel-jaw grasp, Z is approach and X is closure.

A subsystem needing another frame, an IMU for instance, adds a member here
rather than passing a raw string at the call site.
"""

from __future__ import annotations

from enum import StrEnum


class Frame(StrEnum):
    """Canonical coordinate frames used across the stack.

    Each member's string value is what serialization writes and reads back, so
    those values are part of the wire format.
    """

    WORLD = "world"
    BASE = "base"
    CAMERA = "camera"
    MARKER = "marker"
    TCP = "tcp"
    TOOL = "tool"
    OBJECT = "object"
    GRASP = "grasp"

    def __repr__(self) -> str:  # pragma: no cover (cosmetic)
        return f"Frame.{self.name}"
