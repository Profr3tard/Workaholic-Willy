"""
Coordinate frames for the Workaholic-Willy geometry subsystem.

Every public :class:`~src.geometry.pose.Pose` and
:class:`~src.geometry.transform.Transform` carries an explicit
:class:`Frame`. There are no implicit / unnamed frames in public APIs.

The set is deliberately small. It covers the canonical robot vision
stack:

* ``WORLD``  - application-defined fixed frame (e.g. table corner).
* ``BASE``   - robot base frame (link 0 of the kinematic chain).
* ``CAMERA`` - optical frame of a (mono / stereo / depth) camera.
* ``MARKER`` - fiducial marker frame, origin at the marker center.
* ``TCP``    - Tool Centre Point reported by the robot controller.
* ``TOOL``   - physical tool / flange frame (often == TCP, but not always).
* ``OBJECT`` - frame attached to a perceived object instance.
* ``GRASP``  - frame of a parallel-jaw grasp (Z = approach, X = closure).

If a future subsystem needs more frames (e.g. ``IMU``, ``MARKER``), they
should be added here *never* introduced as raw strings at call sites.
"""

from __future__ import annotations

from enum import StrEnum


class Frame(StrEnum):
    """Canonical coordinate frames used across Workaholic-Willy."""

    WORLD = "world"
    BASE = "base"
    CAMERA = "camera"
    MARKER = "marker"
    TCP = "tcp"
    TOOL = "tool"
    OBJECT = "object"
    GRASP = "grasp"

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Frame.{self.name}"
