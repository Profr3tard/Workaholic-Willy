"""Isaac-backed simulation :class:`RobotArm` driver package.

Importing it is **always safe**, even on a host without
Isaac SDK installed:

* :mod:`.config` and :mod:`.adapter` are pure Python.
* :mod:`.session` and :mod:`.arm` only touch Isaac SDK symbols inside
  :meth:`IsaacSimSession.start`, invoked by :meth:`IsaacRobotArm.connect`.

Calling :meth:`IsaacRobotArm.connect` on a host without Isaac raises
:class:`~src.robot.core.IsaacNotAvailableError` (re-exported here for
convenience). The :class:`RobotVendor.SIM` factory in
:mod:`src.robot.drivers` therefore stays import-safe while still
constructing the driver eagerly so the protocol surface can be inspected for
tests. The Isaac-backed *grippers* live in
:mod:`src.robot.grippers.sim` (sim-only, deliberately unregistered).
"""

from __future__ import annotations

from .adapter import (
    willy_joints_to_isaac,
    willy_pose_to_isaac,
    isaac_joints_to_willy,
    isaac_pose_to_willy,
    isaac_rotmat_to_wxyz,
    isaac_wxyz_to_xyzw,
    metres_to_millimetres,
    millimetres_to_metres,
    xyzw_to_isaac_wxyz,
)
from ...core import IsaacNotAvailableError
from .arm import ISAAC_CAPABILITIES, IsaacRobotArm
from .config import SimCameraConfig, SimRobotConfig
from .session import IsaacSimSession

__all__ = [
    "ISAAC_CAPABILITIES",
    "IsaacNotAvailableError",
    "IsaacRobotArm",
    "IsaacSimSession",
    "SimCameraConfig",
    "SimRobotConfig",
    "willy_joints_to_isaac",
    "willy_pose_to_isaac",
    "isaac_joints_to_willy",
    "isaac_pose_to_willy",
    "isaac_rotmat_to_wxyz",
    "isaac_wxyz_to_xyzw",
    "metres_to_millimetres",
    "millimetres_to_metres",
    "xyzw_to_isaac_wxyz",
]
