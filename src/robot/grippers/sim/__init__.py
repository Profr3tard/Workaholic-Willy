"""SIM-ONLY grippers (Isaac Sim backends).

These :class:`~src.robot.core.Gripper` implementations drive gripper
articulations **inside Isaac Sim**. They are deliberately kept OUT of the
:mod:`src.robot.grippers` vendor registry no :class:`GripperVendor`
member, not reachable via :func:`~src.robot.grippers.create_gripper` —
so they can NEVER be selected for a real robot from ``from_robot_config``. A
real cell must use a real vendor driver (``robotiq`` / ``dummy`` / ``none``).
The sim runners hand-build these with a live ``IsaacSimSession``; every
``isaacsim.*`` import stays lazy so this package imports on macOS / CI.

* :class:`IsaacGripper` parallel-jaw, driven by a swappable :class:`GripperProfile`.
* :class:`IsaacSuctionGripper` Isaac surface-gripper (vacuum on/off), driven by a
  swappable :class:`SuctionCupProfile` (a finer cup is a new profile, not a new driver).
"""

from __future__ import annotations

from .gripper import (
    ROBOTIQ_2F85_PROFILE,
    SCHUNK_EGU50_PROFILE,
    SCHUNK_EZU35_PROFILE,
    GripperProfile,
    IsaacGripper,
)
from .suction_gripper import (
    SLIM_SUCTION_CUP,
    STANDARD_SUCTION_CUP,
    IsaacSuctionGripper,
    SuctionCupProfile,
)

__all__ = [
    "ROBOTIQ_2F85_PROFILE",
    "SCHUNK_EGU50_PROFILE",
    "SCHUNK_EZU35_PROFILE",
    "GripperProfile",
    "IsaacGripper",
    "STANDARD_SUCTION_CUP",
    "SLIM_SUCTION_CUP",
    "SuctionCupProfile",
    "IsaacSuctionGripper",
]
