"""
KUKA arm driver, production EthernetKRL (EKI) implementation.

Public surface
--------------
* :class:`KukaRobotArm` :class:`src.robot.core.RobotArm`
  implementation talking to the KUKA controller over TCP/XML.
* :class:`EkiClient` the underlying transport, exposed for advanced
  callers (e.g. integration tests that drive a mock socket).
* :class:`KukaCartesian` KUKA E6POS-equivalent value object.

The KRL / EKI XML side templates live under
``config/data/robot/templates/kuka/``.
"""

from __future__ import annotations

from .arm import KUKA_CAPABILITIES, KukaRobotArm
from .eki_client import EkiClient
from .pose_convert import (
    KukaCartesian,
    joints_deg_to_rad,
    joints_rad_to_deg,
    kuka_cartesian_to_pose,
    pose_to_kuka_cartesian,
)

__all__ = [
    "KUKA_CAPABILITIES",
    "EkiClient",
    "KukaCartesian",
    "KukaRobotArm",
    "joints_deg_to_rad",
    "joints_rad_to_deg",
    "kuka_cartesian_to_pose",
    "pose_to_kuka_cartesian",
]
