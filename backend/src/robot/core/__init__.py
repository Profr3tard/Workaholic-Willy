"""
Vendor-neutral core abstractions for the Workaholic-Willy robot subsystem.

This package defines the abstract surface every supported robot driver
(UR, KUKA, Franka, ROS 2, MoveIt, cuRobo, simulators) must satisfy.
It is the only piece of the ``robot`` package that pipelines, planners,
and the application layer are allowed to depend on directly.

Public exports:

* :class:`RobotArm`, :class:`Gripper`, :class:`ObjectDetectingGripper`
  — the driver `Protocol`s.
* :class:`RobotVendor`, :class:`GripperVendor` — canonical driver identifiers.
* :class:`JointPositions` — typed joint-vector wrapper (radians).
* :class:`RobotCapabilities` — declarative driver feature flags.
* :class:`MotionResult`, :class:`MotionStatus`, :class:`MotionCommand`
  — the typed motion-outcome contract.
* :class:`RobotError` and subclasses — vendor-neutral error hierarchy.

Numerics contract (mirrors :mod:`backend.src.geometry`):

* Translations: **millimetres**, ``float64``.
* Orientations: unit XYZW quaternion, ``float64``, canonical sign.
* Joint angles: **radians**, ``float64``.
* All public ndarrays returned by this layer are read-only.
"""

from __future__ import annotations

from .capabilities import RobotCapabilities
from .errors import (
    RobotConnectionError,
    RobotEmergencyStop,
    RobotError,
    RobotKinematicsError,
    RobotMotionRejected,
    RobotSingularityRisk,
)
from .gripper import Gripper, ObjectDetectingGripper
from .gripper_vendor import GripperVendor
from .joint_positions import JointPositions
from .motion_result import MotionCommand, MotionResult, MotionStatus
from .robot_arm import RobotArm
from .vendor import RobotVendor

__all__ = [
    "Gripper",
    "GripperVendor",
    "JointPositions",
    "MotionCommand",
    "MotionResult",
    "MotionStatus",
    "ObjectDetectingGripper",
    "RobotArm",
    "RobotCapabilities",
    "RobotConnectionError",
    "RobotEmergencyStop",
    "RobotError",
    "RobotKinematicsError",
    "RobotMotionRejected",
    "RobotSingularityRisk",
    "RobotVendor",
]
