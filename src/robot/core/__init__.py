"""
Vendor-neutral core abstractions for the Workaholic-Willy robot subsystem.

This package defines the abstract surface every supported robot driver
(UR, KUKA, Franka, ROS 2, MoveIt, cuRobo, simulators) must satisfy.

Public exports:

* :class:`RobotArm`, :class:`Gripper`, :class:`ObjectDetectingGripper`
  the driver `Protocol`s.
* :class:`RobotVendor`, :class:`GripperVendor` canonical driver identifiers.
* :class:`JointPositions` typed joint-vector wrapper (radians).
* :class:`RobotCapabilities` declarative driver feature flags.
* :class:`MotionResult`, :class:`MotionStatus`, :class:`MotionCommand`
  the typed motion-outcome contract.
* :class:`RobotError` and subclasses vendor-neutral error hierarchy.

Numerics contract (mirrors :mod:`src.geometry`):

* Translations: **millimetres**, ``float64``.
* Orientations: unit XYZW quaternion, ``float64``, canonical sign.
* Joint angles: **radians**, ``float64``.
* All public ndarrays returned by this layer are read-only.
"""

from __future__ import annotations

from .arm_capabilities import (
    DigitalIOPort,
    RobotMode,
    RobotStatus,
    SafetyMode,
    SupportsDigitalIO,
    SupportsForceTorque,
    SupportsRobotStatus,
    Wrench,
)
from .capabilities import RobotCapabilities
from .errors import (
    IsaacNotAvailableError,
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
from .motion_result import (
    NO_PLAN_FAIL_SAFE_MESSAGE,
    MotionCommand,
    MotionResult,
    MotionStatus,
)
from .robot_arm import RobotArm
from .vendor import RobotVendor

__all__ = [
    "DigitalIOPort",
    "Gripper",
    "GripperVendor",
    "IsaacNotAvailableError",
    "JointPositions",
    "MotionCommand",
    "MotionResult",
    "MotionStatus",
    "NO_PLAN_FAIL_SAFE_MESSAGE",
    "ObjectDetectingGripper",
    "RobotArm",
    "RobotCapabilities",
    "RobotConnectionError",
    "RobotEmergencyStop",
    "RobotError",
    "RobotKinematicsError",
    "RobotMode",
    "RobotMotionRejected",
    "RobotSingularityRisk",
    "RobotStatus",
    "RobotVendor",
    "SafetyMode",
    "SupportsDigitalIO",
    "SupportsForceTorque",
    "SupportsRobotStatus",
    "Wrench",
]
