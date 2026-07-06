"""
Vendor-neutral error hierarchy for the Workaholic-Willy robot subsystem.

Every driver MUST translate vendor-specific failures (``ur_rtde``
exceptions, ROS 2 service errors, ``franky`` faults, …) into one of
these types before re-raising. Application / pipeline code only ever
catches the abstract types defined here.

Hierarchy::

    RobotError
        ├── RobotConnectionError    (cannot reach controller, dropped link)
        ├── RobotKinematicsError    (FK / IK failure, unreachable target)
        ├── RobotMotionRejected     (workspace guard / safety pre-check denied)
        │       └── RobotSingularityRisk   (target too close to a singularity)
        └── RobotEmergencyStop      (e-stop / protective stop active)
"""

from __future__ import annotations

__all__ = [
    "RobotConnectionError",
    "RobotEmergencyStop",
    "RobotError",
    "RobotKinematicsError",
    "RobotMotionRejected",
    "RobotSingularityRisk",
]


class RobotError(Exception):
    """Base class for all vendor-neutral robot failures."""


class RobotConnectionError(RobotError):
    """Raised when the controller link cannot be opened or has dropped."""


class RobotKinematicsError(RobotError):
    """Raised when forward / inverse kinematics fail or a target is unreachable."""


class RobotMotionRejected(RobotError):
    """Raised when a motion request is rejected before reaching the driver.

    Typical causes: workspace-guard rejection, joint-limit pre-check
    failure, soft-limit violation, or input-validation error.
    """


class RobotSingularityRisk(RobotMotionRejected):
    """Raised when a target is rejected because singularity risk is too high."""


class RobotEmergencyStop(RobotError):
    """Raised when the controller reports an active e-stop / protective stop."""
