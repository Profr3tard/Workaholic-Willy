"""
:class:`GripperVendor` canonical identifiers for gripper drivers.

Mirrors :class:`RobotVendor`: every gripper driver under
:mod:`src.robot.grippers` registers a factory under one of
these enum members. Pipelines never read this enum; they go through
the :class:`~src.robot.core.Gripper` Protocol.

Adding a new gripper is a two-line change here PLUS a driver module
under ``src/robot/grippers/<name>.py`` (or subpackage).
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["GripperVendor"]


class GripperVendor(StrEnum):
    """Canonical vendor identifiers for gripper drivers.

    Values are lowercase, whitespace-free strings so they round-trip
    cleanly through YAML / JSON config without quoting issues.

    Members
    -------
    ROBOTIQ
        Robotiq HE / HE-X over the SDU ``robotiq_gripper`` driver.
    FRANKA_HAND
        Franka Hand (libfranka). Reserved for a future driver.
    SCHUNK
        Schunk EGK / EGN. Reserved.
    VACUUM
        A suction end-effector actuated over the controller's digital I/O.
    JAW_IO
        A parallel-jaw gripper actuated over the controller's digital I/O.
    DUMMY
        Pure-Python sim gripper for tests / offline development.
    NONE
        Explicit "no gripper attached"
    """

    ROBOTIQ = "robotiq"
    FRANKA_HAND = "franka_hand"
    SCHUNK = "schunk"
    VACUUM = "vacuum"
    JAW_IO = "jaw_io"
    DUMMY = "dummy"
    NONE = "none"

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"GripperVendor.{self.name}"

    @classmethod
    def from_string(cls, value: str) -> GripperVendor:
        """Coerce a free-form string (case-insensitive) into a member.

        Raises :class:`ValueError` for unknown vendors with a list of
        valid options in the message.
        """
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise ValueError(
                f"GripperVendor expects a string, got {type(value).__name__}"
            )
        normalised = value.strip().lower()
        try:
            return cls(normalised)
        except ValueError as exc:
            valid = ", ".join(v.value for v in cls)
            raise ValueError(
                f"unknown gripper vendor {value!r}; valid: {valid}"
            ) from exc
