"""
:class:`RobotVendor` canonical identifiers for arm-driver vendors.

Used by:

* :class:`config.schema.robot.RobotConfig` (operators select a driver via the
  ``vendor`` field, validated against THIS enum at config-load).
* :mod:`src.robot.drivers.registry` (driver factories register under one of
  these identifiers).
* :class:`src.robot.core.RobotCapabilities`, its ``vendor`` field is deliberately
  FREE-FORM (lowercase/whitespace-validated only, NOT enum-checked) so a custom-rig
  capability descriptor can name a driver outside this enum. Enum membership is enforced at
  the config layer, not here.

Adding a new vendor is a deliberate, two-line change here PLUS a driver
package under ``src/robot/drivers/<name>/``. Pipelines never
read this enum directly they go through the
:class:`~src.robot.core.RobotArm` Protocol.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["RobotVendor"]


class RobotVendor(StrEnum):
    """Canonical vendor identifiers for arm drivers.

    Values are lowercase, whitespace-free strings so they round-trip
    cleanly through YAML / JSON config without quoting issues.
    """

    UR = "ur"
    KUKA = "kuka"
    FRANKA = "franka"
    ROS2 = "ros2"
    SIM = "sim"
    DUMMY = "dummy"

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"RobotVendor.{self.name}"

    @classmethod
    def from_string(cls, value: str) -> RobotVendor:
        """Coerce a free-form string (case-insensitive) into a member.

        Raises :class:`ValueError` for unknown vendors with a list of
        valid options in the message.
        """
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise ValueError(
                f"RobotVendor expects a string, got {type(value).__name__}"
            )
        normalised = value.strip().lower()
        try:
            return cls(normalised)
        except ValueError as exc:
            valid = ", ".join(v.value for v in cls)
            raise ValueError(
                f"unknown robot vendor {value!r}; valid: {valid}"
            ) from exc
