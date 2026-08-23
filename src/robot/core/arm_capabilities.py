"""Optional capability extensions for arm drivers.

Extras a particular controller *can* offer
digital / analog **I/O**, live **force /torque**,
live **robot / safety status** are declared here as SEPARATE
``runtime_checkable`` Protocols, exactly mirroring
:class:`~src.robot.core.gripper.ObjectDetectingGripper`.

A driver **opts in** by implementing a capability Protocol; callers feature-check
with ``isinstance(arm, SupportsForceTorque)`` and fall back gracefully when the
capability is absent.

Units + conventions mirror the rest of the ``robot`` layer: forces in **newtons**,
torques in **newton-metres**, every wrench frame-tagged (:class:`~src.geometry.Frame`).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from src.geometry import Frame

__all__ = [
    "DigitalIOPort",
    "RobotMode",
    "RobotStatus",
    "SafetyMode",
    "SupportsDigitalIO",
    "SupportsForceTorque",
    "SupportsRobotStatus",
    "Wrench",
]


class DigitalIOPort(StrEnum):
    """Which digital I/O bank a pin lives on."""

    STANDARD = "standard"
    CONFIGURABLE = "configurable"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class Wrench:
    """A 6-DoF force/torque reading.

    Forces (``fx, fy, fz``) are **newtons**, torques (``tx, ty, tz``) are
    **newton-metres**, expressed in ``frame``.
    """

    fx: float
    fy: float
    fz: float
    tx: float
    ty: float
    tz: float
    frame: Frame = Frame.BASE

    @property
    def force(self) -> tuple[float, float, float]:
        """The linear force component ``(fx, fy, fz)`` in newtons."""
        return (self.fx, self.fy, self.fz)

    @property
    def torque(self) -> tuple[float, float, float]:
        """The moment component ``(tx, ty, tz)`` in newton-metres."""
        return (self.tx, self.ty, self.tz)

    @property
    def force_magnitude(self) -> float:
        """Euclidean magnitude of the linear force (N), the primary hand-over signal."""
        return (self.fx * self.fx + self.fy * self.fy + self.fz * self.fz) ** 0.5


class RobotMode(StrEnum):
    """Vendor-neutral robot operating mode."""

    DISCONNECTED = "disconnected"
    CONFIRM_SAFETY = "confirm_safety"
    BOOTING = "booting"
    POWER_OFF = "power_off"
    POWER_ON = "power_on"
    IDLE = "idle"
    BACKDRIVE = "backdrive"
    RUNNING = "running"
    UPDATING_FIRMWARE = "updating_firmware"
    UNKNOWN = "unknown"


class SafetyMode(StrEnum):
    """Vendor-neutral safety mode."""

    NORMAL = "normal"
    REDUCED = "reduced"
    PROTECTIVE_STOP = "protective_stop"
    RECOVERY = "recovery"
    SAFEGUARD_STOP = "safeguard_stop"
    SYSTEM_EMERGENCY_STOP = "system_emergency_stop"
    ROBOT_EMERGENCY_STOP = "robot_emergency_stop"
    VIOLATION = "violation"
    FAULT = "fault"
    UNKNOWN = "unknown"

    @property
    def is_stopped(self) -> bool:
        """True for any mode where the arm is halted / faulted (not NORMAL / REDUCED / RECOVERY)."""
        return self in {
            SafetyMode.PROTECTIVE_STOP,
            SafetyMode.SAFEGUARD_STOP,
            SafetyMode.SYSTEM_EMERGENCY_STOP,
            SafetyMode.ROBOT_EMERGENCY_STOP,
            SafetyMode.VIOLATION,
            SafetyMode.FAULT,
        }


@dataclass(frozen=True, slots=True)
class RobotStatus:
    """A snapshot of the controller's live robot + safety state."""

    robot_mode: RobotMode
    safety_mode: SafetyMode
    protective_stopped: bool
    emergency_stopped: bool
    message: str = ""

    @property
    def is_operational(self) -> bool:
        """True only when the arm is powered, running, in NORMAL safety, and not stopped."""
        return (
            self.robot_mode == RobotMode.RUNNING
            and self.safety_mode == SafetyMode.NORMAL
            and not self.protective_stopped
            and not self.emergency_stopped
        )

    @property
    def is_stopped(self) -> bool:
        """True when a protective / emergency stop is active or the safety mode is a stop state."""
        return self.protective_stopped or self.emergency_stopped or self.safety_mode.is_stopped


@runtime_checkable
class SupportsDigitalIO(Protocol):
    """Capability extension: read/write the controller's digital + analog I/O."""

    def set_digital_output(
        self, pin: int, value: bool, *, port: DigitalIOPort = DigitalIOPort.STANDARD
    ) -> None:
        """Drive a digital output pin high (``True``) or low (``False``)."""
        ...

    def get_digital_input(self, pin: int, *, port: DigitalIOPort = DigitalIOPort.STANDARD) -> bool:
        """Read a digital input pin's logic level."""
        ...

    def get_digital_output(self, pin: int, *, port: DigitalIOPort = DigitalIOPort.STANDARD) -> bool:
        """Read back a digital output pin's commanded logic level."""
        ...

    def set_analog_output(self, pin: int, value: float, *, current: bool = False) -> None:
        """Set an analog output: ``value`` is a voltage (V) unless ``current=True`` (A)."""
        ...


@runtime_checkable
class SupportsForceTorque(Protocol):
    """Capability extension: read live TCP force/torque + per-joint torques."""

    def get_tcp_wrench(self) -> Wrench:
        """The current generalized force/torque at the TCP (newtons / newton-metres, BASE frame)."""
        ...

    def get_joint_torques(self) -> tuple[float, ...]:
        """The current torque at each joint (newton-metres), base-to-tool order."""
        ...


@runtime_checkable
class SupportsRobotStatus(Protocol):
    """Capability extension: read the live robot/safety state + recover a protective stop."""

    def get_robot_status(self) -> RobotStatus:
        """A snapshot of the controller's live robot mode + safety mode + stop flags."""
        ...

    def recover_from_protective_stop(self) -> bool:
        """Attempt to clear an active protective stop; ``True`` if the controller acknowledged."""
        ...
