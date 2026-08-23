"""Parallel-jaw gripper driven through the robot controller's digital I/O.

The vendor-neutral twin of :mod:`.vacuum`, written for the same reason and on the same rule: a
pneumatic or electric two-finger gripper on a UR is one or two output pins plus, usually, reed
switches on inputs.

Distinct from :mod:`.robotiq`, which is also jaws: the Robotiq speaks a socket protocol on port
63352 that the URCap opens, not I/O. Both can be mounted on the same cell; only the ``vendor`` string
changes.

**Width semantics.** The :class:`Gripper` Protocol is width-based because jaws travel. A digital-I/O
jaw does not: it is OPEN or CLOSED and nothing in between is commandable. Width is therefore
reinterpreted exactly as the suction driver reinterprets it, which keeps the two I/O end-effectors
honest about each other::

    set_width_mm(w)   w <= closed_below_mm  ->  CLOSE
                      w >  closed_below_mm  ->  OPEN

``speed`` and ``force`` are accepted for Protocol parity and ignored as a solenoid has one setting.
Regulating either means a pressure regulator or a drive, neither of which is on a digital pin.

**The real payoff is feedback, and specifically TWO reed switches.** With a fully-open and a
fully-closed switch the driver can tell apart the two outcomes a jaw cell has never been able to
distinguish:

===================  ==================  ===========================================================
``open_confirm``     ``closed_confirm``  meaning
===================  ==================  ===========================================================
True                 False               jaws fully open
False                True                fully closed **the jaws met each other: EMPTY GRASP**
False                False               stopped in between **something is between them: HOLDING**
True                 True                contradictory wiring/sensor fail closed, report no object
===================  ==================  ===========================================================
"""

from __future__ import annotations

import time
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from src.robot.core import RobotConnectionError
from src.robot.core.arm_capabilities import DigitalIOPort, SupportsDigitalIO

from ..constants import JAW_IO_GRIPPER_LOG_FILE, create_robot_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from config.schema.robot.robot_schema import GripperConfig

__all__ = ["JawIOGripper", "JawState"]


class JawState(StrEnum):
    """What the wired feedback says the jaws are doing."""

    OPEN = "open"
    #: Fully closed after a close command: the jaws met, so nothing was between them.
    CLOSED_EMPTY = "closed_empty"
    #: Stopped between the end stops: something is between the jaws.
    HOLDING = "holding"
    #: Both end-stop switches active at once impossible mechanically, so it is the wiring or a
    #: failed sensor. Treated as "no object" (fail closed), never as a successful grasp.
    CONTRADICTORY = "contradictory"
    UNKNOWN = "unknown"


class JawIOGripper:
    """A parallel-jaw gripper actuated over the controller's digital I/O."""

    def __init__(
        self,
        io: SupportsDigitalIO,
        *,
        config: "GripperConfig | None" = None,
        actuation: str = "single_solenoid",
        close_output_pin: int = 0,
        open_output_pin: int | None = None,
        pulse_s: float = 0.2,
        part_present_input_pin: int | None = None,
        closed_confirm_input_pin: int | None = None,
        open_confirm_input_pin: int | None = None,
        io_port: DigitalIOPort | str = DigitalIOPort.TOOL,
        close_timeout_s: float = 1.0,
        close_settle_s: float = 0.3,
        closed_below_mm: float = 5.0,
        open_on_connect_without_feedback: bool = False,
        min_width_mm: float = 0.0,
        max_width_mm: float = 85.0,
        sleep: Any = time.sleep,
    ) -> None:
        if not isinstance(io, SupportsDigitalIO):
            raise TypeError(
                "JawIOGripper needs a digital-I/O source (the arm driver on a real cell): "
                f"{type(io).__name__} does not satisfy SupportsDigitalIO."
            )
        if actuation not in ("single_solenoid", "double_solenoid"):
            raise ValueError(
                f"JawIOGripper: unknown actuation {actuation!r} expected 'single_solenoid' "
                "(one output, spring return) or 'double_solenoid' (two pulsed outputs, bistable)."
            )
        if actuation == "double_solenoid" and open_output_pin is None:
            raise ValueError(
                "JawIOGripper: actuation 'double_solenoid' needs open_output_pin a bistable valve "
                "has no spring to open it."
            )
        if actuation == "single_solenoid" and open_output_pin is not None:
            # A single solenoid opens by DROPPING the close pin, so a second pin would never be
            # driven.
            raise ValueError(
                "JawIOGripper: actuation 'single_solenoid' opens by dropping close_output_pin, so "
                "open_output_pin is never driven. Drop it, or use 'double_solenoid'."
            )
        if open_output_pin is not None and int(open_output_pin) == int(close_output_pin):
            raise ValueError(
                f"JawIOGripper: close_output_pin and open_output_pin are both {close_output_pin} -- "
                "one pin cannot drive both coils."
            )
        self._io = io
        self._actuation = actuation
        self._close_pin = int(close_output_pin)
        self._open_pin = None if open_output_pin is None else int(open_output_pin)
        self._pulse_s = float(pulse_s)
        self._part_pin = None if part_present_input_pin is None else int(part_present_input_pin)
        self._closed_pin = (
            None if closed_confirm_input_pin is None else int(closed_confirm_input_pin)
        )
        self._open_confirm_pin = (
            None if open_confirm_input_pin is None else int(open_confirm_input_pin)
        )
        self._port = DigitalIOPort(io_port) if not isinstance(io_port, DigitalIOPort) else io_port
        self._close_timeout_s = float(close_timeout_s)
        self._close_settle_s = float(close_settle_s)
        self._closed_below_mm = float(closed_below_mm)
        self._open_on_connect_without_feedback = bool(open_on_connect_without_feedback)
        if config is not None:
            min_width_mm, max_width_mm = float(config.min_width_mm), float(config.max_width_mm)
        self._min_width_mm = float(min_width_mm)
        self._max_width_mm = float(max_width_mm)
        self._sleep = sleep
        self._connected = False
        # What the driver last COMMANDED not what the jaws are doing. Kept apart from the sensed
        # state on purpose: conflating the two is how a gripper reports a grasp it never made.
        self._closed = False
        self.logger = create_robot_logger("JawIOGripper", JAW_IO_GRIPPER_LOG_FILE)

    # -- state ------------------------------------------------------------
    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def min_width_mm(self) -> float:
        return self._min_width_mm

    @property
    def max_width_mm(self) -> float:
        return self._max_width_mm

    @property
    def has_feedback(self) -> bool:
        """Whether ANY feedback pin is wired. False means every verdict is a commanded state."""
        return any(
            pin is not None
            for pin in (self._part_pin, self._closed_pin, self._open_confirm_pin)
        )

    @property
    def jaws_closed(self) -> bool:
        """Whether the jaws are currently COMMANDED closed (the driver's view, not the sensors')."""
        return self._closed

    # -- lifecycle --------------------------------------------------------
    def connect(self) -> None:
        """Adopt the arm's I/O and reach a safe state WITHOUT blindly dropping a workpiece."""
        self._connected = True
        state = self._read_state()
        if state is JawState.HOLDING:
            self.logger.warning(
                "connect(): the jaws report a part between them (%s). NOT opening releasing here "
                "would drop it wherever the arm is standing. Clear it before running.", state.value,
            )
            self._closed = True
            return
        if state is JawState.CONTRADICTORY:
            self.logger.warning(
                "connect(): both end-stop switches read active, which is mechanically impossible "
                "check the wiring / sensors. NOT actuating on an unreadable state.",
            )
            return
        if state is JawState.UNKNOWN and not self._open_on_connect_without_feedback:
            self.logger.info(
                "connect(): no feedback wired, so the jaws cannot be proven empty not actuating. "
                "Set open_on_connect_without_feedback=true to assert an open state instead.",
            )
            return
        self._actuate(close=False)
        # Which pins, and what the jaws read before being opened the two facts that make a later
        # "it dropped the part on connect" report answerable instead of arguable.
        self.logger.info(
            "connected on %s: close pin=%s, open pin=%s, feedback=%s; jaws read %s -> opened",
            self._port.value, self._close_pin,
            self._open_pin if self._open_pin is not None else "none",
            "wired" if self.has_feedback else "none", state.value,
        )

    def disconnect(self) -> None:
        """Detach WITHOUT releasing. Never raises."""
        self._connected = False
        # Says WHAT WAS LEFT BEHIND, because this driver deliberately does not release: a cell found
        # holding a part next morning should be explained by this line.
        self.logger.info(
            "disconnected without releasing (jaws left %s)", "CLOSED" if self._closed else "open",
        )

    def activate(self) -> None:
        """No homing to run a solenoid jaw has no calibration. Kept for Protocol parity."""
        self._require_connected("activate")

    # -- commands ---------------------------------------------------------
    def set_width_mm(
        self, width_mm: float, *, speed: float | None = None, force: float | None = None,
    ) -> None:
        """Reinterpret width as jaws open/closed (``<= closed_below_mm`` -> CLOSE)."""
        self._require_connected("set_width_mm")
        close = float(width_mm) <= self._closed_below_mm
        self._actuate(close=close)
        if close:
            self._await_close()

    def get_width_mm(self) -> float:
        """Reported opening: the closed band while closed, the open band while open."""
        return self._min_width_mm if self._closed else self._max_width_mm

    def is_object_detected(self) -> bool:
        """``True`` when the wired feedback says a part is between the jaws."""
        self._require_connected("is_object_detected")
        if not self._closed:
            # Nothing can be held by open jaws, and the reed pair reads "between the end stops"
            # during travel, which would otherwise look exactly like a grasp.
            return False
        if self._part_pin is not None:
            return bool(self._io.get_digital_input(self._part_pin, port=self._port))
        state = self._read_state()
        if state is JawState.UNKNOWN:
            return self._closed
        return state is JawState.HOLDING

    def read_jaw_state(self) -> JawState:
        """The sensed jaw state, for operators and bring-up. Not part of any Protocol."""
        self._require_connected("read_jaw_state")
        return self._read_state()

    # -- internals --------------------------------------------------------
    def _require_connected(self, what: str) -> None:
        if not self._connected:
            raise RobotConnectionError(f"JawIOGripper.{what} requires connect() first.")

    def _actuate(self, *, close: bool) -> None:
        """Drive the valve. Single solenoid holds a level; double solenoid pulses the right coil."""
        self.logger.debug("actuating jaws %s via %s", "CLOSED" if close else "OPEN", self._actuation)
        if self._actuation == "single_solenoid":
            self._io.set_digital_output(self._close_pin, bool(close), port=self._port)
        else:
            # A bistable valve LATCHES, so the coil is energised only long enough to throw it 
            # holding it high is what cooks the coil.
            pin = self._close_pin if close else self._open_pin
            other = self._open_pin if close else self._close_pin
            assert pin is not None and other is not None  # narrowed by the constructor's refusal
            self._io.set_digital_output(other, False, port=self._port)
            self._io.set_digital_output(pin, True, port=self._port)
            self._sleep(self._pulse_s)
            self._io.set_digital_output(pin, False, port=self._port)
        self._closed = bool(close)

    def _read_state(self) -> JawState:
        """Classify the jaws from whichever inputs are wired. Never raises, never guesses."""
        closed_confirm = (
            None if self._closed_pin is None
            else bool(self._io.get_digital_input(self._closed_pin, port=self._port))
        )
        open_confirm = (
            None if self._open_confirm_pin is None
            else bool(self._io.get_digital_input(self._open_confirm_pin, port=self._port))
        )
        if closed_confirm is not None and open_confirm is not None:
            if closed_confirm and open_confirm:
                return JawState.CONTRADICTORY
            if open_confirm:
                return JawState.OPEN
            if closed_confirm:
                return JawState.CLOSED_EMPTY
            return JawState.HOLDING
        if closed_confirm is not None:
            # One switch, on the closed end stop.
            if closed_confirm:
                return JawState.CLOSED_EMPTY
            return JawState.HOLDING if self._closed else JawState.OPEN
        if open_confirm is not None:
            # One switch, on the OPEN end stop. It can prove "open" and nothing else: off the stop,
            # "holding a part" and "closed on nothing" are the same reading.
            return JawState.OPEN if open_confirm else JawState.UNKNOWN
        if self._part_pin is not None:
            if bool(self._io.get_digital_input(self._part_pin, port=self._port)):
                return JawState.HOLDING
            return JawState.CLOSED_EMPTY if self._closed else JawState.OPEN
        return JawState.UNKNOWN

    def _await_close(self) -> JawState:
        """Wait for the jaws to settle after a close. Returns the verdict; a timeout is not an error."""
        if not self.has_feedback:
            # Nothing to poll. The cylinder still needs its travel time, and that wait is the only
            # thing separating "the valve was commanded" from "the part is gripped".
            self._sleep(self._close_settle_s)
            return JawState.UNKNOWN
        started = time.monotonic()
        deadline = started + self._close_timeout_s
        while True:
            state = self._read_state()
            if state in (JawState.HOLDING, JawState.CLOSED_EMPTY):
                # HOLDING vs CLOSED_EMPTY is the grasp verdict, and the travel time is what
                # close_settle_s is budgeted from both worth having per attempt.
                self.logger.info(
                    "jaws settled: %s after %.3f s", state.value, time.monotonic() - started,
                )
                return state
            if time.monotonic() >= deadline:
                self.logger.info(
                    "jaws did not reach a settled state within %.2f s (last read: %s) treating as "
                    "no grasp (a missed grasp, not a fault)", self._close_timeout_s, state.value,
                )
                return state
            self._sleep(0.02)
