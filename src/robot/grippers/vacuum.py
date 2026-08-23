"""Vacuum (suction) gripper driven through the robot controller's digital I/O.

Willy could simulate suction, model its seal physics and score suction grasps, but a real cell could
not be told it HAD a suction gripper: :class:`GripperVendor` listed only jaw vendors, so the whole
suction stack was reachable in sim and unreachable on hardware.

**This driver is deliberately vendor-neutral, and that is the design, not a shortcut.** A suction
end-effector on a UR is an ejector or a pump wired to a controller output: assert the pin, vacuum
builds; drop it (usually with a short blow-off pulse) and the part releases.

What it needs is an object that can drive controller I/O **with** the :class:`SupportsDigitalIO` capability
the UR driver already advertises. Anything satisfying that Protocol works, including a fake in tests.

**Width semantics.** The :class:`Gripper` Protocol is width-based because jaws are. A cup has no
opening, so width is reinterpreted exactly as the simulated suction gripper already does keeping the
two honest about each other:

    set_width_mm(w)   w <= vacuum_on_below_mm  ->  vacuum ON  (engage)
                      w >  vacuum_on_below_mm  ->  vacuum OFF (release)

**The real payoff is feedback.** With a vacuum switch wired to an input, this driver implements
:class:`ObjectDetectingGripper`, which opts the cell into post-close verification in
``GraspExecutionPolicy``. A jaw gripper mostly has to be trusted; a suction cup can be ASKED whether
it is holding something. That is a better verification signal than anything the jaw path has.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from src.robot.core import RobotConnectionError
from src.robot.core.arm_capabilities import DigitalIOPort, SupportsDigitalIO

from ..constants import VACUUM_GRIPPER_LOG_FILE, create_robot_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from config.schema.robot.robot_schema import GripperConfig

__all__ = ["VacuumGripper"]


class VacuumGripper:
    """A vacuum end-effector actuated over the controller's digital I/O."""

    def __init__(
        self,
        io: SupportsDigitalIO,
        *,
        config: "GripperConfig | None" = None,
        vacuum_output_pin: int = 0,
        blow_off_output_pin: int | None = None,
        vacuum_ok_input_pin: int | None = None,
        io_port: DigitalIOPort | str = DigitalIOPort.TOOL,
        engage_timeout_s: float = 1.0,
        blow_off_s: float = 0.15,
        min_width_mm: float = 0.0,
        max_width_mm: float = 30.0,
        vacuum_on_below_mm: float = 5.0,
        sleep: Any = time.sleep,
    ) -> None:
        if not isinstance(io, SupportsDigitalIO):
            raise TypeError(
                "VacuumGripper needs a digital-I/O source (the arm driver on a real cell): "
                f"{type(io).__name__} does not satisfy SupportsDigitalIO."
            )
        self._io = io
        self._pin = int(vacuum_output_pin)
        self._blow_off_pin = None if blow_off_output_pin is None else int(blow_off_output_pin)
        self._ok_pin = None if vacuum_ok_input_pin is None else int(vacuum_ok_input_pin)
        self._port = DigitalIOPort(io_port) if not isinstance(io_port, DigitalIOPort) else io_port
        self._engage_timeout_s = float(engage_timeout_s)
        self._blow_off_s = float(blow_off_s)
        if config is not None:
            min_width_mm, max_width_mm = float(config.min_width_mm), float(config.max_width_mm)
        self._min_width_mm = float(min_width_mm)
        self._max_width_mm = float(max_width_mm)
        self._vacuum_on_below_mm = float(vacuum_on_below_mm)
        self._sleep = sleep
        self._connected = False
        self._vacuum_on = False
        self.logger = create_robot_logger("VacuumGripper", VACUUM_GRIPPER_LOG_FILE)

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
    def vacuum_on(self) -> bool:
        """Whether vacuum is currently commanded (the driver's own view, not the switch)."""
        return self._vacuum_on

    # -- lifecycle --------------------------------------------------------
    def connect(self) -> None:
        """Adopt the arm's I/O and command a known state: vacuum OFF.

        A cell that starts with the ejector latched on from a previous run would hold a part it does
        not know about, so the safe state is asserted rather than assumed.
        """
        self._connected = True
        self._set_vacuum(False)
        # The pin map is the half of this driver that is impossible to infer from a failure later:
        # "the cup never sealed" reads very differently once the log says which pins it was watching.
        self.logger.info(
            "connected on %s: vacuum pin=%d, ok input=%s, blow-off pin=%s; commanded OFF",
            self._port.value, self._pin,
            self._ok_pin if self._ok_pin is not None else "none",
            self._blow_off_pin if self._blow_off_pin is not None else "none",
        )

    def disconnect(self) -> None:
        """Release vacuum and detach. Never raises a teardown path must not strand a held part.

        Best-effort by design: if the I/O has already gone away there is nothing useful to do, and
        raising here would mask whatever actually tore the cell down.
        """
        try:
            if self._connected:
                self._set_vacuum(False)
        except Exception:  # noqa: BLE001 - teardown must not raise
            self.logger.warning("VacuumGripper.disconnect: releasing vacuum failed", exc_info=True)
        finally:
            self._connected = False
            self.logger.info("disconnected (vacuum released, I/O handed back to the arm)")

    def activate(self) -> None:
        """No calibration to run a cup has no jaws to home. Kept for Protocol parity."""
        self._require_connected("activate")

    # -- commands ---------------------------------------------------------
    def set_width_mm(
        self, width_mm: float, *, speed: float | None = None, force: float | None = None,
    ) -> None:
        """Reinterpret width as vacuum on/off (``<= vacuum_on_below_mm`` -> ON), matching the sim cup."""
        self._require_connected("set_width_mm")
        engage = float(width_mm) <= self._vacuum_on_below_mm
        self._set_vacuum(engage)
        if engage and self._ok_pin is not None:
            self._await_vacuum()

    def get_width_mm(self) -> float:
        """Reported opening: the closed band while engaged, the open band while released."""
        return self._min_width_mm if self._vacuum_on else self._max_width_mm

    def is_object_detected(self) -> bool:
        """``True`` when the vacuum switch says a part is held (:class:`ObjectDetectingGripper`)."""
        self._require_connected("is_object_detected")
        if self._ok_pin is None:
            return self._vacuum_on
        return bool(self._io.get_digital_input(self._ok_pin, port=self._port))

    # -- internals --------------------------------------------------------
    def _require_connected(self, what: str) -> None:
        if not self._connected:
            raise RobotConnectionError(f"VacuumGripper.{what} requires connect() first.")

    def _set_vacuum(self, on: bool) -> None:
        self.logger.debug("vacuum %s (pin=%d on %s)", "ON" if on else "OFF", self._pin, self._port.value)
        self._io.set_digital_output(self._pin, bool(on), port=self._port)
        # Releasing a suction grasp is not just "stop pulling": residual vacuum keeps a light part stuck
        # to the cup and it lets go somewhere unintended. A blow-off pulse pushes it off deliberately.
        if not on and self._blow_off_pin is not None:
            self._io.set_digital_output(self._blow_off_pin, True, port=self._port)
            self._sleep(self._blow_off_s)
            self._io.set_digital_output(self._blow_off_pin, False, port=self._port)
        self._vacuum_on = bool(on)

    def _await_vacuum(self) -> bool:
        """Poll the vacuum switch until it reports a seal or the timeout expires. Returns the verdict.

        A timeout is NOT an error here: a missed seal is a normal grasp outcome, and the verification
        stage is what decides. Raising would turn "the cup did not catch this one" into a crash.
        """
        started = time.monotonic()
        deadline = started + self._engage_timeout_s
        while True:
            if bool(self._io.get_digital_input(self._ok_pin, port=self._port)):  # type: ignore[arg-type]
                # How long the ejector took to build is the number engage_timeout_s is budgeted from,
                # and on a real cell it drifts as the cup wears.
                self.logger.info(
                    "vacuum switch confirmed a seal after %.3f s", time.monotonic() - started,
                )
                return True
            if time.monotonic() >= deadline:
                self.logger.info(
                    "vacuum did not reach the switch threshold within %.2f s -- treating as no seal "
                    "(a missed grasp, not a fault)", self._engage_timeout_s,
                )
                return False
            self._sleep(0.02)
