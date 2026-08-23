"""Bench measurements for a gripper wired to the controller's digital I/O.

Every field on ``VacuumGripperConfig`` and ``JawIOGripperConfig`` is a number somebody has to MEASURE
on the actual cell: which pin closes the jaws, whether the reed switch is active-high, how long the
cylinder takes to travel, how long the ejector needs to build a seal. That design is what lets the
I/O end-effectors be built before the hardware exists, but it only pays off if those numbers are
cheap to obtain. Without a tool, "which pin is it?" is answered by editing a YAML, running a pick,
watching it fail, and guessing again.

Pure by construction: everything takes a :class:`SupportsDigitalIO` plus injected ``clock``/``sleep``
seams, so a test can drive it deterministically and a bench run gets the real thing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from src.robot.constants import UR_IO_BENCH_LOG_FILE, create_robot_logger
from src.robot.core.arm_capabilities import DigitalIOPort, SupportsDigitalIO

__all__ = [
    "IOSnapshot",
    "TransitionResult",
    "measure_transition",
    "pulse_output",
    "read_snapshot",
    "set_output",
    "watch_input",
]

#: UR standard/configurable/tool banks all expose 8 pins at most; the tool bank has 2.
_PIN_RANGE = range(8)

logger = create_robot_logger("URIOBench", UR_IO_BENCH_LOG_FILE)


@dataclass(frozen=True, slots=True)
class IOSnapshot:
    """Every readable pin on one bank, at one instant."""

    port: DigitalIOPort
    inputs: dict[int, bool] = field(default_factory=dict)
    outputs: dict[int, bool] = field(default_factory=dict)

    def render(self) -> str:
        """A table an operator can read off the bench, aligned so a changed bit is obvious."""
        pins = sorted(set(_PIN_RANGE) | set(self.inputs) | set(self.outputs))
        head = f"port={self.port.value:<13} " + " ".join(f"{p:>3}" for p in pins)
        ins = "  in " + " " * 11 + " ".join(_bit(self.inputs.get(p)) for p in pins)
        outs = "  out" + " " * 11 + " ".join(_bit(self.outputs.get(p)) for p in pins)
        return "\n".join((head, ins, outs))


def _bit(value: bool | None) -> str:
    return "  -" if value is None else ("  1" if value else "  0")


@dataclass(frozen=True, slots=True)
class TransitionResult:
    """What a commanded output did to a watched input, and how long it took.

    ``elapsed_s`` is the number that goes into ``close_settle_s`` / ``engage_timeout_s``. Measure it
    a few times and take the worst, not the best: the config field is a budget, not a typical value.
    """

    pin: int
    watched_input: int
    #: The input level BEFORE the output was driven.
    before: bool
    #: The input level when the wait ended, either because it changed or because time ran out.
    after: bool
    elapsed_s: float
    changed: bool
    timed_out: bool

    def render(self) -> str:
        verdict = (
            f"changed {int(self.before)} -> {int(self.after)} after {self.elapsed_s * 1000:.0f} ms"
            if self.changed
            else f"NO CHANGE (stayed {int(self.before)}) after {self.elapsed_s * 1000:.0f} ms"
        )
        note = "  [TIMED OUT]" if self.timed_out else ""
        return f"out {self.pin} -> in {self.watched_input}: {verdict}{note}"


def read_snapshot(
    io: SupportsDigitalIO, *, port: DigitalIOPort = DigitalIOPort.STANDARD
) -> IOSnapshot:
    """Read every pin on one bank. Read-only safe to run on a live cell at any time.

    A pin the controller does not expose raises rather than returning a level, so it is recorded as
    absent (``-``) instead of as a confident ``0``.
    """
    inputs: dict[int, bool] = {}
    outputs: dict[int, bool] = {}
    for pin in _PIN_RANGE:
        try:
            inputs[pin] = bool(io.get_digital_input(pin, port=port))
        except Exception:  # noqa: BLE001 - an unexposed pin is a fact, not a failure
            pass
        try:
            outputs[pin] = bool(io.get_digital_output(pin, port=port))
        except Exception:  # noqa: BLE001
            pass
    logger.info(
        "read %s bank: %d input(s), %d output(s) answered of %d pins probed",
        port.value, len(inputs), len(outputs), len(_PIN_RANGE),
    )
    return IOSnapshot(port=port, inputs=inputs, outputs=outputs)


def set_output(
    io: SupportsDigitalIO,
    pin: int,
    value: bool,
    *,
    port: DigitalIOPort = DigitalIOPort.STANDARD,
    settle_timeout_s: float = 0.5,
    poll_s: float = 0.01,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Any] = time.sleep,
) -> bool:
    """Drive one output and READ IT BACK. Returns the read-back level."""
    io.set_digital_output(int(pin), bool(value), port=port)
    # SETTLE BEFORE READING, because the read-back is the whole point and it was WRONG without this.
    deadline = clock() + float(settle_timeout_s)
    read = bool(io.get_digital_output(int(pin), port=port))
    while read != bool(value) and clock() < deadline:
        sleep(float(poll_s))
        read = bool(io.get_digital_output(int(pin), port=port))
    if read == bool(value):
        logger.info("drove %s output %d := %d, read back %d", port.value, int(pin), int(bool(value)), int(read))
    else:
        logger.warning(
            "drove %s output %d := %d but it reads back %d wrong bank, a reserved pin, or nothing "
            "wired there",
            port.value, int(pin), int(bool(value)), int(read),
        )
    return read


def pulse_output(
    io: SupportsDigitalIO,
    pin: int,
    *,
    seconds: float = 0.2,
    port: DigitalIOPort = DigitalIOPort.STANDARD,
    sleep: Callable[[float], Any] = time.sleep,
) -> None:
    """Drive an output high for ``seconds``, then low. The double-solenoid / blow-off shape.

    Always drops the pin, including when the sleep is interrupted: a latching coil left energised is
    the failure this exists to avoid, and a Ctrl-C mid-pulse is exactly when it would happen.
    """
    logger.info("pulsing %s output %d high for %.3f s", port.value, int(pin), float(seconds))
    io.set_digital_output(int(pin), True, port=port)
    try:
        sleep(float(seconds))
    finally:
        io.set_digital_output(int(pin), False, port=port)


def watch_input(
    io: SupportsDigitalIO,
    pin: int,
    *,
    timeout_s: float = 10.0,
    port: DigitalIOPort = DigitalIOPort.STANDARD,
    poll_s: float = 0.01,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Any] = time.sleep,
) -> list[tuple[float, bool]]:
    """Poll one input and record every transition as ``(seconds_since_start, level)``."""
    start = clock()
    level = bool(io.get_digital_input(int(pin), port=port))
    log: list[tuple[float, bool]] = [(0.0, level)]
    while clock() - start < float(timeout_s):
        sleep(float(poll_s))
        now = bool(io.get_digital_input(int(pin), port=port))
        if now != level:
            level = now
            log.append((clock() - start, level))
    # One line for the whole watch, not one per transition: the transitions are the caller's return
    # value, and a bouncing switch would otherwise flood the file.
    logger.info(
        "watched %s input %d for %.1f s: %d transition(s) after the initial %d",
        port.value, int(pin), float(timeout_s), len(log) - 1, int(log[0][1]),
    )
    return log


def measure_transition(
    io: SupportsDigitalIO,
    *,
    output_pin: int,
    value: bool,
    watched_input: int,
    timeout_s: float = 2.0,
    port: DigitalIOPort = DigitalIOPort.STANDARD,
    poll_s: float = 0.005,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Any] = time.sleep,
) -> TransitionResult:
    """Drive an output, then time how long the watched input takes to respond. THE bench measurement."""
    before = bool(io.get_digital_input(int(watched_input), port=port))
    start = clock()
    io.set_digital_output(int(output_pin), bool(value), port=port)
    while True:
        now = bool(io.get_digital_input(int(watched_input), port=port))
        elapsed = clock() - start
        if now != before:
            # THE bench number: this is what becomes close_settle_s / engage_timeout_s.
            logger.info(
                "%s out %d := %d -> in %d changed %d -> %d after %.0f ms",
                port.value, int(output_pin), int(bool(value)), int(watched_input),
                int(before), int(now), elapsed * 1000.0,
            )
            return TransitionResult(
                pin=int(output_pin), watched_input=int(watched_input), before=before,
                after=now, elapsed_s=elapsed, changed=True, timed_out=False,
            )
        if elapsed >= float(timeout_s):
            logger.warning(
                "%s out %d := %d -> in %d never answered within %.1f s (stayed %d)",
                port.value, int(output_pin), int(bool(value)), int(watched_input),
                float(timeout_s), int(before),
            )
            return TransitionResult(
                pin=int(output_pin), watched_input=int(watched_input), before=before,
                after=now, elapsed_s=elapsed, changed=False, timed_out=True,
            )
        sleep(float(poll_s))
