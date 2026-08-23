"""Bench exerciser for a gripper wired to the UR controller's digital I/O.

    python -m src.robot.drivers.ur --read                      # read every pin, changes nothing
    python -m src.robot.drivers.ur --watch 0 --for 15          # trip a sensor by hand, see it
    python -m src.robot.drivers.ur --set 4=1 --yes             # drive one output
    python -m src.robot.drivers.ur --pulse 4 --for 0.2 --yes   # the double-solenoid shape
    python -m src.robot.drivers.ur --measure 4=1 --watch 0 --yes   # <- the number you came for

Exit codes: 0 the command ran - 1 config/connection refused - 2 the measurement timed out (the output
was driven and the input never answered) - 3 unexpected error.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from src.robot.constants import UR_IO_CLI_LOG_FILE, create_robot_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from config.schema.robot import RobotConfig

_EXIT_OK, _EXIT_REFUSED, _EXIT_TIMEOUT, _EXIT_ERROR = 0, 1, 2, 3

logger = create_robot_logger("URIOBenchCLI", UR_IO_CLI_LOG_FILE)


def _load_robot_config(profile: str | None, data_dir: str | None) -> "RobotConfig":
    """Load the config tree, honouring an explicit profile chain (mirrors ``real_cell``)."""
    import os

    from config.loader import load_config

    previous = os.environ.get("WILLY_PROFILE")
    if profile is not None:
        os.environ["WILLY_PROFILE"] = profile
    try:
        cfg = load_config(data_dir) if data_dir else load_config()
    finally:
        if profile is not None:
            if previous is None:
                os.environ.pop("WILLY_PROFILE", None)
            else:
                os.environ["WILLY_PROFILE"] = previous
    robot = getattr(cfg, "robot", None)
    if robot is None:
        raise SystemExit("the loaded config has no `robot` block")
    return robot


def _parse_pin_value(spec: str) -> tuple[int, bool]:
    """``"4=1"`` -> ``(4, True)``. Rejects anything ambiguous rather than picking a reading."""
    if "=" not in spec:
        raise SystemExit(f"--set/--measure want PIN=VALUE (e.g. 4=1), got {spec!r}")
    pin_s, val_s = spec.split("=", 1)
    try:
        pin = int(pin_s)
    except ValueError:
        raise SystemExit(f"--set/--measure: {pin_s!r} is not a pin number") from None
    val = val_s.strip().lower()
    if val in ("1", "true", "high", "on"):
        return pin, True
    if val in ("0", "false", "low", "off"):
        return pin, False
    raise SystemExit(f"--set/--measure: {val_s!r} is not a level (use 1/0, high/low, on/off)")


def _confirm(args: argparse.Namespace, what: str) -> bool:
    """Gate a physical write. ``--yes`` is the only way through in a non-interactive shell.

    Refusing rather than prompting when stdin is not a terminal is deliberate: a prompt that reads
    EOF and takes the default is how an unattended script drives a coil nobody authorised.
    """
    if args.yes:
        return True
    if not sys.stdin.isatty():
        print(
            f"REFUSED: {what} would drive a real output and --yes was not given (stdin is not a "
            "terminal, so there is nobody to ask). Re-run with --yes if the cell is clear.",
            flush=True,
        )
        logger.warning("refused (no --yes, stdin is not a terminal): %s", what)
        return False
    print(f"\n⚠  {what}")
    print("   This drives a real output: jaws close on whatever is between them, an ejector starts.")
    return input("   Is the cell clear? type 'yes' to proceed: ").strip().lower() == "yes"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m src.robot.drivers.ur",
        description="Bench exerciser for a gripper on the UR controller's digital I/O. Never moves "
                    "the arm; every write is gated behind --yes.",
    )
    ap.add_argument("--profile", type=str, default=None,
                    help="config profile chain, e.g. 'ur3e' or 'ur3e,tiltcam'")
    ap.add_argument("--data-dir", type=str, default=None, help="config data root")
    ap.add_argument("--port", choices=["standard", "configurable", "tool"], default="tool",
                    help="which I/O bank (a tool-mounted gripper is usually on TOOL)")
    ap.add_argument("--read", action="store_true", help="read every pin on the bank and exit")
    ap.add_argument("--watch", type=int, default=None, metavar="PIN",
                    help="poll an input and print every transition (read-only)")
    ap.add_argument("--set", type=str, default=None, metavar="PIN=VALUE",
                    help="drive one output, then read it back")
    ap.add_argument("--pulse", type=int, default=None, metavar="PIN",
                    help="drive an output high for --for seconds, then low")
    ap.add_argument("--measure", type=str, default=None, metavar="PIN=VALUE",
                    help="drive an output and time how long --watch's input takes to answer")
    ap.add_argument("--for", dest="duration", type=float, default=None,
                    help="seconds: --watch timeout (default 10), --pulse length (0.2), "
                         "--measure timeout (2)")
    ap.add_argument("--yes", action="store_true",
                    help="confirm that the cell is clear; required for any write")
    args = ap.parse_args(argv)

    actions = [bool(args.read), args.watch is not None, args.set is not None,
               args.pulse is not None, args.measure is not None]
    if not any(actions):
        ap.error("nothing to do: pass --read, --watch, --set, --pulse or --measure")
    if args.measure is not None and args.watch is None:
        ap.error("--measure needs --watch PIN: it times how long THAT input takes to answer")

    from src.robot.core.arm_capabilities import DigitalIOPort, SupportsDigitalIO

    from . import io_bench

    port = DigitalIOPort(args.port)

    logger.info(
        "bench session: port=%s profile=%s read=%s watch=%s set=%s pulse=%s measure=%s",
        port.value, args.profile, args.read, args.watch, args.set, args.pulse, args.measure,
    )

    try:
        robot_cfg = _load_robot_config(args.profile, args.data_dir)
    except SystemExit as exc:
        print(f"config refused: {exc}", flush=True)
        logger.error("config refused: %s", exc)
        return _EXIT_REFUSED

    try:
        from src.robot.drivers import create_arm
        from src.robot.core import RobotVendor

        vendor = RobotVendor.from_string(robot_cfg.vendor)
        if vendor is not RobotVendor.UR:
            print(f"REFUSED: robot.vendor is {vendor.value!r}. Digital I/O is a UR capability here, "
                  "only the UR driver advertises SupportsDigitalIO.", flush=True)
            logger.error("refused: robot.vendor is %r, not 'ur'", vendor.value)
            return _EXIT_REFUSED
        arm = create_arm(vendor, config=robot_cfg)
    except Exception as exc:  # noqa: BLE001 - a build failure is a refusal, not a crash
        print(f"could not build the arm: {type(exc).__name__}: {exc}", flush=True)
        logger.error("could not build the arm: %s: %s", type(exc).__name__, exc)
        return _EXIT_REFUSED

    if not isinstance(arm, SupportsDigitalIO):
        print("REFUSED: this arm does not advertise SupportsDigitalIO.", flush=True)
        logger.error("refused: %s does not advertise SupportsDigitalIO", type(arm).__name__)
        return _EXIT_REFUSED

    try:
        # connect() is where the UR driver fails closed on an undeclared tool frame and an incoherent
        # payload. Those checks are worth passing even for an I/O-only session: a cell whose TCP is
        # undeclared is not a cell anyone should be commissioning a gripper on.
        arm.connect()
    except Exception as exc:  # noqa: BLE001
        print(f"connect refused: {type(exc).__name__}: {exc}", flush=True)
        logger.error("connect refused: %s: %s", type(exc).__name__, exc)
        return _EXIT_REFUSED

    try:
        if args.read:
            print(io_bench.read_snapshot(arm, port=port).render(), flush=True)
            return _EXIT_OK

        if args.measure is not None:
            pin, value = _parse_pin_value(args.measure)
            if not _confirm(args, f"drive output {pin} -> {int(value)} on the {port.value} bank"):
                return _EXIT_REFUSED
            result = io_bench.measure_transition(
                arm, output_pin=pin, value=value, watched_input=int(args.watch),
                timeout_s=args.duration if args.duration is not None else 2.0, port=port,
            )
            print(result.render(), flush=True)
            if result.changed:
                print(f"\n-> use {result.elapsed_s:.3f} s as the FLOOR for close_settle_s / "
                      "engage_timeout_s. Repeat it a few times and configure the WORST reading: the "
                      "driver waits the budget out, so a typical value drops parts on a slow stroke.",
                      flush=True)
                return _EXIT_OK
            print("\n-> the output was driven and the input never answered. Check: the right bank "
                  "(--port), the right pin, sensor power, and whether the switch is active-LOW.",
                  flush=True)
            return _EXIT_TIMEOUT

        if args.set is not None:
            pin, value = _parse_pin_value(args.set)
            if not _confirm(args, f"drive output {pin} -> {int(value)} on the {port.value} bank"):
                return _EXIT_REFUSED
            read_back = io_bench.set_output(arm, pin, value, port=port)
            ok = "OK" if read_back == value else "MISMATCH"
            print(f"out {pin} := {int(value)} -> reads back {int(read_back)}  [{ok}]", flush=True)
            if read_back != value:
                print("-> the controller accepted the command and the pin did not move. Usually the "
                      "wrong bank: try --port standard / configurable / tool.", flush=True)
                return _EXIT_TIMEOUT
            return _EXIT_OK

        if args.pulse is not None:
            seconds = args.duration if args.duration is not None else 0.2
            if not _confirm(args, f"pulse output {args.pulse} for {seconds:.3f} s "
                                  f"on the {port.value} bank"):
                return _EXIT_REFUSED
            io_bench.pulse_output(arm, args.pulse, seconds=seconds, port=port)
            print(f"out {args.pulse} pulsed for {seconds:.3f} s (left LOW)", flush=True)
            return _EXIT_OK

        # --watch on its own: read-only, no gate.
        timeout = args.duration if args.duration is not None else 10.0
        print(f"watching input {args.watch} on {port.value} for {timeout:.1f} s — "
              "trip the sensor by hand now", flush=True)
        log = io_bench.watch_input(arm, int(args.watch), timeout_s=timeout, port=port)
        for at_s, level in log:
            print(f"  {at_s:7.3f} s  {int(level)}", flush=True)
        if len(log) == 1:
            print("-> one entry means nothing changed the whole time. Wrong pin, wrong bank, or the "
                  "sensor is unpowered.", flush=True)
        return _EXIT_OK
    except KeyboardInterrupt:
        print("\ninterrupted", flush=True)
        return _EXIT_ERROR
    finally:
        try:
            arm.disconnect()
        except Exception:  # noqa: BLE001 - teardown must not mask the result
            pass


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
