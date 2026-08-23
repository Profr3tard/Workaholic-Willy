"""Driver readiness doctor.

Optional vendor SDKs (``ur_rtde`` for UR, ``isaacsim`` for the sim, ``robotiq_gripper`` for the Robotiq
gripper) are imported LAZILY inside the driver factories / at ``connect()``, so a missing SDK fails
LATE with a cryptic error. This module probes which vendor SDKs are importable up-front and reports a
per-vendor readiness table, and exposes :func:`require_arm_vendor_ready`, a fail-early startup gate the
composition root calls so a misconfigured host fails with a clear "host not ready for vendor X" message
instead of deep inside ``connect()``.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

from src.robot.constants import DRIVER_DOCTOR_LOG_FILE, create_robot_logger
from src.robot.core.errors import RobotConnectionError
from src.robot.core.gripper_vendor import GripperVendor
from src.robot.core.vendor import RobotVendor

__all__ = [
    "SdkStatus",
    "VendorReadiness",
    "arm_vendor_readiness",
    "gripper_vendor_readiness",
    "probe_module",
    "readiness_table",
    "require_arm_vendor_ready",
]

# Vendor -> required importable SDK module(s). Empty tuple = no third-party SDK (always importable).
# KUKA speaks EKI/KRL over a plain TCP socket
# DUMMY/NONE are pure-Python.
_ARM_VENDOR_SDKS: dict[RobotVendor, tuple[str, ...]] = {
    RobotVendor.UR: ("rtde_control", "rtde_receive"),
    RobotVendor.KUKA: (),
    RobotVendor.SIM: ("isaacsim",),
    RobotVendor.DUMMY: (),
}
_GRIPPER_VENDOR_SDKS: dict[GripperVendor, tuple[str, ...]] = {
    GripperVendor.ROBOTIQ: ("robotiq_gripper",),
    GripperVendor.DUMMY: (),
    GripperVendor.NONE: (),
    # Vacuum ships a real driver (grippers/vacuum.py) and needs NO SDK -- it drives the arm's digital
    # I/O. Omitting it made the doctor report "no driver registered" for a driver that exists, which in a
    # repo that prizes honest logs is a worse bug than the missing capability would be.
    GripperVendor.VACUUM: (),
}
# Vendors whose SDK need NOT be present when running in mock mode (no real device is driven).
_MOCKABLE_ARM_VENDORS: frozenset[RobotVendor] = frozenset({RobotVendor.SIM, RobotVendor.DUMMY})

# The gate below is the LAST cheap moment before a cryptic late failure, so what it saw is worth
# keeping: a bring-up that dies inside connect() is diagnosed from "the doctor passed with these SDK
# versions" far faster than from the traceback alone.
logger = create_robot_logger("DriverDoctor", DRIVER_DOCTOR_LOG_FILE)


@dataclass(frozen=True, slots=True)
class SdkStatus:
    """Whether one SDK module is importable (+ its version when discoverable)."""

    module: str
    importable: bool
    version: str | None


@dataclass(frozen=True, slots=True)
class VendorReadiness:
    """Readiness of one vendor: registered as a driver AND all its SDK modules importable."""

    vendor: str
    kind: str  # "arm" | "gripper"
    registered: bool
    sdks: tuple[SdkStatus, ...]
    ready: bool
    note: str


def probe_module(module: str) -> SdkStatus:
    """Best-effort: is ``module`` importable? (no import side effects, uses find_spec)."""
    importable = False
    try:
        importable = importlib.util.find_spec(module) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        importable = False
    version: str | None = None
    if importable:
        try:
            from importlib.metadata import PackageNotFoundError, version as _pkg_version

            try:
                version = _pkg_version(module)
            except PackageNotFoundError:
                version = None
        except Exception:  # noqa: BLE001 - version detection is best-effort only
            version = None
    return SdkStatus(module=module, importable=importable, version=version)


def _is_arm_registered(vendor: RobotVendor) -> bool:
    from src.robot.drivers.registry import is_vendor_registered

    return bool(is_vendor_registered(vendor.value))


def _readiness(vendor_value: str, kind: str, sdk_modules: tuple[str, ...], registered: bool) -> VendorReadiness:
    sdks = tuple(probe_module(m) for m in sdk_modules)
    sdks_ok = all(s.importable for s in sdks)
    ready = registered and sdks_ok
    if not registered:
        note = "no driver registered (reserved slot / not implemented)"
    elif not sdks_ok:
        missing = ", ".join(s.module for s in sdks if not s.importable)
        note = f"SDK not installed: {missing}"
    else:
        note = "ready"
    return VendorReadiness(
        vendor=vendor_value, kind=kind, registered=registered, sdks=sdks, ready=ready, note=note,
    )


def arm_vendor_readiness() -> list[VendorReadiness]:
    """Readiness for every known arm vendor (registered drivers + reserved empty slots)."""
    out: list[VendorReadiness] = []
    for vendor in RobotVendor:
        sdks = _ARM_VENDOR_SDKS.get(vendor, ())
        out.append(_readiness(vendor.value, "arm", sdks, _is_arm_registered(vendor)))
    return out


def gripper_vendor_readiness() -> list[VendorReadiness]:
    """Readiness for every known gripper vendor (SDK importability; reserved slots noted)."""
    out: list[VendorReadiness] = []
    for vendor in GripperVendor:
        if vendor in _GRIPPER_VENDOR_SDKS:
            out.append(_readiness(vendor.value, "gripper", _GRIPPER_VENDOR_SDKS[vendor], registered=True))
        else:
            # FRANKA_HAND / SCHUNK are reserved enum slots with no registered factory.
            out.append(
                VendorReadiness(
                    vendor=vendor.value, kind="gripper", registered=False, sdks=(),
                    ready=False, note="no driver registered (reserved slot / not implemented)",
                )
            )
    return out


def readiness_table() -> str:
    """Render a human-readable readiness table for every arm + gripper vendor."""
    rows = arm_vendor_readiness() + gripper_vendor_readiness()
    lines = [f"{'kind':<8}{'vendor':<14}{'ready':<7}note"]
    for r in rows:
        mark = "yes" if r.ready else "NO"
        lines.append(f"{r.kind:<8}{r.vendor:<14}{mark:<7}{r.note}")
    return "\n".join(lines)


def require_arm_vendor_ready(vendor: RobotVendor, *, mock_mode: bool = False) -> None:
    """Fail-early: raise if the configured arm vendor's SDK is missing.

    ``mock_mode=True`` skips the probe for vendors whose mock path drives no real device (the sim runs
    headless-mock without isaacsim; the dummy needs nothing).
    """
    if mock_mode and vendor in _MOCKABLE_ARM_VENDORS:
        logger.debug("mock mode: skipping the SDK gate for arm vendor %r", vendor.value)
        return
    probed = [probe_module(m) for m in _ARM_VENDOR_SDKS.get(vendor, ())]
    missing = [s.module for s in probed if not s.importable]
    if missing:
        raise RobotConnectionError(
            f"host not ready for arm vendor '{vendor.value}': missing SDK module(s) {missing}. "
            "Install the vendor extra (e.g. `pip install ur_rtde`, or the Isaac bundled python for the "
            "sim) and re-run, or `python -m src.robot.drivers.doctor` for the full table."
        )
    # The versions are the point: "it worked yesterday" is settled by comparing this line, not by
    # re-probing a host that has since been updated.
    logger.info(
        "arm vendor %r cleared the SDK gate (%s)",
        vendor.value,
        ", ".join(f"{s.module}=={s.version or '?'}" for s in probed) or "no third-party SDK required",
    )


def main(argv: list[str] | None = None) -> int:
    """CLI: print the readiness table; with --require VENDOR exit non-zero unless that arm vendor is ready.

    Exit codes: 0 = ok (table printed, or required vendor ready), 1 = required vendor not ready,
    2 = bad arguments (unknown vendor).
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="python -m src.robot.drivers.doctor",
        description="Probe which vendor SDKs are importable + report driver readiness.",
    )
    parser.add_argument("--require", metavar="VENDOR", default=None,
                        help="exit non-zero unless this arm vendor is ready (e.g. ur, sim, kuka).")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of the human table.")
    args = parser.parse_args(argv)

    rows = arm_vendor_readiness() + gripper_vendor_readiness()
    logger.info(
        "readiness probe: %d/%d vendors ready; not ready: %s",
        sum(1 for r in rows if r.ready), len(rows),
        ", ".join(f"{r.kind}/{r.vendor} ({r.note})" for r in rows if not r.ready) or "none",
    )
    if args.json:
        print(json.dumps(
            [
                {
                    "kind": r.kind, "vendor": r.vendor, "registered": r.registered, "ready": r.ready,
                    "note": r.note,
                    "sdks": [
                        {"module": s.module, "importable": s.importable, "version": s.version}
                        for s in r.sdks
                    ],
                }
                for r in rows
            ],
            indent=2,
        ))
    else:
        print(readiness_table())

    if args.require is not None:
        try:
            vendor = RobotVendor.from_string(args.require)
        except Exception:  # noqa: BLE001 - unknown vendor string -> bad-args exit
            print(f"unknown arm vendor: {args.require!r}")
            return 2
        ready = next((r.ready for r in rows if r.kind == "arm" and r.vendor == vendor.value), False)
        return 0 if ready else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
