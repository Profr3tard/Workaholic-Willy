"""
Driver registry the single entry point for selecting a vendor-specific
:class:`~src.robot.core.RobotArm` implementation.

Design
------
The registry stores **lazy factory callables**, not driver classes. This
matters because:

* It keeps :mod:`src.robot.core` and the registry itself free
  of vendor-SDK imports.
  Vendor SDK imports (``rtde_control``, ``franky``, ``rclpy``, ...)
  happen inside the factory body, the first time a caller actually
  asks for that vendor.
* It lets a driver fail to load gracefully on a host where its SDK is
  not installed: the call to :func:`create_arm` raises a clear
  :class:`RobotConnectionError`, all other vendors stay usable.

Public surface
--------------

* :func:`register_arm_driver` decorator-style registration. Used
  internally by :mod:`.ur` / :mod:`.dummy`; downstream KUKA / Franka
  drivers do exactly the same in their own ``__init__``.
* :func:`create_arm` factory: ``vendor -> RobotArm``.
* :func:`available_vendors` list of registered vendor strings.
* :func:`is_vendor_registered` single-vendor probe.
* :func:`unregister_arm_driver` mostly for tests.

A tiny worked example::

    from src.robot.core import RobotVendor
    from src.robot.drivers import create_arm, register_arm_driver

    @register_arm_driver(RobotVendor.SIM)
    def _make_sim(**kwargs):
        from .sim.arm import SimArm
        return SimArm(**kwargs)

    arm = create_arm(RobotVendor.SIM, dof=7)
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from src.robot.constants import DRIVER_REGISTRY_LOG_FILE, create_robot_logger
from src.robot.core import RobotArm, RobotConnectionError, RobotVendor

__all__ = [
    "ArmDriverFactory",
    "available_vendors",
    "create_arm",
    "is_vendor_registered",
    "register_arm_driver",
    "unregister_arm_driver",
]


# A driver factory is any callable that, given vendor-specific kwargs,
# returns a fully-constructed :class:`RobotArm`.
ArmDriverFactory = Callable[..., RobotArm]


_REGISTRY: dict[RobotVendor, ArmDriverFactory] = {}
_LOCK = threading.RLock()


logger = create_robot_logger("ArmDriverRegistry", DRIVER_REGISTRY_LOG_FILE)


def _coerce(vendor: RobotVendor | str) -> RobotVendor:
    if isinstance(vendor, RobotVendor):
        return vendor
    return RobotVendor.from_string(vendor)


def register_arm_driver(
    vendor: RobotVendor | str,
    *,
    overwrite: bool = False,
) -> Callable[[ArmDriverFactory], ArmDriverFactory]:
    """Register a factory under ``vendor``.

    Use as a decorator::

        @register_arm_driver(RobotVendor.UR)
        def _make_ur(**kwargs) -> RobotArm:
            from src.robot.drivers.ur.arm import URRobotArm
            return URRobotArm(kwargs["config"])

    By default re-registering an existing vendor raises
    :class:`ValueError`. Pass ``overwrite=True`` to replace (used by
    tests).
    """
    key = _coerce(vendor)

    def _decorator(factory: ArmDriverFactory) -> ArmDriverFactory:
        if not callable(factory):
            raise TypeError(
                f"register_arm_driver: factory must be callable, got {type(factory).__name__}"
            )
        with _LOCK:
            if key in _REGISTRY and not overwrite:
                raise ValueError(
                    f"register_arm_driver: vendor {key.value!r} is already registered; "
                    "pass overwrite=True to replace."
                )
            _REGISTRY[key] = factory
        logger.debug(
            "registered arm driver: vendor=%s factory=%s%s",
            key.value, getattr(factory, "__qualname__", repr(factory)),
            " (overwrite)" if overwrite else "",
        )
        return factory

    return _decorator


def unregister_arm_driver(vendor: RobotVendor | str) -> None:
    """Remove a previously-registered factory.

    No-op if the vendor was not registered. Mainly useful in tests.
    """
    key = _coerce(vendor)
    with _LOCK:
        _REGISTRY.pop(key, None)


def is_vendor_registered(vendor: RobotVendor | str) -> bool:
    """``True`` iff a factory is registered under ``vendor``."""
    key = _coerce(vendor)
    with _LOCK:
        return key in _REGISTRY


def available_vendors() -> list[str]:
    """Snapshot of currently-registered vendor strings, sorted."""
    with _LOCK:
        return sorted(v.value for v in _REGISTRY)


def create_arm(vendor: RobotVendor | str, **kwargs) -> RobotArm:
    """Instantiate the driver registered under ``vendor``.

    All keyword arguments are forwarded to the factory verbatim 
    each driver documents its own kwargs (see the per-driver
    ``__init__.py`` modules under :mod:`src.robot.drivers`).

    Raises
    ------
    ValueError
        If ``vendor`` is not a known :class:`RobotVendor`.
    RobotConnectionError
        If the vendor is recognised but no factory is registered (the
        driver package is missing or its SDK failed to import).
    """
    key = _coerce(vendor)
    with _LOCK:
        factory = _REGISTRY.get(key)
    if factory is None:
        registered = ", ".join(available_vendors()) or "(none)"
        raise RobotConnectionError(
            f"no driver registered for vendor {key.value!r}; "
            f"available: {registered}. Make sure the driver package "
            f"and its SDK are installed."
        )
    arm = factory(**kwargs)
    logger.info(
        "created arm driver: vendor=%s -> %s (kwargs=%s)",
        key.value, type(arm).__name__, sorted(kwargs),
    )
    # The Protocol is runtime_checkable; use it as a structural guard
    # so a buggy factory cannot quietly return the wrong shape.
    if not isinstance(arm, RobotArm):
        raise TypeError(
            f"factory for vendor {key.value!r} returned "
            f"{type(arm).__name__}, which does not satisfy the RobotArm Protocol."
        )
    return arm
