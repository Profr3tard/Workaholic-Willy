"""
Gripper driver registry entry point for selecting a vendor-specific
:class:`~src.robot.core.Gripper` implementation.

Mirrors :mod:`src.robot.drivers.registry` exactly:

* Stores **lazy factory callables**, not classes, so vendor SDK
  imports stay inside the factory body.
* Importing this module is safe on hosts without ``robotiq_gripper``
  / ``libfranka`` / etc. installed the SDK is only touched when
  :func:`create_gripper` is actually called for that vendor.
* :func:`create_gripper` validates the returned object against the
  :class:`Gripper` Protocol so a buggy factory cannot quietly return
  the wrong shape.

Public surface
--------------

* :func:`register_gripper_driver` decorator-style registration.
* :func:`create_gripper` factory: ``vendor -> Gripper``.
* :func:`available_gripper_vendors` snapshot of registered vendors.
* :func:`is_gripper_vendor_registered`,
  :func:`unregister_gripper_driver` small utilities for tests.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from src.robot.constants import GRIPPER_REGISTRY_LOG_FILE, create_robot_logger
from src.robot.core import (
    Gripper,
    GripperVendor,
    RobotConnectionError,
)

__all__ = [
    "GripperDriverFactory",
    "available_gripper_vendors",
    "create_gripper",
    "is_gripper_vendor_registered",
    "register_gripper_driver",
    "unregister_gripper_driver",
]


GripperDriverFactory = Callable[..., Gripper]


_REGISTRY: dict[GripperVendor, GripperDriverFactory] = {}
_LOCK = threading.RLock()

# Logging for registry changes and factory calls.
logger = create_robot_logger("GripperDriverRegistry", GRIPPER_REGISTRY_LOG_FILE)


def _coerce(vendor: GripperVendor | str) -> GripperVendor:
    if isinstance(vendor, GripperVendor):
        return vendor
    return GripperVendor.from_string(vendor)


def register_gripper_driver(
    vendor: GripperVendor | str,
    *,
    overwrite: bool = False,
) -> Callable[[GripperDriverFactory], GripperDriverFactory]:
    """Register a factory under ``vendor``.

    Use as a decorator::

        @register_gripper_driver(GripperVendor.ROBOTIQ)
        def _make_robotiq(**kwargs) -> Gripper:
            from .robotiq import GripperController
            return GripperController(**kwargs)

    Re-registering an existing vendor raises :class:`ValueError` unless
    ``overwrite=True``.
    """
    key = _coerce(vendor)

    def _decorator(factory: GripperDriverFactory) -> GripperDriverFactory:
        if not callable(factory):
            raise TypeError(
                "register_gripper_driver: factory must be callable, "
                f"got {type(factory).__name__}"
            )
        with _LOCK:
            if key in _REGISTRY and not overwrite:
                raise ValueError(
                    f"register_gripper_driver: vendor {key.value!r} is "
                    "already registered; pass overwrite=True to replace."
                )
            _REGISTRY[key] = factory
        logger.debug(
            "registered gripper driver: vendor=%s factory=%s%s",
            key.value, getattr(factory, "__qualname__", repr(factory)),
            " (overwrite)" if overwrite else "",
        )
        return factory

    return _decorator


def unregister_gripper_driver(vendor: GripperVendor | str) -> None:
    """Remove a previously-registered factory. No-op if absent."""
    key = _coerce(vendor)
    with _LOCK:
        _REGISTRY.pop(key, None)


def is_gripper_vendor_registered(vendor: GripperVendor | str) -> bool:
    """``True`` iff a factory is registered under ``vendor``."""
    key = _coerce(vendor)
    with _LOCK:
        return key in _REGISTRY


def available_gripper_vendors() -> list[str]:
    """Snapshot of currently-registered gripper vendor strings, sorted."""
    with _LOCK:
        return sorted(v.value for v in _REGISTRY)


def create_gripper(vendor: GripperVendor | str, **kwargs) -> Gripper:
    """Instantiate the gripper driver registered under ``vendor``.

    Raises
    ------
    ValueError
        If ``vendor`` is not a known :class:`GripperVendor`.
    RobotConnectionError
        If the vendor is recognised but no factory is registered.
    TypeError
        If the factory returned an object that does not satisfy the
        :class:`Gripper` Protocol.
    """
    key = _coerce(vendor)
    with _LOCK:
        factory = _REGISTRY.get(key)
    if factory is None:
        registered = ", ".join(available_gripper_vendors()) or "(none)"
        raise RobotConnectionError(
            f"no gripper driver registered for vendor {key.value!r}; "
            f"available: {registered}. Make sure the driver module and "
            "its SDK are installed."
        )
    gripper = factory(**kwargs)
    logger.info(
        "created gripper driver: vendor=%s -> %s (kwargs=%s)",
        key.value, type(gripper).__name__, sorted(kwargs),
    )
    if not isinstance(gripper, Gripper):
        raise TypeError(
            f"factory for gripper vendor {key.value!r} returned "
            f"{type(gripper).__name__}, which does not satisfy the "
            "Gripper Protocol."
        )
    return gripper
