"""
Vendor-specific gripper driver modules.

Every driver implements :class:`src.robot.core.Gripper`. Adding
a new gripper means dropping a sibling module here that exposes one
class satisfying the Protocol pipelines never need to change.

Currently shipping:

* :class:`.robotiq.GripperController` Robotiq HE / HE-X via the
  ``robotiq_gripper`` SDK shipped with the SDU ``ur_rtde`` stack.
* :class:`.dummy.DummyGripper` pure-Python sim gripper.
* :class:`.null.NullGripper` explicit no-op for "no gripper
  attached" robots.

Public surface
--------------

* :class:`src.robot.core.GripperVendor` canonical identifiers.
* :func:`register_gripper_driver` decorator-style registration.
* :func:`create_gripper` factory: ``vendor -> Gripper``.
* :func:`available_gripper_vendors`,
  :func:`is_gripper_vendor_registered`,
  :func:`unregister_gripper_driver` introspection / test utilities.

Backwards compatibility
-----------------------
The historic :class:`GripperController` symbol is still re-exported
from this package so existing imports keep working.
"""

from __future__ import annotations

from typing import cast

from src.robot.core import Gripper, GripperVendor

from .registry import (
    GripperDriverFactory,
    available_gripper_vendors,
    create_gripper,
    is_gripper_vendor_registered,
    register_gripper_driver,
    unregister_gripper_driver,
)
from .null import GripperSubstitution, SubstitutionReason
from .robotiq import GripperController  # back-compat re-export

__all__ = [
    "Gripper",
    "GripperController",
    "GripperDriverFactory",
    "GripperSubstitution",
    "GripperVendor",
    "SubstitutionReason",
    "available_gripper_vendors",
    "create_gripper",
    "is_gripper_vendor_registered",
    "register_gripper_driver",
    "unregister_gripper_driver",
]


# ---------------------------------------------------------------------------
# Built-in driver registration
# ---------------------------------------------------------------------------


@register_gripper_driver(GripperVendor.ROBOTIQ)
def _make_robotiq(**kwargs) -> Gripper:
    """Build a :class:`.robotiq.GripperController`.

    Required kwargs::

        config : GripperConfig     physical opening limits
        ip     : str               IP of the UR controller (gripper is
                                   daisy-chained on the tool I/O)

    Optional kwargs::

        port           : int   = 63352
        driver_factory : Callable returning a Robotiq-like driver

    The ``robotiq_gripper`` SDK is imported lazily by the controller
    itself --- this factory only constructs the wrapper.
    """
    from .robotiq import GripperController
    return cast(Gripper, GripperController(**kwargs))


@register_gripper_driver(GripperVendor.DUMMY)
def _make_dummy(**kwargs) -> Gripper:
    """Build a :class:`.dummy.DummyGripper` (pure-Python sim).

    Accepted kwargs (all optional)::

        min_width_mm     : float = 0.0
        max_width_mm     : float = 150.0
        initial_width_mm : float = max_width_mm
    """
    from .dummy import DummyGripper
    return DummyGripper(**kwargs)


@register_gripper_driver(GripperVendor.VACUUM)
def _make_vacuum(**kwargs) -> Gripper:
    """Build a :class:`.vacuum.VacuumGripper` (suction over the controller's digital I/O).

    Required kwargs::

        io : SupportsDigitalIO     the I/O source; on a real cell this is
                                   the ARM driver, because the ejector is
                                   wired to the robot controller.

    Optional kwargs mirror :class:`VacuumGripperConfig` (pins, port, timings,
    the vacuum-on width threshold) plus ``config`` for the width limits.
    """
    from .vacuum import VacuumGripper
    return cast(Gripper, VacuumGripper(**kwargs))


@register_gripper_driver(GripperVendor.JAW_IO)
def _make_jaw_io(**kwargs) -> Gripper:
    """Build a :class:`.jaw_io.JawIOGripper` (parallel jaws over the controller's digital I/O).

    Required kwargs::

        io : SupportsDigitalIO     the I/O source; on a real cell this is
                                   the ARM driver, because the valve is
                                   wired to the robot controller.

    Optional kwargs mirror :class:`JawIOGripperConfig` (actuation, pins,
    port, timings, the closed-below width threshold) plus ``config`` for
    the width limits.
    """
    from .jaw_io import JawIOGripper
    return cast(Gripper, JawIOGripper(**kwargs))


@register_gripper_driver(GripperVendor.NONE)
def _make_null(**kwargs) -> Gripper:
    """Build a :class:`.null.NullGripper` (no gripper attached).

    Accepted kwargs (all optional)::

        min_width_mm : float = 0.0
        max_width_mm : float = 0.0
        substitution : GripperSubstitution | None = None

    ``substitution`` is how a build path says "a real gripper was asked for and could not be made"
    rather than "this cell has no end-effector", which are the same object and very different facts.
    """
    from .null import NullGripper
    return NullGripper(**kwargs)
