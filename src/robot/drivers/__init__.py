"""
Vendor-specific arm driver packages.

Every driver implements :class:`src.robot.core.RobotArm`. Adding
a new vendor means dropping a sibling subpackage here that exposes one
class satisfying the Protocol pipelines / planners / the application
layer never need to change.

Registered today:

* :mod:`.ur`     Universal Robots, RTDE-based.
* :mod:`.kuka`   KUKA, EthernetKRL (EKI/KRL) over TCP/XML.
* :mod:`.dummy`  pure-Python sim arm for tests / offline development.

Package slots exist for ``franka``, ``ros2``, and ``sim``.

Vendor isolation
----------------
Per the architecture directive, **no module outside this package
imports vendor SDKs (``ur_rtde``, ``franky``, ``rclpy``, ...) directly.**
The :mod:`.registry` factories defer SDK imports until
:func:`create_arm` is actually invoked, so importing this package on a
laptop without the UR SDK does not blow up.

Public surface
--------------

* :class:`src.robot.core.RobotVendor` canonical identifiers.
* :func:`register_arm_driver` decorator-style driver registration.
* :func:`create_arm` factory: ``vendor -> RobotArm``.
* :func:`available_vendors` snapshot of registered vendors.
* :func:`is_vendor_registered`, :func:`unregister_arm_driver` small
  utilities mainly for tests / runtime introspection.

See ``drivers_README.md`` for a step-by-step guide to completing a new
vendor (KUKA, Franka, ROS 2, ...).
"""

from __future__ import annotations

from src.robot.core import RobotArm, RobotVendor

from .registry import (
    ArmDriverFactory,
    available_vendors,
    create_arm,
    is_vendor_registered,
    register_arm_driver,
    unregister_arm_driver,
)

__all__ = [
    "ArmDriverFactory",
    "RobotVendor",
    "available_vendors",
    "create_arm",
    "is_vendor_registered",
    "register_arm_driver",
    "unregister_arm_driver",
]


# ---------------------------------------------------------------------------
# Built-in driver registration
# ---------------------------------------------------------------------------


@register_arm_driver(RobotVendor.DUMMY)
def _make_dummy(**kwargs) -> RobotArm:
    """Build a :class:`.dummy.DummyRobotArm`.

    Accepted kwargs (all optional)::

        dof          : int   = 6
        initial_pose : Pose  = TCP at (400, 0, 300, identity)
    """
    from .dummy import DummyRobotArm

    # The app-layer factory always forwards ``config=...`` regardless of
    # vendor so operators can switch vendors purely by config edits.
    kwargs.pop("config", None)
    return DummyRobotArm(**kwargs)


@register_arm_driver(RobotVendor.UR)
def _make_ur(**kwargs) -> RobotArm:
    """Build a UR-backed :class:`src.robot.drivers.ur.URRobotArm`.

    Accepted kwargs::

        config : RobotConfig   the full robot config tree (required)

    This factory imports the UR driver chain only when UR is selected.
    The ``ur_rtde`` dependency itself is still deferred until connection
    time by :class:`URConnection`.
    """
    from .ur.arm import URRobotArm

    config = kwargs.pop("config", None)
    if config is None:
        raise TypeError("create_arm(RobotVendor.UR) requires a 'config=RobotConfig(...)' kwarg.")
    if kwargs:
        raise TypeError(
            f"create_arm(RobotVendor.UR) got unexpected kwargs: {sorted(kwargs)}"
        )
    return URRobotArm(config)


@register_arm_driver(RobotVendor.KUKA)
def _make_kuka(**kwargs) -> RobotArm:
    """Build a KUKA-backed :class:`src.robot.drivers.kuka.KukaRobotArm`.

    Accepted kwargs::

        config : RobotConfig   the full robot config tree (required)

    The KUKA driver speaks EthernetKRL (EKI) over TCP/XML to the
    controller-side KRL program shipped under
    ``config/data/robot/templates/kuka/``. No vendor SDK
    dependency is required on the Willy side.
    """
    from .kuka.arm import KukaRobotArm

    config = kwargs.pop("config", None)
    if config is None:
        raise TypeError(
            "create_arm(RobotVendor.KUKA) requires a 'config=RobotConfig(...)' kwarg."
        )
    if kwargs:
        raise TypeError(
            f"create_arm(RobotVendor.KUKA) got unexpected kwargs: {sorted(kwargs)}"
        )
    return KukaRobotArm(config)


@register_arm_driver(RobotVendor.SIM)
def _make_sim(**kwargs) -> RobotArm:
    """Build an Isaac-backed :class:`IsaacRobotArm` skeleton.

    Accepted kwargs::

        config : SimRobotConfig   sim cell config (required)

    Construction itself is import-safe on macOS: the Isaac SDK is only
    touched when :meth:`IsaacRobotArm.connect` runs. On a host without
    Isaac, ``connect()`` raises
    :class:`src.robot.drivers.sim.IsaacNotAvailableError`.
    """
    from .sim.arm import IsaacRobotArm
    from .sim.config import SimRobotConfig

    config = kwargs.pop("config", None)
    if config is None:
        raise TypeError(
            "create_arm(RobotVendor.SIM) requires a 'config=SimRobotConfig(...)' kwarg."
        )
    if not isinstance(config, SimRobotConfig):
        raise TypeError(
            "create_arm(RobotVendor.SIM) requires config to be a SimRobotConfig; "
            f"got {type(config).__name__}."
        )
    if kwargs:
        raise TypeError(
            f"create_arm(RobotVendor.SIM) got unexpected kwargs: {sorted(kwargs)}"
        )
    return IsaacRobotArm(config)
