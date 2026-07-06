"""
Declarative capability descriptor for a :class:`RobotArm` driver.

Each driver advertises its feature set via a frozen
:class:`RobotCapabilities` dataclass. Pipeline / planner code consults
the capabilities to decide what is safe to call (e.g. skip
``move_linear`` if the driver does not support Cartesian moves; or fall
back to an offline FK if ``has_native_fk == False``).

Pipeline / planner code branches on these flags instead of doing
``isinstance(driver, ...)`` checks against concrete driver classes.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["RobotCapabilities"]


@dataclass(frozen=True, slots=True)
class RobotCapabilities:
    """
    Feature flags advertised by a robot-arm driver.

    Parameters
    ----------
    vendor : str
        Free-form vendor / driver identifier (e.g. ``"ur"``, ``"franka"``,
        ``"ros2"``, ``"sim"``, ``"dummy"``). Lower-case, no whitespace.
    model : str
        Arm model identifier (e.g. ``"ur5e"``, ``"panda"``,
        ``"iiwa14"``); free-form, may be empty.
    dof : int
        Degrees of freedom (joint count). Validated against
        :class:`JointPositions` lengths at the protocol boundary.
    supports_joint_move : bool
        Driver implements :meth:`RobotArm.move_joint`.
    supports_linear_move : bool
        Driver implements :meth:`RobotArm.move_linear` (Cartesian).
    supports_async_move : bool
        Driver can dispatch non-blocking moves (the existing UR
        ``moveJ(asynchronous=True)`` path).
    has_native_fk : bool
        Driver provides controller-side forward kinematics.
    has_native_ik : bool
        Driver provides controller-side inverse kinematics.
    has_force_control : bool
        Driver exposes a Cartesian force-control / admittance-control
        primitive. Reserved for a future force-control surface; defaults to ``False``.
    is_simulated : bool
        ``True`` if the driver is purely software (no real hardware).
    """

    vendor: str
    model: str = ""
    dof: int = 6
    supports_joint_move: bool = True
    supports_linear_move: bool = True
    supports_async_move: bool = False
    has_native_fk: bool = False
    has_native_ik: bool = False
    has_force_control: bool = False
    is_simulated: bool = False

    def __post_init__(self) -> None:
        if not self.vendor:
            raise ValueError("RobotCapabilities.vendor must be non-empty.")
        if " " in self.vendor or self.vendor != self.vendor.lower():
            raise ValueError(
                f"RobotCapabilities.vendor must be lowercase and whitespace-free; "
                f"got {self.vendor!r}."
            )
        if self.dof <= 0:
            raise ValueError(f"RobotCapabilities.dof must be > 0; got {self.dof}.")
