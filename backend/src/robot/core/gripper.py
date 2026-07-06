"""
:class:`Gripper` — vendor-neutral abstract surface for a parallel-jaw gripper.

Drivers under :mod:`backend.src.robot.drivers` (Robotiq HE, Robotiq HE-X,
Franka Hand, sim, dummy) implement this Protocol. The arm package
exposes a configured :class:`Gripper` alongside the :class:`RobotArm`,
but the two are independent — a driver that has no gripper attached
returns a no-op implementation rather than ``None``.

Numerics contract
-----------------
* All widths are in **millimetres** (``float``).
* ``force`` is normalised to ``[0.0, 1.0]`` — a driver-dependent mapping to the underlying
  force scale / counts (e.g. the Robotiq driver maps it to 0-255; it is NOT newtons).
* Speeds are normalised to ``[0.0, 1.0]`` (driver-dependent mapping to
  the underlying counts / mm-per-second). Pipelines that need physical
  units must inspect :class:`RobotCapabilities`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["Gripper", "ObjectDetectingGripper"]


@runtime_checkable
class Gripper(Protocol):
    """Vendor-neutral gripper interface."""

    # ---- introspection --------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """``True`` while a live link to the gripper is open."""
        ...

    @property
    def min_width_mm(self) -> float:
        """Smallest commandable jaw opening (mm)."""
        ...

    @property
    def max_width_mm(self) -> float:
        """Largest commandable jaw opening (mm)."""
        ...

    # ---- lifecycle ------------------------------------------------------

    def connect(self) -> None:
        """Open the link to the gripper. Idempotent."""
        ...

    def disconnect(self) -> None:
        """Close the link. Idempotent."""
        ...

    def activate(self) -> None:
        """Run the vendor-specific calibration / activation routine."""
        ...

    # ---- commands -------------------------------------------------------

    def set_width_mm(
        self,
        width_mm: float,
        *,
        speed: float | None = None,
        force: float | None = None,
    ) -> None:
        """
        Command jaw opening to ``width_mm``.

        Parameters
        ----------
        width_mm
            Target opening, clamped into ``[min_width_mm, max_width_mm]``
            by the driver.
        speed
            Optional normalised speed in ``[0.0, 1.0]``.
        force
            Optional normalised grasp force in ``[0.0, 1.0]`` (NOT newtons —
            see the numerics contract above). ``None`` keeps the driver
            default.
        """
        ...

    def get_width_mm(self) -> float:
        """Current jaw opening (mm)."""
        ...


@runtime_checkable
class ObjectDetectingGripper(Protocol):
    """Capability extension: gripper can report whether it is holding an object.

    Drivers that advertise this Protocol opt-in to **post-close
    verification** in :class:`GraspExecutionPolicy`. Drivers without a
    feedback channel simply do not implement this method, and the
    policy then trusts the close command (the documented default).

    The check is run after the gripper has commanded its close target
    width; it must return ``True`` when the jaws have engaged the object
    and ``False`` otherwise (slipped / missed / empty grasp).
    """

    def is_object_detected(self) -> bool:
        """Return ``True`` if the gripper is currently holding an object."""
        ...
