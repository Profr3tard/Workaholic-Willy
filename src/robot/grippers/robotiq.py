"""
GripperController: thin wrapper around the Robotiq HE/HE-X gripper.

Talks to the gripper through the ``robotiq_gripper`` module that ships
with the Universal Robots / SDU ``ur_rtde`` ecosystem. The gripper's
native API speaks in raw 0--255 *position counts*; this class converts
them to and from millimetres using the configured opening range so the
rest of the codebase can stay in physical units.

Numerics contract
-----------------
* Public widths are millimetres.
* ``speed`` and ``force`` are normalised in ``[0.0, 1.0]`` (1.0 = max).
* The internal driver counts (0-255) never leak through the public API.

Robustness
----------
* The native ``robotiq_gripper`` module is deferred-imported so unit
  tests can mock it on machines that do not have the dependency.
* Every operation is logged to the shared robot logfile.
* A fully usable test seam is provided via the ``driver_factory``
  constructor argument: pass any callable returning an object with
  ``connect``, ``activate_if_needed`` (or ``activate``), ``move``,
  ``get_current_position``, and ``disconnect`` methods.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from config.schema.robot import GripperConfig

from ..constants import GRIPPER_LOG_FILE, create_robot_logger

# Native Robotiq position counts.
_POS_OPEN = 0          # fully open
_POS_CLOSED = 255      # fully closed
_WIDTH_CLOSED_MM = 0.0
_DEFAULT_PORT = 63352
_DEFAULT_SPEED = 1.0
_DEFAULT_FORCE = 0.5  # Robotiq dashboard port on the UR controller


def _default_driver_factory() -> Any:
    """Default factory: deferred import of the SDU ``robotiq_gripper`` module."""
    try:
        from robotiq_gripper import RobotiqGripper  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only on bare envs
        raise ImportError(
            "robotiq_gripper is not installed. Install ur_rtde with the "
            "Robotiq extras, or pass a custom driver_factory."
        ) from exc
    return RobotiqGripper()


class GripperController:
    """High-level Robotiq gripper control.

    Parameters
    ----------
    config : GripperConfig
        Physical opening limits (``min_width_mm`` .. ``max_width_mm``).
    ip : str
        IP address of the UR controller (the gripper is daisy-chained on
        the robot's tool I/O; the same IP is used).
    port : int
        Robotiq dashboard port. Defaults to 63352, the factory default.
    driver_factory : Callable returning a Robotiq-like driver, optional
        Test seam. Defaults to deferred-importing
        ``robotiq_gripper.RobotiqGripper``.
    """

    def __init__(
        self,
        config: GripperConfig,
        ip: str,
        port: int = _DEFAULT_PORT,
        driver_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.config = config
        self.ip = ip
        self.port = port
        self._driver_factory = driver_factory or _default_driver_factory
        self.logger = create_robot_logger("GripperController", GRIPPER_LOG_FILE)

        self._driver: Any | None = None
        self._activated: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._driver is not None

    @property
    def min_width_mm(self) -> float:
        """Smallest commandable jaw opening (forwarded from config). A POLICY floor, not the mechanism."""
        return float(self.config.min_width_mm)

    @property
    def closed_width_mm(self) -> float:
        """What :meth:`get_width_mm` reads with the jaws shut on nothing (2F-85: 0 mm).

        Distinct from :attr:`min_width_mm`, which is a policy floor. Verification asks for THIS one:
        "did the jaws collapse on nothing?" is a question about the mechanism.
        """
        return float(self.config.closed_width_mm)

    @property
    def max_width_mm(self) -> float:
        """Largest commandable jaw opening (forwarded from config)."""
        return float(self.config.max_width_mm)

    def connect(self) -> None:
        """Open the dashboard connection AND activate. Idempotent."""
        if self.is_connected:
            self.logger.debug("connect() called but already connected -- ignored.")
            return
        self.logger.info("Connecting to gripper at %s:%d ...", self.ip, self.port)
        driver = self._driver_factory()
        driver.connect(self.ip, self.port)
        self._driver = driver
        # Fail closed: a connected-but-unactivated gripper accepts commands and does not grip, which
        # reads as a bad grasp rather than an unconfigured tool. Roll the connection back instead.
        try:
            self.activate()
        except BaseException as exc:  # noqa: BLE001 - the rollback below is the point
            self.logger.error("gripper activation failed; rolling back the connection: %s", exc)
            self.disconnect()
            raise

    def disconnect(self) -> None:
        """Close the dashboard connection. Always safe to call."""
        if self._driver is None:
            return
        try:
            self._driver.disconnect()
        except (RuntimeError, OSError) as exc:
            self.logger.warning("Gripper disconnect raised %s -- ignoring.", exc)
        finally:
            self._driver = None
            self._activated = False

    def activate(self) -> None:
        """Activate the gripper if it is not already calibrated."""
        drv = self._require_connected()
        # Robotiq drivers expose either ``activate_if_needed`` (idempotent)
        # or just ``activate``; support both.
        fn = (
            getattr(drv, "activate_if_needed", None)
            or getattr(drv, "activate", None)
        )
        if fn is None:
            raise RuntimeError("Driver exposes neither activate() nor activate_if_needed().")
        self.logger.info("Activating gripper ...")
        fn()
        self._activated = True

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def open(self, *, speed: float | None = None, force: float | None = None) -> None:
        """Fully open the gripper. Query :meth:`get_width_mm` for the result."""
        self.set_width_mm(self.config.max_width_mm, speed=speed, force=force)

    def close(self, *, speed: float | None = None, force: float | None = None) -> None:
        """Fully close the gripper. Query :meth:`get_width_mm` for the result."""
        self.set_width_mm(self.config.min_width_mm, speed=speed, force=force)

    def set_width_mm(
        self, width_mm: float, *, speed: float | None = None, force: float | None = None,
    ) -> None:
        """Command an opening of ``width_mm``; clamped to the configured range.

        Parameters
        ----------
        width_mm : float
            Target opening in millimetres.
        speed, force : float, optional
            Normalised in ``[0, 1]``. Mapped to the driver's 0--255 scale internally.
            ``None`` (the :class:`Gripper` Protocol's default, and what
            :class:`GraspExecutionPolicy` sends unless a caller sets ``close_speed`` /
            ``close_force_n`` nothing in the repo does) means "the driver's default":
            :data:`_DEFAULT_SPEED` / :data:`_DEFAULT_FORCE`.

        Notes
        -----
        Returns ``None`` per the :class:`Gripper` Protocol — query
        :meth:`get_width_mm` for the achieved opening.
        """
        drv = self._require_connected()
        if not self._activated:
            self.logger.warning("set_width_mm called before activate(); activating now.")
            self.activate()

        clamped_mm = self._clamp_width_mm(width_mm)
        target_count = self._mm_to_count(clamped_mm)
        # Resolve the Protocol's optional-means-driver-default HERE, at the boundary. The vendor
        # arithmetic below takes a real number and nothing else.
        speed = _DEFAULT_SPEED if speed is None else speed
        force = _DEFAULT_FORCE if force is None else force
        speed_count = self._normalise_to_count(speed)
        force_count = self._normalise_to_count(force)

        self.logger.debug(
            "Gripper move: width=%.2f mm (cnt=%d), speed=%.2f, force=%.2f",
            clamped_mm, target_count, speed, force,
        )
        drv.move(target_count, speed_count, force_count)

    def get_width_mm(self) -> float:
        """Read the current opening width in millimetres."""
        drv = self._require_connected()
        count = int(drv.get_current_position())
        return float(self._count_to_mm(count))

    # ------------------------------------------------------------------
    # Context manager sugar
    # ------------------------------------------------------------------

    def __enter__(self) -> GripperController:
        self.connect()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_connected(self) -> Any:
        if self._driver is None:
            raise RuntimeError("Gripper is not connected. Call connect() first.")
        return self._driver

    def _clamp_width_mm(self, width_mm: float) -> float:
        lo = self.config.min_width_mm
        hi = self.config.max_width_mm
        if width_mm < lo:
            self.logger.debug("Clamping width %.2f -> %.2f (min).", width_mm, lo)
            return lo
        if width_mm > hi:
            self.logger.debug("Clamping width %.2f -> %.2f (max).", width_mm, hi)
            return hi
        return float(width_mm)

    def _mm_to_count(self, width_mm: float) -> int:
        """Map a finger gap in mm to Robotiq driver counts (_POS_CLOSED=255 .. _POS_OPEN=0)."""
        hi = self.config.max_width_mm
        span = hi - _WIDTH_CLOSED_MM
        frac = (width_mm - _WIDTH_CLOSED_MM) / span if span > 0 else 0.0  # 0..1
        frac = min(1.0, max(0.0, frac))
        count = _POS_CLOSED + (_POS_OPEN - _POS_CLOSED) * frac
        return int(round(count))

    def _count_to_mm(self, count: int) -> float:
        """Inverse of :meth:`_mm_to_count`: driver counts -> physical finger gap in mm."""
        hi = self.config.max_width_mm
        count = min(_POS_CLOSED, max(_POS_OPEN, count))
        frac = (count - _POS_CLOSED) / (_POS_OPEN - _POS_CLOSED)
        return _WIDTH_CLOSED_MM + (hi - _WIDTH_CLOSED_MM) * frac

    @staticmethod
    def _normalise_to_count(value: float) -> int:
        v = min(1.0, max(0.0, float(value)))
        return int(round(v * 255))
