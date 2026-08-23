"""Isaac Sim-backed :class:`Gripper` driver (parallel-jaw).

``IsaacGripper`` implements Willy's vendor-neutral :class:`~src.robot.core.Gripper`
Protocol by driving a gripper *articulation* in Isaac. Everything brand-specific which
joint to command, the (nonlinear) width↔angle mapping, the open/closed extremes lives in
a swappable :class:`GripperProfile`. So a second gripper brand is a **new profile**, not a
new driver; ``IsaacGripper`` drives any profile.

Design notes (validated on-box, 2026-06-05, Robotiq 2F-85):
* The 2F-85 is a closed-loop mimic mechanism commanding only ``finger_joint`` makes the
  other 5 joints follow 1:1, so the profile names a single ``driven_joint``.
* In this Isaac USD the convention is ``finger_joint = 0 → closed (0 mm)``,
  ``0.8 rad → open (~85 mm)`` (measured, not the datasheet's angle convention).
* The width↔angle curve is mildly nonlinear (4-bar linkage); the profile carries the
  measured lookup table and interpolates both ways.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ...core import IsaacNotAvailableError, RobotConnectionError

if TYPE_CHECKING:  # pragma: no cover - typing only; keeps the runtime edge out of drivers.sim
    from src.robot.drivers.sim._isaac_protocols import IsaacArticulation
    from src.robot.drivers.sim.session import IsaacSimSession

__all__ = [
    "GripperProfile",
    "ROBOTIQ_2F85_PROFILE",
    "SCHUNK_EGU50_PROFILE",
    "SCHUNK_EZU35_PROFILE",
    "IsaacGripper",
]

# Settle: max gripper-joint speed (rad/s) below which the jaws are treated as stopped.
_GRIPPER_SETTLE_VEL = 1e-2


@dataclass(frozen=True)
class GripperProfile:
    """Brand-specific sim-gripper kinematics behind the vendor-neutral Gripper Protocol.

    A new gripper brand/model = a new ``GripperProfile`` (its ``driven_joint`` + an in-sim
    calibrated ``angle_width_table``), reused by the same :class:`IsaacGripper`.

    ``angle_width_table`` is ``((angle_rad, width_mm), …)`` sorted by ascending angle and
    monotonic in width (as measured in-sim). Both lookups use linear interpolation.
    """

    name: str
    driven_joint: str
    angle_width_table: tuple[tuple[float, float], ...]

    @property
    def _angles(self) -> np.ndarray:
        return np.asarray([a for a, _ in self.angle_width_table], dtype=np.float64)

    @property
    def _widths(self) -> np.ndarray:
        return np.asarray([w for _, w in self.angle_width_table], dtype=np.float64)

    @property
    def min_width_mm(self) -> float:
        return float(self._widths.min())

    @property
    def max_width_mm(self) -> float:
        return float(self._widths.max())

    @property
    def min_angle(self) -> float:
        return float(self._angles.min())

    @property
    def max_angle(self) -> float:
        return float(self._angles.max())

    def width_to_angle(self, width_mm: float) -> float:
        """Map a jaw width (mm) to the ``driven_joint`` angle (rad), clamped to range."""
        w = float(np.clip(width_mm, self.min_width_mm, self.max_width_mm))
        order = np.argsort(self._widths)
        return float(np.interp(w, self._widths[order], self._angles[order]))

    def angle_to_width(self, angle_rad: float) -> float:
        """Map a ``driven_joint`` angle (rad) to the jaw width (mm)."""
        order = np.argsort(self._angles)
        return float(np.interp(float(angle_rad), self._angles[order], self._widths[order]))


# Robotiq 2F-85 calibrated in Isaac Sim 5.1 by measuring the REAL jaw gap (the world AABB gap
# between the inner-finger inner edges) vs finger_joint, with the gripper pointing down.
ROBOTIQ_2F85_PROFILE = GripperProfile(
    name="robotiq_2f85",
    driven_joint="finger_joint",
    angle_width_table=(
        (0.0, 87.1),
        (0.1, 77.8),
        (0.2, 68.1),
        (0.3, 57.9),
        (0.4, 47.2),
        (0.5, 36.2),
        (0.6, 25.0),
        (0.7, 13.6),
        (0.8, 2.2),
    ),
)


# Schunk EGU-50 a DIFFERENT-vendor parallel gripper, mounted on the UR5e via
# the mount harness (ur5e Gripper variant="None" + reference the standalone egu_50.usd + drop its
# articulation root + a fixed joint wrist_3_link->base, so it merges into the UR5e articulation).
SCHUNK_EGU50_PROFILE = GripperProfile(
    name="schunk_egu50",
    driven_joint="Jaw_Drive",
    angle_width_table=(
        (0.000, 4.1),
        (0.005, 14.1),
        (0.010, 24.1),
        (0.015, 34.1),
        (0.020, 44.1),
        (0.025, 54.1),
        (0.030, 64.1),
        (0.035, 74.1),
        (0.040, 84.1),
    ),
)


# Schunk EZU-35 a 3-FINGER CENTRIC gripper ("more than two fingers"). Mounted via the
# SAME harness as the EGU-50.
SCHUNK_EZU35_PROFILE = GripperProfile(
    name="schunk_ezu35",
    driven_joint="Jaw_Drive",
    angle_width_table=(  # measured on the mounted gripper (h42_ezu_grip): the 3-finger INSCRIBED grip diameter
        (0.000, 11.0),    # Jaw_Drive lower limit -> fingers nearly touching (~11 mm inscribed)
        (0.010, 31.0),
        (0.020, 51.0),
        (0.030, 71.0),
        (0.035, 81.0),    # Jaw_Drive upper limit -> open ~81 mm
    ),
)


class IsaacGripper:
    """Isaac-backed parallel-jaw :class:`Gripper` (drives one articulation via a profile).

    Parameters
    ----------
    session
        The (shared) :class:`IsaacSimSession` the gripper steps it to let the jaws settle.
        The arm owns the session and the gripper shares it.
    gripper_prim_path
        Prim path of the gripper articulation (e.g. the 2F-85).
    profile
        The brand-specific :class:`GripperProfile` (defaults to the Robotiq 2F-85).
    mock_mode
        Pure-Python width cache; no Isaac is touched (macOS / CI).
    settle_timeout_s
        Max sim seconds to wait for the jaws to stop after a command.
    """

    def __init__(
        self,
        *,
        session: IsaacSimSession,
        gripper_prim_path: str | None,  # schema field is str | None; connect() enforces non-empty
        profile: GripperProfile = ROBOTIQ_2F85_PROFILE,
        mock_mode: bool = False,
        settle_timeout_s: float = 2.0,
    ) -> None:
        self._session = session
        self._prim_path = gripper_prim_path
        self._profile = profile
        self._mock_mode = mock_mode
        self._settle_timeout_s = settle_timeout_s
        self._connected = False
        self._articulation: IsaacArticulation | None = None  # typed Isaac handle
        self._joint_index: int | None = None
        # Mock cache (and the pre-connect default): start fully open.
        self._mock_width_mm = profile.max_width_mm

    # ---- introspection --------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def min_width_mm(self) -> float:
        return self._profile.min_width_mm

    @property
    def max_width_mm(self) -> float:
        return self._profile.max_width_mm

    @property
    def profile(self) -> GripperProfile:
        return self._profile

    # ---- lifecycle ------------------------------------------------------

    def connect(self) -> None:
        """Wrap the gripper articulation (non-mock). Idempotent. Session must be started."""
        if self._connected:
            return
        if self._mock_mode:
            self._connected = True
            return
        if not self._prim_path:
            raise RobotConnectionError("IsaacGripper requires a gripper_prim_path (non-mock).")
        from isaacsim.core.prims import SingleArticulation  # type: ignore[import-not-found]

        articulation = SingleArticulation(prim_path=self._prim_path)
        articulation.initialize()
        names = list(articulation.dof_names)
        if self._profile.driven_joint not in names:
            raise IsaacNotAvailableError(
                f"driven joint {self._profile.driven_joint!r} not in gripper dof_names {names}."
            )
        self._articulation = articulation
        self._joint_index = names.index(self._profile.driven_joint)
        self._connected = True

    def disconnect(self) -> None:
        self._articulation = None
        self._joint_index = None
        self._connected = False

    def activate(self) -> None:
        """No-op in sim (the real Robotiq runs a calibration sweep here)."""
        return None

    # ---- commands -------------------------------------------------------

    def set_width_mm(
        self,
        width_mm: float,
        *,
        speed: float | None = None,
        force: float | None = None,
    ) -> None:
        """Command the jaw opening to ``width_mm`` (clamped to the profile range)."""
        if not self._connected:
            raise RobotConnectionError("IsaacGripper is not connected.")
        target = float(np.clip(width_mm, self.min_width_mm, self.max_width_mm))
        if self._mock_mode or self._articulation is None or self._joint_index is None:
            self._mock_width_mm = target
            return
        from isaacsim.core.utils.types import ArticulationAction  # type: ignore[import-not-found]

        angle = self._profile.width_to_angle(target)
        self._articulation.apply_action(
            ArticulationAction(
                joint_positions=np.array([angle], dtype=np.float64),
                joint_indices=np.array([self._joint_index]),
            )
        )
        self._settle()

    def get_width_mm(self) -> float:
        """Current jaw opening (mm)."""
        if not self._connected:
            raise RobotConnectionError("IsaacGripper is not connected.")
        if self._mock_mode or self._articulation is None or self._joint_index is None:
            return self._mock_width_mm
        q = np.asarray(self._articulation.get_joint_positions(), dtype=np.float64)
        return self._profile.angle_to_width(float(q[self._joint_index]))

    def open(self) -> None:
        """Open the jaws fully (profile ``max_width_mm``)."""
        self.set_width_mm(self.max_width_mm)

    def close(self) -> None:
        """Close the jaws fully (profile ``min_width_mm``)."""
        self.set_width_mm(self.min_width_mm)

    # ---- internals ------------------------------------------------------

    def _settle(self) -> bool:
        if self._articulation is None:
            return True
        dt = self._session.config.step_dt_s if self._session.config.step_dt_s > 0 else 1.0 / 60.0
        for _ in range(max(1, int(self._settle_timeout_s / dt))):
            self._session.step()
            vel = np.asarray(self._articulation.get_joint_velocities(), dtype=np.float64)
            if float(np.max(np.abs(vel))) < _GRIPPER_SETTLE_VEL:
                return True
        return False
