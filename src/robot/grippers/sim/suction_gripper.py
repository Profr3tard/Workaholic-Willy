"""Isaac Sim suction :class:`Gripper` driver (surface-gripper / vacuum cup).

``IsaacSuctionGripper`` is the second end-effector modality behind Willy's vendor-neutral
:class:`~src.robot.core.Gripper` Protocol: instead of a parallel jaw it drives an Isaac
**surface gripper** (a binary proximity-joint vacuum surrogate) authored on the UR5e wrist by
:func:`src.willy_sim.suction_mount.mount_suction_cup`. The Protocol's width semantics are
reinterpreted as vacuum on/off:

* ``set_width_mm(w)`` with ``w <= vacuum_on_below_mm`` -> **vacuum ON** (engage; step + poll until the
  surface gripper reads ``Closed`` with a body bonded);
* ``set_width_mm(w)`` above that -> **vacuum OFF** (release).

So the SAME :class:`~src.robot.grasping.motion.execution_policy.GraspExecutionPolicy`
choreography (pre-open during descent, "close" at contact, retreat lift) drives a suction pick
unchanged: the runner sets ``pre_open_width_mm`` high (vacuum off on the way down) and
``close_width_mm`` low (vacuum on at contact). The gripper also advertises
:class:`~src.robot.core.ObjectDetectingGripper` so the policy's post-close verification polls
the real bond status.

Honest scope: Isaac's surface gripper is a BINARY proximity joint: it models the attach + lift (does a
cup at the contact hold the part through the move), NOT seal quality (air leak / cup deformation). Seal
and wrench *quality* live in the analytical scorer (``grasping/suction/``); this driver realises WHERE
the synthesis chose, in sim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...core import RobotConnectionError

__all__ = [
    "SuctionCupProfile",
    "STANDARD_SUCTION_CUP",
    "SLIM_SUCTION_CUP",
    "IsaacSuctionGripper",
]


@dataclass(frozen=True)
class SuctionCupProfile:
    """A named suction end-effector: cup geometry + vacuum threshold."""

    name: str
    cup_radius_mm: float       # contact-cup radius (the seal footprint / collision + visual radius)
    cup_height_mm: float       # cup depth along the approach axis
    shaft_radius_mm: float     # the stem behind the cup (wrist flange -> cup)
    shaft_length_mm: float     # stem length
    vacuum_on_below_mm: float  # set_width_mm at/below this commands vacuum ON, above it OFF
    visual_usd_asset: str | None = None

    @property
    def max_width_mm(self) -> float:
        """Nominal width band upper bound = the cup contact diameter."""
        return 2.0 * self.cup_radius_mm


#: The default cup: a ~30 mm-diameter contact on a slim shaft (matches the collision-envelope defaults).
STANDARD_SUCTION_CUP = SuctionCupProfile(
    name="standard",
    cup_radius_mm=15.0,
    cup_height_mm=25.0,
    shaft_radius_mm=10.0,
    shaft_length_mm=40.0,
    vacuum_on_below_mm=5.0,
)

#: A finer cup (~20 mm-diameter contact, thin shaft) for tight gaps and small flat faces.
SLIM_SUCTION_CUP = SuctionCupProfile(
    name="slim",
    cup_radius_mm=10.0,
    cup_height_mm=15.0,
    shaft_radius_mm=5.0,
    shaft_length_mm=40.0,
    vacuum_on_below_mm=5.0,
)


class IsaacSuctionGripper:
    """Isaac surface-gripper :class:`Gripper` (vacuum on/off via width reinterpretation).

    Parameters
    ----------
    session
        The shared ``IsaacSimSession`` stepped to let the bond form / release.
    gripper_prim_path
        Prim path of the surface-gripper schema prim (from :func:`mount_suction_cup`).
    profile
        The cup :class:`SuctionCupProfile` (geometry + vacuum threshold); defaults to the standard cup.
    max_width_mm, vacuum_on_below_mm
        Optional overrides; ``None`` uses the profile's values.
    mock_mode
        Pure-Python on/off cache; no Isaac is touched.
    settle_timeout_s
        Max sim seconds to wait for the bond to form after commanding vacuum on.
    """

    def __init__(
        self,
        *,
        session: object,
        gripper_prim_path: str | None,
        profile: SuctionCupProfile = STANDARD_SUCTION_CUP,
        max_width_mm: float | None = None,
        vacuum_on_below_mm: float | None = None,
        mock_mode: bool = False,
        settle_timeout_s: float = 2.0,
    ) -> None:
        self._session = session
        self._prim_path = gripper_prim_path
        self._profile = profile
        self._max_width_mm = float(max_width_mm) if max_width_mm is not None else profile.max_width_mm
        self._vacuum_on_below_mm = (
            float(vacuum_on_below_mm) if vacuum_on_below_mm is not None else profile.vacuum_on_below_mm
        )
        self._mock_mode = mock_mode
        self._settle_timeout_s = settle_timeout_s
        self._connected = False
        self._view: object | None = None
        # Cache: start OFF (open). In mock mode is_object_detected mirrors the last vacuum command.
        self._vacuum_on = False

    # ---- introspection --------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def profile(self) -> SuctionCupProfile:
        return self._profile

    @property
    def min_width_mm(self) -> float:
        return 0.0

    @property
    def max_width_mm(self) -> float:
        return self._max_width_mm

    # ---- lifecycle ------------------------------------------------------

    def connect(self) -> None:
        """Wrap the surface-gripper view (non-mock). Idempotent. Session must be started."""
        if self._connected:
            return
        if self._mock_mode:
            self._connected = True
            return
        if not self._prim_path:
            raise RobotConnectionError("IsaacSuctionGripper requires a gripper_prim_path (non-mock).")
        from isaacsim.robot.surface_gripper import GripperView  # type: ignore[import-not-found]

        self._view = GripperView(paths=self._prim_path)
        self._connected = True

    def disconnect(self) -> None:
        self._view = None
        self._connected = False

    def activate(self) -> None:
        """No-op in sim (a real vacuum generator would spin up here)."""
        return None

    # ---- commands -------------------------------------------------------

    def set_width_mm(
        self,
        width_mm: float,
        *,
        speed: float | None = None,
        force: float | None = None,
    ) -> None:
        """Reinterpret width as vacuum on/off: ``<= vacuum_on_below_mm`` -> ON, else OFF.

        ``speed``/``force`` are accepted for Protocol parity (the surface gripper's force limits are set
        at mount time); not applied here.
        """
        if not self._connected:
            raise RobotConnectionError("IsaacSuctionGripper is not connected.")
        vacuum_on = float(width_mm) <= self._vacuum_on_below_mm
        self._vacuum_on = vacuum_on
        if self._mock_mode or self._view is None:
            return
        # The Isaac surface-gripper action is a signed command: +0.5 = close (engage), -0.5 = open (release)
        # (decoded from the gantry example). +1.0 did NOT transition the status out of Open on-box.
        action = np.array([0.5 if vacuum_on else -0.5])
        self._view.apply_gripper_action(action)  # type: ignore[attr-defined]
        self._settle(expect_closed=vacuum_on)

    def get_width_mm(self) -> float:
        """Nominal width: 0 when vacuum is on (gripping), ``max_width_mm`` when off."""
        if not self._connected:
            raise RobotConnectionError("IsaacSuctionGripper is not connected.")
        return 0.0 if self._vacuum_on else self._max_width_mm

    def open(self) -> None:
        """Release the vacuum (Protocol ``max_width_mm`` -> OFF)."""
        self.set_width_mm(self._max_width_mm)

    def close(self) -> None:
        """Engage the vacuum (Protocol ``min_width_mm`` -> ON)."""
        self.set_width_mm(0.0)

    # ---- ObjectDetectingGripper -----------------------------------------

    def is_object_detected(self) -> bool:
        """``True`` iff the surface gripper has bonded a body (status Closed + a gripped object)."""
        if not self._connected:
            raise RobotConnectionError("IsaacSuctionGripper is not connected.")
        if self._mock_mode or self._view is None:
            return self._vacuum_on
        return self._is_closed()

    # ---- internals ------------------------------------------------------

    def _is_closed(self) -> bool:
        if self._view is None:
            return False
        try:
            status = list(self._view.get_surface_gripper_status())  # type: ignore[attr-defined]
            gripped = list(self._view.get_gripped_objects())  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - status read is best-effort
            return False
        closed = bool(status) and str(status[0]) == "Closed"
        return closed and bool(gripped) and any(bool(g) for g in gripped)

    def _settle(self, *, expect_closed: bool) -> bool:
        """Step the session until the bond forms (vacuum on) / releases (off), or timeout."""
        if self._view is None:
            return True
        cfg = getattr(self._session, "config", None)
        dt = getattr(cfg, "step_dt_s", 0.0) or 1.0 / 60.0
        for _ in range(max(1, int(self._settle_timeout_s / dt))):
            self._session.step()  # type: ignore[attr-defined]
            if self._is_closed() == expect_closed:
                return True
        return False
