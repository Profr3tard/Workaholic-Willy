"""
URConnection — thin RTDE-based wrapper around a Universal Robots controller.

Uses the ``ur_rtde`` library (``pip install ur_rtde``) to communicate with
the robot over the Real-Time Data Exchange (RTDE) protocol.

Provides:
    - connect / disconnect lifecycle (also usable as a context manager)
    - read current TCP pose  (``get_tcp_pose``)
    - read current joint angles  (``get_joint_positions``)
    - forward-kinematics helper  (``fk``)
    - inverse-kinematics helper  (``ik``)
    - low-level move wrappers (``moveJ``, ``moveL``)
"""

from __future__ import annotations

from typing import TYPE_CHECKING



from ...constants import UR_CONNECTION_LOG_FILE, create_robot_logger
from ...core import RobotConnectionError

try:
    import rtde_control  # ur_rtde
    import rtde_receive  # ur_rtde
except ImportError:
    rtde_control = None        # type: ignore[assignment]
    rtde_receive = None        # type: ignore[assignment]

try:
    import rtde_io  # ur_rtde
except ImportError:
    rtde_io = None             # type: ignore[assignment]
try:
    import dashboard_client  # ur_rtde
except ImportError:
    dashboard_client = None    # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - import only for static analysis
    from dashboard_client import DashboardClient
    from rtde_control import RTDEControlInterface
    from rtde_io import RTDEIOInterface
    from rtde_receive import RTDEReceiveInterface


class URConnection:
    """
    Manages the RTDE connection to a UR robot.

    Parameters:
        ip:           IP address of the robot controller.
        vel:          Default joint velocity (rad/s).
        acc:          Default joint acceleration (rad/s²).
        frequency:    RTDE exchange frequency (Hz). 0 = default (125 Hz).
    """

    def __init__(
        self,
        ip: str,
        vel: float = 1.0,
        acc: float = 0.5,
        frequency: float = 0.0,
    ):
        self.ip = ip
        self.vel = vel
        self.acc = acc
        self.frequency = frequency
        self.logger = create_robot_logger("URConnection", UR_CONNECTION_LOG_FILE)

        self._ctrl: RTDEControlInterface | None = None
        self._recv: RTDEReceiveInterface | None = None
        self._io: RTDEIOInterface | None = None
        self._dashboard: DashboardClient | None = None
        #: Cached controller TCP offset for :meth:`_fk_tcp_offset`; cleared on every (dis)connect.
        self._tcp_offset_cache: list[float] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        if rtde_control is None or rtde_receive is None:
            raise ImportError(
                "ur_rtde is not installed. Run: pip install ur_rtde"
            )
        if self.is_connected:
            self.logger.debug("connect() called but already connected, ignored.")
            return
        self._tcp_offset_cache = None
        self.logger.info("Connecting to UR robot at %s …", self.ip)
        try:
            # 0.0 IS OUR "NOT CONFIGURED", AND IT IS NOT ur_rtde's. The schema ships
            # `ur.rtde_frequency: 0.0` with `ge=0.0`, meaning "let the controller decide"; ur_rtde's
            # sentinel for the same thing is -1.0, and it takes 0.0 literally as zero hertz. MEASURED
            # against URSim 5.26.0 on 2026-08-18, one variable at a time:
            #
            #     frequency=0.0   -> FAILED in 6.1 s: "Failed to start RTDE data synchronization"
            #     frequency=-1.0  -> connected in 0.1 s
            #     frequency=500.0 -> connected in 0.6 s
            #
            frequency = float(self.frequency) if float(self.frequency) > 0.0 else -1.0
            self._ctrl = rtde_control.RTDEControlInterface(
                self.ip, frequency,
            )
            self._recv = rtde_receive.RTDEReceiveInterface(self.ip)
        except Exception as exc:  # noqa: BLE001 - roll back a partial connect, then reraise the transport fault
            # Roll back a partially-opened connection.
            self.logger.error("Failed to connect to %s: %s", self.ip, exc)
            self._safe_teardown()
            blocked = self._diagnose_connect_failure()
            if blocked is not None:
                raise RobotConnectionError(f"{blocked} The underlying transport error was: {exc}") from exc
            raise
        if rtde_io is not None:
            try:
                self._io = rtde_io.RTDEIOInterface(self.ip)
            except Exception as exc:  # noqa: BLE001 - I/O is optional; log and continue without it
                self.logger.warning("RTDEIOInterface unavailable (%s); digital/analog I/O disabled.", exc)
        if dashboard_client is not None:
            try:
                self._dashboard = dashboard_client.DashboardClient(self.ip)
                self._dashboard.connect()
            except Exception as exc:  # noqa: BLE001 - dashboard is optional; log and continue without it
                self.logger.warning("DashboardClient unavailable (%s); safety text/recovery disabled.", exc)
                self._dashboard = None
        self.logger.info("Connected (control + receive%s%s).",
                         " + io" if self._io is not None else "",
                         " + dashboard" if self._dashboard is not None else "")

    def disconnect(self) -> None:
        """Idempotent: safe to call multiple times or after a failed connect()."""
        self._safe_teardown()
        self.logger.info("Disconnected.")

    def _safe_teardown(self) -> None:
        """Best-effort release of both RTDE interfaces; never raises."""
        self._tcp_offset_cache = None
        if self._ctrl is not None:
            try:
                self._ctrl.stopScript()
            except Exception as exc:  # noqa: BLE001 - best-effort teardown: log and continue, never raise
                self.logger.warning("stopScript() failed: %s", exc)
            try:
                self._ctrl.disconnect()
            except Exception as exc:  # noqa: BLE001 - best-effort teardown: log and continue, never raise
                self.logger.warning("control.disconnect() failed: %s", exc)
            self._ctrl = None
        if self._recv is not None:
            try:
                self._recv.disconnect()
            except Exception as exc:  # noqa: BLE001 - best-effort teardown: log and continue, never raise
                self.logger.warning("receive.disconnect() failed: %s", exc)
            self._recv = None
        if self._io is not None:
            try:
                self._io.disconnect()
            except Exception as exc:  # noqa: BLE001 - best-effort teardown: log and continue, never raise
                self.logger.warning("io.disconnect() failed: %s", exc)
            self._io = None
        if self._dashboard is not None:
            try:
                self._dashboard.disconnect()
            except Exception as exc:  # noqa: BLE001 - best-effort teardown: log and continue, never raise
                self.logger.warning("dashboard.disconnect() failed: %s", exc)
            self._dashboard = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.disconnect()

    @property
    def is_connected(self) -> bool:
        return self._ctrl is not None and self._recv is not None

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_tcp_pose(self) -> list[float]:
        """
        Current TCP pose as ``[x, y, z, rx, ry, rz]``.

        Translations are in **metres**, rotations are axis-angle (rad).
        """
        self._require_connected()
        assert self._recv is not None  # guaranteed by _require_connected()
        return self._recv.getActualTCPPose()

    def get_joint_positions(self) -> list[float]:
        """Current joint angles ``[q0 … q5]`` in radians."""
        self._require_connected()
        assert self._recv is not None  # guaranteed by _require_connected()
        return self._recv.getActualQ()

    # ------------------------------------------------------------------
    # Kinematics helpers
    # ------------------------------------------------------------------

    #: A TCP offset of exactly zero cannot be passed to ``getForwardKinematics`` -- measured
    #: 2026-08-18, that call fails and on one attempt hung until its timeout. One nanometre in tool Z
    #: is the smallest value that works and is physically nil (1e-9 m against a tolerance measured in
    #: millimetres).
    _FK_EPSILON_OFFSET: "tuple[float, ...]" = (0.0, 0.0, 1e-9, 0.0, 0.0, 0.0)

    def _fk_tcp_offset(self) -> list[float]:
        """The TCP offset to hand :meth:`fk` explicitly, so it never reads a stale register."""
        if self._tcp_offset_cache is None:
            assert self._ctrl is not None  # guaranteed by the caller's _require_connected()
            try:
                offset = [float(v) for v in self._ctrl.getTCPOffset()]
            except Exception as exc:  # noqa: BLE001 - fall back to the epsilon rather than to a stale register
                self.logger.warning(
                    "getTCPOffset() failed (%s); using a 1 nm offset so FK still bypasses the "
                    "stale-register path. A controller with a TCP set would report the FLANGE here.",
                    exc,
                )
                offset = list(self._FK_EPSILON_OFFSET)
            if not any(offset):
                offset = list(self._FK_EPSILON_OFFSET)
            self._tcp_offset_cache = offset
        return list(self._tcp_offset_cache)

    def fk(self, joint_positions: list[float]) -> list[float]:
        """
        Forward kinematics: joints → TCP pose ``[x, y, z, rx, ry, rz]``.

        Uses the controller's built-in FK.
        """
        self._require_connected()
        assert self._ctrl is not None  # guaranteed by _require_connected()
        return self._ctrl.getForwardKinematics(joint_positions, self._fk_tcp_offset())

    def fk_current(self) -> list[float]:
        """The controller's CURRENT TCP via ``getForwardKinematics()`` with NO arguments.

        Deliberately separate from :meth:`fk`, because the two are not equally trustworthy and the
        difference is not academic, it is the difference between a cell that reconnects and one
        that refuses to. MEASURED against URSim 5.26.0 / ur_rtde 1.6.5 on 2026-08-18, comparing both
        against ``getActualTCPPose()`` with the arm steady:

            state                    fk(q)          fk()  (no argument)
            fresh controller         0.000 mm       0.000 mm
            after ONE moveJ        656.780 mm       0.000 mm
            after further motion  1000.093 mm       0.000 mm

        ``getForwardKinematics(q)`` shares the controller's float registers with the motion commands,
        so ANY moveJ/moveL leaves values there that the next q-form call reads as a TCP offset (the
        mechanism GitLab #348 describes: "these can be other values than a tcp offset of zero, because
        of other functions using the same float registers"). The no-argument form does not touch those
        registers and was correct in every state tried.

        The consequence was reproduced end to end: the first ``connect()`` verifies the tool frame at
        0.00 mm, the cell moves once, and the NEXT connect is refused with a nonsense delta of over a
        metre a perfectly good arm turned away on run two.

        Use this wherever "FK of the joints the robot is at right now" is meant. It is NOT a
        substitute for :meth:`fk`, which answers a different question (FK of an ARBITRARY q) and has
        no immune equivalent see the note on that method.
        """
        self._require_connected()
        assert self._ctrl is not None  # guaranteed by _require_connected()
        return list(self._ctrl.getForwardKinematics())

    def ik(
        self,
        tcp_pose: list[float],
        q_near: list[float] | None = None,
    ) -> list[float]:
        """
        Inverse kinematics: TCP pose → joint angles.

        Parameters:
            tcp_pose: ``[x, y, z, rx, ry, rz]``
            q_near:   Seed joints for the nearest IK solution (optional).
        """
        self._require_connected()
        assert self._ctrl is not None  # guaranteed by _require_connected()
        if q_near is not None:
            return self._ctrl.getInverseKinematics(tcp_pose, q_near)
        return self._ctrl.getInverseKinematics(tcp_pose)

    # ------------------------------------------------------------------
    # Motion primitives
    # ------------------------------------------------------------------

    def moveJ(
        self,
        joint_positions: list[float],
        vel: float | None = None,
        acc: float | None = None,
        asynchronous: bool = False,
    ) -> bool:
        """
        Joint-space move.

        Returns ``True`` when the move completes (sync) or starts (async).
        """
        self._require_connected()
        assert self._ctrl is not None  # guaranteed by _require_connected()
        v = vel if vel is not None else self.vel
        a = acc if acc is not None else self.acc
        return self._ctrl.moveJ(joint_positions, v, a, asynchronous)

    def moveL(
        self,
        tcp_pose: list[float],
        vel: float | None = None,
        acc: float | None = None,
        asynchronous: bool = False,
    ) -> bool:
        """
        Linear (Cartesian) move.

        ``tcp_pose`` is ``[x, y, z, rx, ry, rz]`` with metres / axis-angle.
        """
        self._require_connected()
        assert self._ctrl is not None  # guaranteed by _require_connected()
        v = vel if vel is not None else self.vel
        a = acc if acc is not None else self.acc
        return self._ctrl.moveL(tcp_pose, v, a, asynchronous)

    def stop(self, deceleration: float = 2.0) -> None:
        """
        Immediately stop all movement.

        Calls both ``stopJ`` and ``stopL`` so the robot halts regardless of
        whether the active command is a joint-space or a linear move.
        Errors from the controller are logged but never re-raised, stopping
        must be best-effort so cleanup paths never fail.

        Parameters:
            deceleration: Rate in rad/s² (stopJ) / m/s² (stopL).
        """
        self._require_connected()
        assert self._ctrl is not None  # guaranteed by _require_connected()
        try:
            self._ctrl.stopJ(deceleration)
        except Exception as exc:  # noqa: BLE001 - best-effort stop: log and try the next stop variant
            self.logger.warning("stopJ failed: %s", exc)
        try:
            self._ctrl.stopL(deceleration)
        except Exception as exc:  # noqa: BLE001 - best-effort stop: log and continue
            self.logger.warning("stopL failed: %s", exc)

    def is_steady(self) -> bool:
        """True when the robot has finished its current motion."""
        self._require_connected()
        assert self._ctrl is not None  # guaranteed by _require_connected()
        return self._ctrl.isSteady()

    def set_payload(
        self,
        mass_kg: float,
        cog_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Push payload mass + CoG to the controller via RTDE.

        Calls ``RTDEControlInterface.setPayload(mass_kg, cog_m)``. The
        CoG is converted from millimetres to metres at the boundary
        because Willy's vendor-neutral units are mm + kg, while ur_rtde
        expects metres + kg.

        Errors from the controller are surfaced verbatim; the Python
        ``rtde_control`` binding tends to raise :class:`RuntimeError`
        on protocol faults.

        Parameters
        ----------
        mass_kg
            Tool / payload mass in kilograms.
        cog_mm
            Centre-of-gravity offset from the flange in mm. Converted
            to metres for the ``setPayload`` call.
        """
        self._require_connected()
        assert self._ctrl is not None  # guaranteed by _require_connected()
        if mass_kg < 0.0:
            raise ValueError(
                f"set_payload: mass_kg must be >= 0; got {mass_kg}"
            )
        cog_m = (
            float(cog_mm[0]) * 1e-3,
            float(cog_mm[1]) * 1e-3,
            float(cog_mm[2]) * 1e-3,
        )
        self.logger.info(
            "Pushing payload to controller: mass=%.3f kg cog_mm=%s",
            mass_kg, cog_mm,
        )
        self._ctrl.setPayload(float(mass_kg), list(cog_m))

    def wait_until_steady(
        self,
        timeout_s: float = 5.0,
        poll_interval_s: float = 0.02,
    ) -> bool:
        """Block until the robot is steady or ``timeout_s`` elapses.

        This replaces blind ``time.sleep(settle_time_s)`` calls in pipelines
        that need a still frame after a move. Polling at ~50 Hz keeps the
        wait close to the actual settle time and never returns before the
        controller reports ``isSteady`` is true.

        Parameters
        ----------
        timeout_s : float
            Maximum wait in seconds. Must be >= 0; values <= 0 return
            immediately with the current ``is_steady()`` value.
        poll_interval_s : float
            Sleep between polls. Capped at ``timeout_s`` for short waits.

        Returns
        -------
        bool
            ``True`` if the robot became steady before the timeout,
            ``False`` if the timeout fired first.
        """
        import time as _time

        self._require_connected()
        assert self._ctrl is not None  # guaranteed by _require_connected()

        if timeout_s <= 0:
            return bool(self._ctrl.isSteady())

        interval = max(1e-3, min(poll_interval_s, timeout_s))
        deadline = _time.monotonic() + float(timeout_s)
        while True:
            if self._ctrl.isSteady():
                return True
            if _time.monotonic() >= deadline:
                self.logger.warning(
                    "wait_until_steady timed out after %.2fs.", timeout_s,
                )
                return False
            _time.sleep(interval)

    # ------------------------------------------------------------------
    # Digital / analog I/O  (RTDEIOInterface set-side + RTDEReceiveInterface read-side)
    # ------------------------------------------------------------------
    # UR groups digital pins into banks: standard (0-7), configurable (0-7), tool (0-1). The
    # SET side takes a per-bank id via a bank-specific method; the READ side uses a GLOBAL id
    # (standard 0-7, configurable 8-15, tool 16-17). ``_global_din_id`` bridges that asymmetry.
    _PORT_BASE = {"standard": 0, "configurable": 8, "tool": 16}

    def set_digital_out(self, pin: int, value: bool, port: str = "standard") -> None:
        """Drive a digital OUTPUT pin (bank ``port``) to ``value``."""
        self._require_io()
        assert self._io is not None  # guaranteed by _require_io()
        if port == "standard":
            self._io.setStandardDigitalOut(pin, value)
        elif port == "configurable":
            self._io.setConfigurableDigitalOut(pin, value)
        elif port == "tool":
            self._io.setToolDigitalOut(pin, value)
        else:
            raise ValueError(f"set_digital_out: unknown port {port!r}")

    def set_analog_out(self, pin: int, value: float, current: bool = False) -> None:
        """Set an analog OUTPUT: voltage (V) unless ``current=True`` (A)."""
        self._require_io()
        assert self._io is not None  # guaranteed by _require_io()
        if current:
            self._io.setAnalogOutputCurrent(pin, value)
        else:
            self._io.setAnalogOutputVoltage(pin, value)

    def get_digital_in(self, pin: int, port: str = "standard") -> bool:
        """Read a digital INPUT pin's level (bank ``port``)."""
        self._require_connected()
        assert self._recv is not None  # guaranteed by _require_connected()
        return bool(self._recv.getDigitalInState(self._global_din_id(pin, port)))

    def get_digital_out(self, pin: int, port: str = "standard") -> bool:
        """Read back a digital OUTPUT pin's commanded level (bank ``port``)."""
        self._require_connected()
        assert self._recv is not None  # guaranteed by _require_connected()
        return bool(self._recv.getDigitalOutState(self._global_din_id(pin, port)))

    def _global_din_id(self, pin: int, port: str) -> int:
        base = self._PORT_BASE.get(port)
        if base is None:
            raise ValueError(f"digital I/O: unknown port {port!r}")
        return base + int(pin)

    # ------------------------------------------------------------------
    # Force / torque
    # ------------------------------------------------------------------

    def get_tcp_force(self) -> list[float]:
        """Generalized force/torque at the TCP ``[Fx,Fy,Fz,Tx,Ty,Tz]`` (N, N·m; BASE frame)."""
        self._require_connected()
        assert self._recv is not None  # guaranteed by _require_connected()
        return list(self._recv.getActualTCPForce())

    def get_joint_torques(self) -> list[float]:
        """Torque at each joint ``[t0 … t5]`` in newton-metres (from the control interface)."""
        self._require_connected()
        assert self._ctrl is not None  # guaranteed by _require_connected()
        return list(self._ctrl.getJointTorques())

    # ------------------------------------------------------------------
    # Robot / safety status  (RTDEReceiveInterface + DashboardClient)
    # ------------------------------------------------------------------

    def get_robot_mode(self) -> int:
        """Raw UR robot mode int (see RobotMode mapping in the driver)."""
        self._require_connected()
        assert self._recv is not None  # guaranteed by _require_connected()
        return int(self._recv.getRobotMode())

    def get_safety_mode(self) -> int:
        """Raw UR safety mode int (see SafetyMode mapping in the driver)."""
        self._require_connected()
        assert self._recv is not None  # guaranteed by _require_connected()
        return int(self._recv.getSafetyMode())

    def is_protective_stopped(self) -> bool:
        self._require_connected()
        assert self._recv is not None  # guaranteed by _require_connected()
        return bool(self._recv.isProtectiveStopped())

    def is_emergency_stopped(self) -> bool:
        self._require_connected()
        assert self._recv is not None  # guaranteed by _require_connected()
        return bool(self._recv.isEmergencyStopped())

    def get_safety_status_bits(self) -> int:
        self._require_connected()
        assert self._recv is not None  # guaranteed by _require_connected()
        return int(self._recv.getSafetyStatusBits())

    def dashboard_safety_status(self) -> str:
        """Human-readable controller safety text ('' when the dashboard is unavailable)."""
        if self._dashboard is None:
            return ""
        try:
            return str(self._dashboard.safetystatus()).strip()
        except Exception as exc:  # noqa: BLE001 - dashboard is best-effort: return no text on a hiccup
            self.logger.warning("dashboard.safetystatus() failed: %s", exc)
            return ""

    def controller_model(self) -> str | None:
        """What the CONTROLLER says it is, e.g. ``'UR3'``. ``None`` when it will not say.

        A dashboard read, so it costs a socket round trip and is never on the motion path. It exists
        because ``ur.model`` keys the safety DH chain, the exact-mesh collision bundle and the cuRobo
        robot config and nothing in this driver ever asked the controller whether it agreed.

        ``None`` is returned for every failure, INCLUDING an unparseable answer, and callers must
        treat it as "no evidence" rather than as a mismatch.
        """
        if self._dashboard is None:
            return None
        try:
            model = str(self._dashboard.getRobotModel()).strip()
        except Exception as exc:  # noqa: BLE001 - best-effort: no answer is not a wrong answer
            self.logger.warning("dashboard.getRobotModel() failed: %s", exc)
            return None
        return model or None

    def controller_serial(self) -> str | None:
        """The controller's serial number, or ``None``. Best-effort, same contract as above."""
        if self._dashboard is None:
            return None
        try:
            serial = str(self._dashboard.getSerialNumber()).strip()
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("dashboard.getSerialNumber() failed: %s", exc)
            return None
        return serial or None

    def _diagnose_connect_failure(self) -> str | None:
        """Name why the control interface refused, when a dashboard can PROVE it. ``None`` otherwise.

        Both causes below produce the same nameless transport error: "Failed to start RTDE data
        synchronization" or "Failed to start control script, before timeout" and neither is
        derivable from it. Both were measured against URSim on 2026-08-18, and both are things a real
        September cell will do:

        * the controller is in LOCAL mode, so it never runs an externally-sent program;
        * the controller is in a PROTECTIVE STOP, e.g. because the last run ended in one and nobody
          cleared it before restarting the cell.
        """
        if dashboard_client is None:
            return None
        dash = None
        try:
            dash = dashboard_client.DashboardClient(self.ip)
            dash.connect()
            try:
                safety = str(dash.safetystatus()).strip().upper()
            except Exception:  # noqa: BLE001
                safety = ""
            if "PROTECTIVE_STOP" in safety or "EMERGENCY" in safety or "VIOLATION" in safety:
                return (
                    f"Could not open the UR control interface at {self.ip}: the controller reports "
                    f"{safety!r}. A stopped controller will not start an external control program, so "
                    "every motion this cell commands would fail the same way. Clear the stop where "
                    "the arm is VISIBLE, that is a deliberate design decision in this stack, which "
                    "is why nothing here offers to clear it for you, then connect again."
                )
            try:
                if not bool(dash.isInRemoteControl()):
                    return (
                        f"Could not open the UR control interface at {self.ip}: the controller is in "
                        "LOCAL control mode. External control script upload, RTDE control, "
                        "dashboard power/brake/protective-stop commands, requires REMOTE control. "
                        "On the teach pendant: hamburger menu -> Settings -> System -> Remote Control "
                        "-> Enable, then switch the top-right selector from Local to Remote. Note what "
                        "that costs: in Remote mode the pendant is locked for motion (no jogging, no "
                        "freedrive, no manual program start) because ISO requires a single source of "
                        "control, switch back to Local to teach."
                    )
            except Exception:  # noqa: BLE001 - cannot ask => cannot claim
                return None
            return None
        except Exception:  # noqa: BLE001
            return None
        finally:
            if dash is not None:
                try:
                    dash.disconnect()
                except Exception:  # noqa: BLE001 - teardown of a diagnostic must not mask the fault
                    pass

    def is_in_remote_control(self) -> bool | None:
        """Is the controller in REMOTE control? ``None`` when no dashboard can answer.

        Three-valued on purpose. ``False`` means the controller said so and external control will be
        refused; ``None`` means nobody could be asked, which is a different fact and must not be
        reported as "local" a caller that refuses on an unanswered question refuses cells that are
        perfectly fine.
        """
        if self._dashboard is None:
            return None
        try:
            return bool(self._dashboard.isInRemoteControl())
        except Exception as exc:  # noqa: BLE001 - an unanswerable question is not a "no"
            self.logger.warning("dashboard.isInRemoteControl() failed: %s", exc)
            return None

    def _remote_control_is_off(self) -> bool:
        """``True`` only when a throwaway dashboard connection PROVES the controller is in Local mode.

        Used on the connect-failure path, where ``self._dashboard`` does not exist yet: the control
        interface is built before the dashboard, so the diagnosis has to open its own.
        """
        if dashboard_client is None:
            return False
        dash = None
        try:
            dash = dashboard_client.DashboardClient(self.ip)
            dash.connect()
            return not bool(dash.isInRemoteControl())
        except Exception:  # noqa: BLE001 - cannot ask => cannot claim
            return False
        finally:
            if dash is not None:
                try:
                    dash.disconnect()
                except Exception:  # noqa: BLE001 - teardown of a diagnostic must not mask the fault
                    pass

    def unlock_protective_stop(self) -> bool:
        """Ask the dashboard to release a protective stop; ``True`` if a dashboard is connected."""
        if self._dashboard is None:
            return False
        try:
            self._dashboard.closeSafetyPopup()
            self._dashboard.unlockProtectiveStop()
            return True
        except Exception as exc:  # noqa: BLE001 - surface the failure as "not recovered", never raise
            self.logger.warning("unlockProtectiveStop() failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _require_connected(self) -> None:
        if not self.is_connected:
            raise RuntimeError("Not connected — call connect() first.")

    def _require_io(self) -> None:
        if self._io is None:
            raise RuntimeError(
                "Digital/analog I/O unavailable, the RTDEIOInterface did not come up "
                "(is ur_rtde's rtde_io present + the controller reachable?)."
            )
