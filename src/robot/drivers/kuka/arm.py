"""
KUKA :class:`RobotArm` production driver over EthernetKRL (EKI).

Bridges Willy's vendor-neutral :class:`src.robot.core.RobotArm`
protocol to the KUKA controller via :class:`EkiClient` (TCP/XML). The
driver itself is the **only** module outside :mod:`src.robot.drivers.kuka`
allowed to know about KRL conventions

Numerics
--------
* All :class:`Pose` arguments / returns: millimetres + XYZW quaternion,
  ``frame=Frame.BASE``.
* Joint vectors on the wire are in **degrees** (KUKA convention); the
  driver converts to / from **radians** at the boundary so the
  vendor-neutral :class:`JointPositions` always carries radians.

Capabilities
------------
* Joint moves, Cartesian (LIN / PTP) moves, native FK + IK provided by
  the controller's KRL utilities (round-tripped through EKI).
* No native async API; ``supports_async_move=False``.
* No native force control yet.

The on-controller side is a small KRL program plus an
``EkiHwInterface`` XML config; templates live under
``config/data/robot/templates/kuka/``.
"""

from __future__ import annotations

from config.schema.robot import RobotConfig
from src.geometry import Frame, FrameMismatchError, Pose
from src.robot.constants import HOME_JOINTS_DEFAULT, KUKA_ARM_LOG_FILE, create_robot_logger
from src.robot.core import (
    JointPositions,
    MotionCommand,
    MotionResult,
    MotionStatus,
    RobotArm,
    RobotCapabilities,
    RobotConnectionError,
    RobotKinematicsError,
    RobotMotionRejected,
)
from src.robot.safety.workspace import WorkspaceGuard
from src.robot.safety import SafetyPreflight

from .eki_client import EkiClient
from .pose_convert import (
    joints_deg_to_rad,
    joints_rad_to_deg,
    kuka_cartesian_to_pose,
    pose_to_kuka_cartesian,
)

__all__ = ["KUKA_CAPABILITIES", "KukaRobotArm"]


KUKA_CAPABILITIES = RobotCapabilities(
    vendor="kuka",
    model="kr6-r900",
    dof=6,
    supports_joint_move=True,
    supports_linear_move=True,
    supports_async_move=False,
    has_native_fk=True,
    has_native_ik=True,
    has_force_control=False,
    is_simulated=False,
)


class KukaRobotArm(RobotArm):
    """Production KUKA driver speaking EKI/KRL over TCP/XML.

    Parameters
    ----------
    config
        The full :class:`RobotConfig` tree. ``config.kuka.eki`` selects
        the wire transport; ``config.workspace_limits`` powers the
        internal :class:`WorkspaceGuard`.
    home_joints
        Optional override for the home joint configuration (radians).
        Defaults to :data:`src.robot.constants.HOME_JOINTS_DEFAULT`.
    eki_client
        Optional pre-constructed :class:`EkiClient` mostly useful
        in tests where the transport is mocked. When ``None`` the
        driver builds an :class:`EkiClient` from ``config.kuka.eki``.
    """

    def __init__(
        self,
        config: RobotConfig,
        *,
        home_joints: list[float] | None = None,
        eki_client: EkiClient | None = None,
    ) -> None:
        self.config = config
        self.logger = create_robot_logger("KukaRobotArm", KUKA_ARM_LOG_FILE)

        kuka_cfg = config.kuka
        eki_cfg = kuka_cfg.eki

        if eki_client is None:
            eki_client = EkiClient(
                role=eki_cfg.role,
                host=eki_cfg.host if eki_cfg.role == "server" else kuka_cfg.controller_ip,
                port=eki_cfg.port,
                timeout_s=eki_cfg.timeout_s,
                heartbeat_s=eki_cfg.heartbeat_s,
                buffer_size=eki_cfg.buffer_size,
            )
        self._eki = eki_client

        self._guard = WorkspaceGuard(config.workspace_limits)
        self._home_joints = list(home_joints or HOME_JOINTS_DEFAULT)
        self._preflight = SafetyPreflight.from_safety_config(
            config.safety, config.workspace_limits,
        )

        # Resolve capability descriptor based on configured model / DoF.
        self._capabilities = (
            KUKA_CAPABILITIES
            if (kuka_cfg.model == KUKA_CAPABILITIES.model and kuka_cfg.dof == 6)
            else RobotCapabilities(
                vendor="kuka",
                model=kuka_cfg.model,
                dof=kuka_cfg.dof,
                supports_joint_move=True,
                supports_linear_move=True,
                supports_async_move=False,
                has_native_fk=True,
                has_native_ik=True,
                has_force_control=False,
                is_simulated=False,
            )
        )
        self.eki = eki_cfg

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def capabilities(self) -> RobotCapabilities:
        return self._capabilities

    @property
    def is_connected(self) -> bool:
        return self._eki.is_connected

    @property
    def guard(self) -> WorkspaceGuard:
        return self._guard

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open the EKI link and wait for the first telemetry frame."""
        self._eki.connect()
        # Block briefly for the first ``<State>`` so subsequent
        # ``get_tcp_pose`` / ``get_joint_positions`` calls return real data.
        if not self._eki.wait_for_first_state():
            self.logger.warning(
                "KUKA EKI: no telemetry within timeout; the link is up "
                "but the first <State> frame has not arrived yet.",
            )

    def disconnect(self) -> None:
        self._eki.disconnect()

    def __enter__(self) -> KukaRobotArm:
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_tcp_pose(self) -> Pose:
        if not self._eki.is_connected:
            raise RobotConnectionError("KUKA driver: get_tcp_pose() requires an open EKI link.")
        snapshot = self._eki.get_state()
        if snapshot.pose is None:
            raise RobotConnectionError(
                "KUKA driver: no <State> telemetry has arrived yet."
            )
        return kuka_cartesian_to_pose(snapshot.pose, label="current")

    def get_joint_positions(self) -> JointPositions:
        if not self._eki.is_connected:
            raise RobotConnectionError(
                "KUKA driver: get_joint_positions() requires an open EKI link."
            )
        snapshot = self._eki.get_state()
        if snapshot.joints_deg is None:
            raise RobotConnectionError(
                "KUKA driver: no <State> telemetry has arrived yet."
            )
        return JointPositions(joints_deg_to_rad(snapshot.joints_deg))

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def move_to_joints(
        self,
        joints: JointPositions,
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
    ) -> MotionResult:
        """The KUKA EKI driver carries no preflight, so it simply drives (see Protocol)."""
        if self._preflight is not None:
            rejected = self._preflight.gate_joint_target(joints, arm=self)
            if rejected is not None:
                return rejected
        self.move_joint(joints, velocity=velocity, acceleration=acceleration)
        return MotionResult.executed(
            MotionCommand.MOVE_JOINTS, target_joints=joints, message="move_to_joints",
        )

    def move_joint(
        self,
        joints: JointPositions,
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
    ) -> None:
        if not self._eki.is_connected:
            raise RobotConnectionError("KUKA driver: move_joint() requires an open EKI link.")
        if joints.dof != self._capabilities.dof:
            raise RobotMotionRejected(
                f"KUKA move_joint() expected {self._capabilities.dof} DoF, got {joints.dof}."
            )
        try:
            self._eki.send_movej(
                joints_rad_to_deg(joints.tolist()),
                vel=velocity,
                acc=acceleration,
            )
        except RobotConnectionError as exc:
            raise RobotMotionRejected(f"KUKA move_joint failed: {exc}") from exc

    def move_linear(
        self,
        pose: Pose,
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
    ) -> None:
        if pose.frame is not Frame.BASE:
            raise FrameMismatchError(
                f"KUKA move_linear() requires Frame.BASE; got {pose.frame!r}."
            )
        if not self._eki.is_connected:
            raise RobotConnectionError(
                "KUKA driver: move_linear() requires an open EKI link."
            )
        try:
            self._eki.send_move_cartesian(
                pose_to_kuka_cartesian(pose),
                mode="LIN",
                vel=velocity,
                acc=acceleration,
            )
        except RobotConnectionError as exc:
            raise RobotMotionRejected(f"KUKA move_linear failed: {exc}") from exc

    def stop(self) -> None:
        """Send an immediate stop to the controller. Always safe."""
        if not self._eki.is_connected:
            return
        try:
            self._eki.send_stop()
        except RobotConnectionError as exc:
            self.logger.warning("KUKA stop() send failed: %s", exc)

    def wait_until_steady(
        self,
        timeout_s: float = 5.0,
        poll_interval_s: float = 0.02,
    ) -> bool:
        """Block until the cached telemetry reports ``Steady=1`` or timeout.

        ``poll_interval_s`` is accepted for protocol compatibility but
        ignored: the EKI client wakes the waiter on every incoming
        ``<State>`` frame.
        """
        return self._eki.wait_until_steady(timeout_s)

    # ------------------------------------------------------------------
    # Kinematics: round-tripped through KRL FORWARD/INV
    # ------------------------------------------------------------------

    def fk(self, joints: JointPositions) -> Pose:
        if joints.dof != self._capabilities.dof:
            raise RobotKinematicsError(
                f"KUKA fk() expected {self._capabilities.dof} DoF, got {joints.dof}."
            )
        if not self._eki.is_connected:
            raise RobotConnectionError("KUKA fk() requires an open EKI link.")
        try:
            cart = self._eki.request_fk(joints_rad_to_deg(joints.tolist()))
        except RobotConnectionError as exc:
            raise RobotKinematicsError(f"KUKA FK failed: {exc}") from exc
        return kuka_cartesian_to_pose(cart, label="fk")

    def ik(
        self,
        pose: Pose,
        *,
        seed: JointPositions | None = None,
    ) -> JointPositions:
        if pose.frame is not Frame.BASE:
            raise FrameMismatchError(
                f"KUKA ik() requires Frame.BASE; got {pose.frame!r}."
            )
        if not self._eki.is_connected:
            raise RobotConnectionError("KUKA ik() requires an open EKI link.")
        if seed is None:
            # Use the cached current joints as the IK seed when the
            # caller did not provide one, best chance of nearest-
            # solution selection on a multi-branch IK like KR6-R900.
            try:
                seed_joints = self.get_joint_positions().tolist()
            except RobotConnectionError:
                seed_joints = list(self._home_joints)
        else:
            seed_joints = seed.tolist()
        seed_deg = joints_rad_to_deg(seed_joints)
        try:
            joints_deg = self._eki.request_ik(pose_to_kuka_cartesian(pose), seed_deg)
        except RobotConnectionError as exc:
            raise RobotKinematicsError(f"KUKA IK failed: {exc}") from exc
        joints_rad = joints_deg_to_rad(joints_deg)
        if len(joints_rad) != self._capabilities.dof:
            raise RobotKinematicsError(
                f"KUKA IK returned {len(joints_rad)} joints; expected {self._capabilities.dof}."
            )
        return JointPositions(joints_rad)

    # ------------------------------------------------------------------
    # High-level pipeline surface
    # ------------------------------------------------------------------

    def is_inside_workspace(self, pose: Pose) -> bool:
        if not isinstance(pose, Pose):
            raise TypeError(
                f"KUKA is_inside_workspace expects Pose; got {type(pose).__name__}."
            )
        return self._guard.is_inside_workspace(pose)

    def move_to(
        self,
        pose: Pose,
        *,
        linear: bool = False,
        vel: float | None = None,
        acc: float | None = None,
        register: bool = True,
    ) -> bool:
        """Vendor-neutral "go there" with workspace-guard pre-check.

        Returns ``True`` on controller ack, ``False`` if the pose is
        rejected by the workspace guard or the driver could not reach
        the controller.
        """
        if not self._eki.is_connected:
            self.logger.error("KUKA move_to: link not connected.")
            return False
        if pose.frame is not Frame.BASE:
            raise FrameMismatchError(
                f"KUKA move_to() requires Frame.BASE; got {pose.frame!r}."
            )
        if not self._guard.is_inside_workspace(pose):
            self.logger.warning(
                "KUKA pose '%s' rejected by workspace guard.",
                pose.label or "<unlabeled>",
            )
            return False

        target = pose_to_kuka_cartesian(pose)
        try:
            self._eki.send_move_cartesian(
                target,
                mode="LIN" if linear else "PTP",
                vel=vel,
                acc=acc,
            )
        except RobotConnectionError as exc:
            self.logger.error("KUKA move_to '%s' failed: %s", pose.label, exc)
            return False
        if register:
            self._guard.accept(pose)
        return True

    def move_home(self) -> bool:
        """Move to the home joint configuration."""
        if not self._eki.is_connected:
            self.logger.error("KUKA move_home: link not connected.")
            return False
        try:
            self._eki.send_movej(joints_rad_to_deg(self._home_joints))
        except RobotConnectionError as exc:
            self.logger.error("KUKA move_home failed: %s", exc)
            return False
        return True

    def move(
        self,
        pose: Pose,
        *,
        linear: bool = False,
        vel: float | None = None,
        acc: float | None = None,
        register: bool = True,
    ) -> MotionResult:
        """Typed counterpart of :meth:`move_to`."""
        if pose.frame is not Frame.BASE:
            return MotionResult.failed(
                MotionStatus.INVALID_TARGET,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message=f"KUKA move requires Frame.BASE; got {pose.frame!r}",
            )
        if not self._eki.is_connected:
            return MotionResult.failed(
                MotionStatus.CONNECTION_ERROR,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message="KUKA EKI link not connected",
            )
        target_joints: JointPositions | None = None
        current_joints: JointPositions | None = None
        try:
            target_joints = self.ik(pose)
        except RobotKinematicsError as exc:
            return MotionResult.failed(
                MotionStatus.IK_FAILED,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message=str(exc),
                exception=exc,
            )
        except RobotConnectionError as exc:
            return MotionResult.failed(
                MotionStatus.CONNECTION_ERROR,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message=str(exc),
                exception=exc,
            )
        try:
            current_joints = self.get_joint_positions()
        except RobotConnectionError:
            current_joints = None
        ctx = self._preflight.context_for_pose(
            pose,
            command=MotionCommand.MOVE_TO,
            target_joints=target_joints,
            current_joints=current_joints,
            arm=self,
        )
        decision = self._preflight.evaluate(ctx)
        rejected = SafetyPreflight.as_motion_result(
            decision, MotionCommand.MOVE_TO, target_pose=pose,
        )
        if rejected is not None:
            return rejected
        try:
            ok = self.move_to(
                pose, linear=linear, vel=vel, acc=acc, register=register,
            )
        except RobotConnectionError as exc:
            return MotionResult.failed(
                MotionStatus.CONNECTION_ERROR,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message=str(exc),
                exception=exc,
            )
        return MotionResult.from_bool(
            ok, MotionCommand.MOVE_TO, target_pose=pose,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def eki_client(self) -> EkiClient:
        """Underlying transport (mostly useful in tests)."""
        return self._eki

    @property
    def home_joints(self) -> list[float]:
        return list(self._home_joints)
