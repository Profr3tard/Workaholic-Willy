"""Universal Robots :class:`RobotArm` implementation."""

from __future__ import annotations

import dataclasses
import re

import numpy as np

from typing import TYPE_CHECKING

from config.schema.robot import RobotConfig
from src.geometry import Frame, FrameMismatchError, Pose
from src.robot.constants import HOME_JOINTS_DEFAULT, UR_ARM_LOG_FILE, create_robot_logger
from src.robot.core import (
    NO_PLAN_FAIL_SAFE_MESSAGE,
    DigitalIOPort,
    JointPositions,
    MotionCommand,
    MotionResult,
    MotionStatus,
    RobotArm,
    RobotCapabilities,
    RobotConnectionError,
    RobotEmergencyStop,
    RobotKinematicsError,
    RobotMode,
    RobotMotionRejected,
    RobotStatus,
    SafetyMode,
    Wrench,
)
from src.robot.grippers import GripperController
from src.robot.safety import SafetyPreflight
from src.robot.safety.planning import CuroboPlanClient, CuroboUnavailableError
from src.robot.safety.workspace import WorkspaceGuard

from .connection import URConnection
from .motion import MotionController
from .pose import URPose
from .pose_adapter import pose_to_urpose, urpose_to_pose

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

    from .curobo_motion import CuroboUrPlanner

__all__ = ["UR_CAPABILITIES", "URRobotArm", "ur_capabilities"]


def ur_capabilities(model: str = "ur5e") -> RobotCapabilities:
    """This vendor's capability surface for ``model``.

    The capability record is what downstream code reads to learn WHICH robot it is holding, and a
    hardcoded "ur5e" made every model-keyed lookup (DH table, mesh bundle, cuRobo config) resolve the
    wrong robot for a real UR3e. Everything except the model string is shared across the e-series.
    """
    return dataclasses.replace(UR_CAPABILITIES, model=model)


UR_CAPABILITIES = RobotCapabilities(
    vendor="ur",
    model="ur5e",
    dof=6,
    supports_joint_move=True,
    supports_linear_move=True,
    supports_async_move=True,
    has_native_fk=True,
    has_native_ik=True,
    has_force_control=True,  # backed: get_tcp_wrench / get_joint_torques via SupportsForceTorque
    is_simulated=False,
)

# UR controller mode integers -> vendor-neutral enums. UR getRobotMode: -1=NO_CONTROLLER,
# 0=DISCONNECTED … 8=UPDATING_FIRMWARE; getSafetyMode: 1=NORMAL … 9=FAULT (10-13 = validate /
# undefined / auto-mode / three-position stops -> UNKNOWN here). Unmapped ints fall back to UNKNOWN.
_UR_ROBOT_MODE: dict[int, RobotMode] = {
    0: RobotMode.DISCONNECTED, 1: RobotMode.CONFIRM_SAFETY, 2: RobotMode.BOOTING,
    3: RobotMode.POWER_OFF, 4: RobotMode.POWER_ON, 5: RobotMode.IDLE,
    6: RobotMode.BACKDRIVE, 7: RobotMode.RUNNING, 8: RobotMode.UPDATING_FIRMWARE,
}
_UR_SAFETY_MODE: dict[int, SafetyMode] = {
    1: SafetyMode.NORMAL, 2: SafetyMode.REDUCED, 3: SafetyMode.PROTECTIVE_STOP,
    4: SafetyMode.RECOVERY, 5: SafetyMode.SAFEGUARD_STOP, 6: SafetyMode.SYSTEM_EMERGENCY_STOP,
    7: SafetyMode.ROBOT_EMERGENCY_STOP, 8: SafetyMode.VIOLATION, 9: SafetyMode.FAULT,
}


class URRobotArm(RobotArm):
    """
    UR-backed arm driver implementation.

    The class satisfies the vendor-neutral :class:`RobotArm` protocol while
    preserving the existing UR convenience API used by the current pipelines.
    """

    def __init__(
        self,
        config: RobotConfig,
        home_joints: list[float] | None = None,
        *,
        curobo_client_factory: Callable[[], CuroboPlanClient] | None = None,
    ) -> None:
        self.config = config
        self.logger = create_robot_logger("URRobotArm", UR_ARM_LOG_FILE)

        self._conn = URConnection(
            ip=config.ur.ip,
            vel=config.motion_limits.max_velocity,
            acc=config.motion_limits.max_acceleration,
            frequency=config.ur.rtde_frequency,
        )
        self._guard = WorkspaceGuard(config.workspace_limits)
        self._motion = MotionController(
            self._conn,
            self._guard,
            max_velocity=config.motion_limits.max_velocity,
            max_acceleration=config.motion_limits.max_acceleration,
        )
        # Vendor-neutral safety preflight pipeline. The
        # preflight owns its OWN WorkspaceGuard (shrunk by the
        # configured ``limits.workspace_margin_mm``) so the typed
        # ``move()`` path enforces the tighter operator-defined margin
        # while ``MotionController`` keeps the original full-box guard
        # as a backstop for the bool path.
        self._preflight = SafetyPreflight.from_safety_config(
            config.safety, config.workspace_limits,
        )
        # Explicit argument > the cell's own config > the UR5e-authored constant. A real cell that
        # declares nothing still gets a home pose, but it gets a WARNED one (see move_home).
        self._home_joints = (
            list(home_joints) if home_joints
            else list(config.home_joint_positions or HOME_JOINTS_DEFAULT)
        )
        self._gripper = GripperController(config.gripper, ip=config.ur.ip)
        self._capabilities = ur_capabilities(config.ur.model)
        # Real-UR cuRobo binding: "ik" (default) uses the controller IK path; "curobo" plans a
        # collision-free trajectory via safety.planning and executes it over ur_rtde (fail-closed).
        self._motion_planner = config.ur.motion_planner
        self._curobo_client_factory = curobo_client_factory
        self._curobo_ur: CuroboUrPlanner | None = None

    # ------------------------------------------------------------------
    # Tool frame (flange <-> TCP)
    # ------------------------------------------------------------------
    #
    # Every Pose crossing the RobotArm Protocol is the TCP, the grasp centre. What the CONTROLLER
    # calls the TCP depends on its tool register, and ``gripper.tool_frame.source`` declares who owns
    # that fact:
    #   "willy"     the controller runs a bare flange and THIS DRIVER composes the transform, the
    #                  way the Isaac driver already does. No vendor SDK call is involved.
    #   "polyscope" the controller holds it; poses pass through untouched.
    # Either way connect() DERIVES what the controller is really using and refuses a mismatch.

    def _declared_tool_matrix(self) -> "np.ndarray | None":
        """The DECLARED flange->TCP 4x4, regardless of who applies it. None while undeclared.

        Two consumers want different things and the difference is not cosmetic:
        :meth:`_pose_to_controller` needs it only in ``willy`` mode (in ``polyscope`` the controller
        applies it), while the cuRobo path needs it ALWAYS -- see :meth:`_pose_to_flange`.
        """
        tf = self.config.gripper.tool_frame
        if tf.source == "undeclared":
            return None
        from .tool_frame import tool_frame_matrix

        return tool_frame_matrix(tf.offset_mm, tf.rotation_quat_xyzw)

    def _pose_to_flange(self, pose: Pose) -> Pose:
        """TCP -> FLANGE (tool0), using the declared transform whatever ``source`` says.

        cuRobo is why this is unconditional. Its kinematic model ends at ``tool0`` by construction 
        ``docs/curobo/build_ur5e_config.py`` attaches the 2F-85 collision spheres TO ``tool0`` and the
        planner's own docstring says "Plan tool0 -> pose" so it never consults the controller's tool
        register, and ``polyscope`` mode does not reach it.
        """
        t = self._declared_tool_matrix()
        if t is None:
            return pose
        from src.geometry.matrix import invert_homogeneous

        m = np.asarray(pose.to_matrix(), dtype=np.float64) @ invert_homogeneous(t)
        return Pose.from_matrix(m, frame=pose.frame, label=pose.label)

    def _pose_from_controller(self, urpose: URPose, *, label: str) -> Pose:
        """Controller frame -> TCP. Identity unless this driver owns the tool frame."""
        pose = urpose_to_pose(urpose, frame=Frame.BASE, label=label)
        t = self._declared_tool_matrix() if self.config.gripper.tool_frame.source == "willy" else None
        if t is None:
            return pose
        m = np.asarray(pose.to_matrix(), dtype=np.float64) @ t
        return Pose.from_matrix(m, frame=Frame.BASE, label=label)

    def _pose_to_controller(self, pose: Pose) -> URPose:
        """TCP -> the frame the CONTROLLER expects. Identity unless this driver owns the tool frame."""
        if self.config.gripper.tool_frame.source != "willy":
            return pose_to_urpose(pose)   # polyscope applies it controller-side; undeclared never moves
        return pose_to_urpose(self._pose_to_flange(pose))

    #: Size classes this driver can compare across the two spellings a controller and a config use.
    _MODEL_SIZES = ("3", "5", "10", "16", "20", "30")

    @classmethod
    def _model_size(cls, name: str | None) -> str | None:
        """``'UR3'`` / ``'ur3e'`` / ``'UR5e CB3'`` -> the size digits. ``None`` when unrecognisable."""
        if not name:
            return None
        match = re.search(r"ur\s*(\d+)", str(name), flags=re.IGNORECASE)
        if match is None:
            return None
        size = match.group(1)
        return size if size in cls._MODEL_SIZES else None

    def _verify_controller_model(self) -> None:
        """Refuse when the CONTROLLER says it is a different size of robot than ``ur.model`` claims.

        ``ur.model`` keys the safety DH chain, the exact-mesh collision bundle and the cuRobo robot
        config. A UR3e planned against UR5e link lengths (a2/a3 -243.55/-213.2 mm vs -425/-392.2) is
        precisely the failure that field exists to prevent, and until now nothing compared the two:
        the mismatch surfaced only downstream, as a tool-frame refusal quoting a distance nobody could
        act on.

        **Fail-closed on evidence, fail-open on its absence.
        """
        declared = self._model_size(self._capabilities.model)
        reported_raw = self._conn.controller_model()
        reported = self._model_size(reported_raw)
        if declared is None or reported is None:
            self.logger.info(
                "controller model not compared: config says %r, controller says %r one of the two "
                "is not a size this driver recognises, so there is no evidence of a mismatch",
                self._capabilities.model, reported_raw,
            )
            return
        if declared == reported:
            self.logger.info(
                "controller model verified: config %r matches the controller's %r",
                self._capabilities.model, reported_raw,
            )
            return
        raise RobotConnectionError(
            f"ur.model is {self._capabilities.model!r}, but this controller reports "
            f"{reported_raw!r}, a UR{reported} is not a UR{declared}. That field keys the safety DH "
            f"chain, the collision mesh bundle and the cuRobo robot config, so every pose this stack "
            f"plans would be computed against another robot's link lengths. Point ur.model at the arm "
            f"that is actually plugged in."
        )

    def _verify_tool_frame(self) -> None:
        """Refuse to stay connected when the controller's ACTUAL tool frame is not what we assume.

        Derived, never read: ``inv(base->flange from the bundled DH table) @ getForwardKinematics(q)``.
        """
        from .tool_frame import compare_tool_frames, derive_active_tool_frame, tool_frame_matrix

        tf = self.config.gripper.tool_frame
        declared = tool_frame_matrix(tf.offset_mm, tf.rotation_quat_xyzw)
        # "willy" composes on our side, so the controller itself must carry NO tool.
        expected = np.eye(4, dtype=np.float64) if tf.source == "willy" else declared
        # The controller's tool frame is derived from the DH table and the current joint vector, so it is
        # meaningless while the arm is moving.
        if not self._conn.is_steady():
            raise RobotConnectionError(
                "cannot verify the tool frame while the arm is moving: the check pairs the joint "
                "vector with the controller's current pose, and those must describe the same "
                "instant. Let the arm come to rest and connect again."
            )
        joints = list(self._conn.get_joint_positions())
        observed = derive_active_tool_frame(
            self._capabilities.model, joints,
            URPose.from_ur_list(self._conn.fk_current()).to_T(),
        )
        if observed is None:
            raise RobotConnectionError(
                f"cannot verify the tool frame: no bundled DH table for ur.model "
                f"{self._capabilities.model!r}, so the controller's active tool frame is not "
                f"derivable. Set a supported model, or gripper.tool_frame.source: undeclared while "
                f"this cell is being built (which refuses to connect for a different, louder reason)."
            )
        d_t, d_r = compare_tool_frames(observed, expected)
        if d_t > tf.verify_tolerance_mm or d_r > 5.0:
            # No disconnect here: connect() rolls the connection back for any failure of this check.
            other = "polyscope" if tf.source == "willy" else "willy"
            raise RobotConnectionError(
                f"gripper.tool_frame.source is {tf.source!r}, which expects the controller's tool "
                f"register to hold {'nothing (a bare flange)' if tf.source == 'willy' else 'the declared transform'}"
                f" but the controller is actually running a tool frame {d_t:.1f} mm / {d_r:.1f} deg "
                f"away from that (tolerance {tf.verify_tolerance_mm:.1f} mm). Commanding motion now "
                f"would drive the arm off by that much on every pose. Either the pendant's TCP is not "
                f"what this config says, or the cell is really in {other!r} mode, or ur.model "
                f"({self._capabilities.model!r}) is not the arm on the other end, this is derived "
                f"from THAT model's DH table, so a wrong model produces a wrong distance here. "
                f"Derived as inv(DH flange) @ getForwardKinematics, no PolyScope read required."
            )
        self.logger.info(
            "tool frame verified: source=%s, controller matches within %.2f mm / %.2f deg",
            tf.source, d_t, d_r,
        )

    def connect(self) -> None:
        """Open the RTDE connection to the robot controller.

        After the connection is up the configured payload envelope is
        pushed to the controller via ``setPayload`` so the
        controller's protective-stop calculations reflect the same
        mass + CoG Willy is enforcing. The push is gated on
        ``config.safety.payload.enforce``, when the operator has
        disabled the payload guard the driver does not touch the
        controller payload either.

        Fails closed on a contradictory payload config BEFORE opening the socket, on TWO counts:
        ``enforce: true`` with ``mass_kg: 0.0`` (the shipped default) would push ``setPayload(0.0)``
        and silently overwrite the controller's payload for a mounted tool; ``mass_kg > 0`` with the
        default ``cog_mm: [0, 0, 0]`` would declare that tool a point mass at the flange face. Both
        are fail-OPEN in the dangerous direction, and both corrupt the controller's protective-stop
        model AND its gravity compensation (hence ``get_tcp_wrench``). A bare flange is expressed by
        ``enforce: false``, which never touches the controller payload.
        """
        payload = self.config.safety.payload
        if payload.enforce and payload.mass_kg == 0.0:
            raise RobotConnectionError(
                "safety.payload has enforce: true but mass_kg: 0.0, connecting would push "
                "setPayload(0.0 kg) and overwrite the controller's configured payload for a mounted "
                "tool, so its protective-stop model would under-read the real mass. "
                "WEIGH THE WHOLE ASSEMBLY, not the gripper: the coupling/adapter plate, every cable and "
                "hose that rides on the wrist, and any workpiece the arm carries. A gripper's datasheet "
                "mass is not this number and no value is shipped here on purpose, a plausible default "
                "is how a cell ends up telling the controller about a tool it is not carrying. Set "
                "safety.payload.mass_kg and cog_mm from the scale, or set safety.payload.enforce: false "
                "if the flange is genuinely bare."
            )
        if payload.enforce and payload.mass_kg > 0.0 and all(v == 0.0 for v in payload.cog_mm):
            raise RobotConnectionError(
                f"safety.payload declares mass_kg: {payload.mass_kg} but leaves cog_mm at its "
                "[0, 0, 0] default, that is the 'not measured yet' marker, and pushing it would "
                "tell the controller the tool is a point mass at the flange face. Its protective-stop "
                "model and its gravity compensation (and therefore get_tcp_wrench) would both be "
                "computed against a tool that cannot exist. Measure the CoG offset from the flange in "
                "mm, the same bench step that gives you the tool frame and set "
                "safety.payload.cog_mm, or set safety.payload.enforce: false for a genuinely bare "
                "flange. A deliberately axis-centred tool is still not [0, 0, 0]: state its real "
                "stand-off, e.g. [0, 0, 60]."
            )
        # The tool frame must be DECLARED before a real arm is allowed to move.
        tool = self.config.gripper.tool_frame
        if tool.source == "undeclared":
            raise RobotConnectionError(
                "gripper.tool_frame.source is 'undeclared', nobody has said where this gripper's "
                "grasp centre sits on the flange, so the driver cannot know whether the poses it "
                "commands mean the TCP or the flange. Measure the coupling + gripper (the same bench "
                "step as safety.payload.cog_mm) and set gripper.tool_frame.offset_mm + "
                "rotation_quat_xyzw, then choose source: 'willy' (this driver composes the transform "
                "and the controller runs a bare flange) or 'polyscope' (an operator set the TCP on the "
                "teach pendant and the driver only verifies it)."
            )
        self._conn.connect()
        if payload.enforce:
            try:
                self._conn.set_payload(payload.mass_kg, payload.cog_mm)
            except BaseException as exc:  # noqa: BLE001 - the rollback below is the whole point
                # Roll back the connection: an unverified payload on a
                # connected controller is more dangerous than no
                # connection at all.
                self.logger.error(
                    "UR setPayload failed; rolling back the connection: %s",
                    exc,
                )
                self._conn.disconnect()
                if isinstance(exc, Exception):
                    raise RobotConnectionError(
                        f"UR setPayload({payload.mass_kg} kg) failed: {exc}"
                    ) from exc
                raise  # KeyboardInterrupt / SystemExit propagate unchanged, but disconnected first
        # Last, because it needs a live connection: confirm the controller's ACTUAL tool frame is the
        # one this config assumes. Fails closed.
        try:
            self._verify_controller_model()
            self._verify_tool_frame()
        except BaseException:
            self._conn.disconnect()
            raise

    def disconnect(self) -> None:
        """Close the RTDE connection (and shut down the cuRobo planning server if one was started)."""
        if self._curobo_ur is not None:
            self._curobo_ur.close()
            self._curobo_ur = None
        self._conn.disconnect()

    def __enter__(self) -> URRobotArm:
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()

    @property
    def is_connected(self) -> bool:
        return self._conn.is_connected

    @property
    def pose(self) -> URPose:
        """Current TCP pose (mm / axis-angle rad)."""
        return self._motion.get_current_pose()

    @property
    def joints(self) -> list[float]:
        """Current joint angles in radians."""
        return self._conn.get_joint_positions()

    @property
    def tcp_raw(self) -> list[float]:
        """Raw UR TCP ``[x_m, y_m, z_m, rx, ry, rz]``."""
        return self._conn.get_tcp_pose()

    def move_to(
        self,
        pose: Pose | URPose,
        *,
        linear: bool = False,
        vel: float | None = None,
        acc: float | None = None,
        register: bool = True,
    ) -> bool:
        """Move to ``pose`` with workspace-guard safety check.

        Accepts either a vendor-neutral :class:`Pose` (preferred, used
        by pipelines and the API) or a :class:`URPose` (still
        used internally by tests and the calibration code path).
        """
        return self._motion.move_to(
            self._coerce_urpose(pose),
            linear=linear, vel=vel, acc=acc, register=register,
        )

    def is_inside_workspace(self, pose: Pose | URPose) -> bool:
        """Vendor-neutral workspace pre-check used by pipelines.

        Accepts either :class:`Pose` (BASE frame) or :class:`URPose`.
        """
        return self._guard.is_inside_workspace(self._coerce_urpose(pose))

    @staticmethod
    def _coerce_urpose(pose: Pose | URPose) -> URPose:
        """Convert a :class:`Pose` to :class:`URPose` if needed."""
        if isinstance(pose, URPose):
            return pose
        if pose.frame is not Frame.BASE:
            raise FrameMismatchError(
                f"URRobotArm requires Frame.BASE; got {pose.frame!r}."
            )
        return pose_to_urpose(pose)

    def move_home(self) -> bool:
        """Move to the home joint configuration, GATED. ``False`` = refused, nothing moved.
        Returns ``bool`` the typed reason is logged.
        """
        joints = JointPositions(np.asarray(self._home_joints, dtype=np.float64))
        if self._preflight is not None:
            rejected = self._preflight.gate_joint_target(joints, arm=self)
            if rejected is not None:
                self.logger.error(
                    "move_home REFUSED by %s: %s", rejected.status, rejected.message,
                )
                return False
        if self._conn.is_connected:
            try:
                home_pose = URPose.from_ur_list(self._conn.fk(list(self._home_joints)), label="home")
                if not self._guard.is_inside_workspace(home_pose):
                    self.logger.error(
                        "move_home REFUSED: the home pose (%.1f, %.1f, %.1f) is outside "
                        "workspace_limits. Set robot.home_joint_positions to a configuration inside "
                        "this cell's own box, the shipped default was authored for a UR5e.",
                        home_pose.x, home_pose.y, home_pose.z,
                    )
                    return False
            except Exception as exc:  # noqa: BLE001 - FK unavailable must not silently wave the move through
                self.logger.error("move_home REFUSED: could not check the home pose (%s)", exc)
                return False
        return self._motion.move_home(self._home_joints)

    def stop(self) -> None:
        """Emergency stop."""
        self._conn.stop()

    def wait_until_steady(
        self,
        timeout_s: float = 5.0,
        poll_interval_s: float = 0.02,
    ) -> bool:
        """Delegate to the underlying RTDE :meth:`URConnection.wait_until_steady`."""
        if not self._conn.is_connected:
            raise RobotConnectionError("wait_until_steady() requires an open connection.")
        return bool(self._conn.wait_until_steady(timeout_s, poll_interval_s))

    def move(
        self,
        pose: Pose,
        *,
        linear: bool = False,
        vel: float | None = None,
        acc: float | None = None,
        register: bool = True,
    ) -> MotionResult:
        """Typed counterpart of :meth:`move_to`.

        Every commanded pose is routed through the vendor-neutral
        :class:`SafetyPreflight` pipeline. A rejection from
        any guard produces the corresponding typed
        :class:`MotionStatus` (``WORKSPACE_REJECTED``,
        ``JOINT_LIMIT_REJECTED``, ``IK_QUALITY_REJECTED``,
        ``SELF_COLLISION_REJECTED``, ``PAYLOAD_REJECTED``,
        ``CONTINUITY_REJECTED``) so callers can branch on the precise
        cause without log scraping. Connection faults raised by the
        underlying bool path are caught and reported as
        :attr:`MotionStatus.CONNECTION_ERROR`.
        """
        if pose.frame is not Frame.BASE:
            return MotionResult.failed(
                MotionStatus.INVALID_TARGET,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message=f"URRobotArm.move requires Frame.BASE; got {pose.frame!r}",
            )
        if self._motion_planner == "curobo":
            return self._drive_curobo(pose, vel=vel, acc=acc)
        # Pre-resolve IK on the driver
        # side so the preflight pipeline sees ``target_joints`` for
        # every Cartesian command.
        target_joints: JointPositions | None = None
        current_joints: JointPositions | None = None
        if self._conn.is_connected:
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
                # Telemetry hiccup mid-call; the IK-jump check will
                # simply skip rather than fail closed because the
                # ``current_joints`` is None.
                current_joints = None
        try:
            ctx = self._preflight.context_for_pose(
                pose,
                command=MotionCommand.MOVE_TO,
                target_joints=target_joints,
                current_joints=current_joints,
                arm=self,
            )
            decision = self._preflight.evaluate(ctx)
        except RobotConnectionError as exc:
            return MotionResult.failed(
                MotionStatus.CONNECTION_ERROR,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message=str(exc),
                exception=exc,
            )
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
        # Attach the TRUE rejection cause the inner controller classified (e.g. a real
        # near-singularity -> IK_QUALITY_REJECTED), not from_bool's generic CONTROLLER_REJECTED default.
        failure_status = (
            self._motion.last_reject_status
            if self._motion.last_reject_status is not None
            else MotionStatus.CONTROLLER_REJECTED
        )
        return MotionResult.from_bool(
            ok, MotionCommand.MOVE_TO, target_pose=pose, failure_status=failure_status,
        )

    def _drive_curobo(
        self,
        pose: Pose,
        *,
        vel: float | None = None,
        acc: float | None = None,
    ) -> MotionResult:
        """Plan tool0 -> ``pose`` with cuRobo, gate the planned final config, then execute (fail-closed).

        cuRobo unavailable -> ``CONTROLLER_REJECTED``; no collision-free plan -> ``TIMEOUT``; a static
        safety-guard rejection of cuRobo's ACTUAL final config -> the matching status. No blind motion.
        """
        if not self._conn.is_connected:
            return MotionResult.failed(
                MotionStatus.CONNECTION_ERROR, MotionCommand.MOVE_TO, target_pose=pose,
                message="cuRobo move requires an open connection.",
            )
        planner = self._curobo_ur_planner()
        # cuRobo plans to tool0, so it gets the FLANGE goal -- in BOTH ownership modes.
        goal = self._pose_to_flange(pose)
        try:
            traj_ur = planner.plan(goal)
        except CuroboUnavailableError as exc:
            return MotionResult.failed(
                MotionStatus.CONTROLLER_REJECTED, MotionCommand.MOVE_TO, target_pose=pose,
                message=f"cuRobo planner unavailable — {exc}", exception=exc,
            )
        if not traj_ur:
            return MotionResult.failed(
                MotionStatus.TIMEOUT, MotionCommand.MOVE_TO, target_pose=pose,
                message=NO_PLAN_FAIL_SAFE_MESSAGE,
            )
        rejected = self._preflight.gate_joint_target(JointPositions(traj_ur[-1]), arm=self)
        if rejected is not None:
            return rejected
        return planner.execute(traj_ur, pose, vel=vel, acc=acc)

    def _curobo_ur_planner(self) -> CuroboUrPlanner:
        """Lazily build the real-UR cuRobo execution glue bound to this arm's RTDE connection."""
        if self._curobo_ur is None:
            from .curobo_motion import CuroboUrPlanner

            # The planner must be built for THIS robot.
            self._curobo_ur = CuroboUrPlanner(
                self._conn,
                client_factory=self._curobo_client_factory or self._default_curobo_client_factory(),
                vel=self.config.motion_limits.max_velocity,
                acc=self.config.motion_limits.max_acceleration,
            )
        return self._curobo_ur

    def _default_curobo_client_factory(self) -> Callable[[], CuroboPlanClient]:
        """A cuRobo client bound to this cell's ROBOT MODEL and its guard's planner margin.

        Mirrors the sim driver: the configured model wins, and the clearance the SelfCollisionGuard will
        demand is handed to the planner so it stops returning configurations the guard would reject.
        """
        from src.robot.drivers.sim.robot_models import curobo_robot_yml

        robot_yml = curobo_robot_yml(self.config.ur.model)
        margin_mm = float(getattr(self.config.safety.self_collision, "planner_margin_mm", 0.0) or 0.0)
        return lambda: CuroboPlanClient(robot_config=robot_yml, self_collision_margin_mm=margin_mm)

    async def amove_to(
        self,
        pose: Pose | URPose,
        *,
        linear: bool = False,
        vel: float | None = None,
        acc: float | None = None,
        register: bool = True,
    ) -> bool:
        """Awaitable variant of :meth:`move_to`."""
        return await self._motion.amove_to(
            self._coerce_urpose(pose),
            linear=linear, vel=vel, acc=acc, register=register,
        )

    async def amove_home(self) -> bool:
        """Awaitable variant of :meth:`move_home`."""
        return await self._motion.amove_home(self._home_joints)

    def fk(self, joints: JointPositions) -> Pose:
        """Forward kinematics: :class:`JointPositions` to TCP :class:`Pose`."""
        if not self._conn.is_connected:
            raise RobotConnectionError("fk() requires an open connection.")
        if joints.dof != self.capabilities.dof:
            raise RobotKinematicsError(
                f"fk() expected {self.capabilities.dof} DoF, got {joints.dof}."
            )
        try:
            tcp_m = self._conn.fk(joints.tolist())
        except (RuntimeError, OSError) as exc:
            raise RobotKinematicsError(f"FK failed: {exc}") from exc
        urpose = URPose.from_ur_list(tcp_m, label="fk")
        return self._pose_from_controller(urpose, label="fk")

    def ik(
        self,
        pose: Pose,
        *,
        seed: JointPositions | None = None,
    ) -> JointPositions:
        """Inverse kinematics: TCP :class:`Pose` in :attr:`Frame.BASE` to joints."""
        if pose.frame is not Frame.BASE:
            raise FrameMismatchError(
                f"ik() requires pose in Frame.BASE; got {pose.frame!r}."
            )
        if not self._conn.is_connected:
            raise RobotConnectionError("ik() requires an open connection.")
        urpose = self._pose_to_controller(pose)
        seed_list = seed.tolist() if seed is not None else None
        try:
            joints = self._conn.ik(urpose.to_ur_list(), q_near=seed_list)
        except (RuntimeError, OSError) as exc:
            raise RobotKinematicsError(f"IK failed: {exc}") from exc
        if not joints or len(joints) != self.capabilities.dof:
            raise RobotKinematicsError(
                f"IK returned no valid solution for pose '{pose.label or '<unlabeled>'}'."
            )
        return JointPositions(joints)

    @property
    def capabilities(self) -> RobotCapabilities:
        """Static feature flags advertised by this driver."""
        return self._capabilities

    def get_tcp_pose(self) -> Pose:
        """Current TCP pose, tagged :attr:`Frame.BASE`."""
        if not self._conn.is_connected:
            raise RobotConnectionError("get_tcp_pose() requires an open connection.")
        urpose = self._motion.get_current_pose()
        return self._pose_from_controller(urpose, label="current")

    def get_joint_positions(self) -> JointPositions:
        """Current joint configuration as a typed vector."""
        if not self._conn.is_connected:
            raise RobotConnectionError(
                "get_joint_positions() requires an open connection."
            )
        return JointPositions(self._conn.get_joint_positions())

    def move_to_joints(
        self,
        joints: JointPositions,
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
    ) -> MotionResult:
        """Typed joint move, gate the destination through the preflight, then drive (see Protocol)."""
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
        """Joint-space move through the UR RTDE driver."""
        if not self._conn.is_connected:
            raise RobotConnectionError("move_joint() requires an open connection.")
        if joints.dof != self.capabilities.dof:
            raise RobotMotionRejected(
                f"move_joint() expected {self.capabilities.dof} DoF, got {joints.dof}."
            )
        vel, acc = self._motion._clamp(velocity, acceleration)
        try:
            ok = self._conn.moveJ(joints.tolist(), vel=vel, acc=acc)
        except (RuntimeError, OSError) as exc:
            raise RobotMotionRejected(f"moveJ failed: {exc}") from exc
        if not ok:
            raise RobotMotionRejected("Driver reported moveJ() failure.")

    def move_linear(
        self,
        pose: Pose,
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
    ) -> None:
        """Cartesian move to ``pose`` in :attr:`Frame.BASE`."""
        if pose.frame is not Frame.BASE:
            raise FrameMismatchError(
                f"move_linear() requires pose in Frame.BASE; got {pose.frame!r}."
            )
        if not self._conn.is_connected:
            raise RobotConnectionError("move_linear() requires an open connection.")
        urpose = self._pose_to_controller(pose)
        ok = self._motion.move_to(
            urpose, linear=True, vel=velocity, acc=acceleration, register=False,
        )
        if not ok:
            raise RobotMotionRejected(
                f"Linear move to '{pose.label or '<unlabeled>'}' was rejected."
            )

    @property
    def connection(self) -> URConnection:
        """Underlying RTDE connection."""
        return self._conn

    @property
    def guard(self) -> WorkspaceGuard:
        """Workspace guard instance."""
        return self._guard

    @property
    def motion(self) -> MotionController:
        """Motion controller instance."""
        return self._motion

    @property
    def gripper(self) -> GripperController:
        """Gripper controller instance."""
        return self._gripper

    # ------------------------------------------------------------------
    # Optional capabilities (SupportsDigitalIO / SupportsForceTorque / SupportsRobotStatus).
    # ------------------------------------------------------------------

    # --- SupportsDigitalIO ---
    def set_digital_output(
        self, pin: int, value: bool, *, port: DigitalIOPort = DigitalIOPort.STANDARD
    ) -> None:
        """Drive a controller digital output pin (bank ``port``) high/low."""
        self._conn.set_digital_out(int(pin), bool(value), str(port))

    def get_digital_input(self, pin: int, *, port: DigitalIOPort = DigitalIOPort.STANDARD) -> bool:
        """Read a controller digital input pin (bank ``port``)."""
        return self._conn.get_digital_in(int(pin), str(port))

    def get_digital_output(self, pin: int, *, port: DigitalIOPort = DigitalIOPort.STANDARD) -> bool:
        """Read back a controller digital output pin's commanded level (bank ``port``)."""
        return self._conn.get_digital_out(int(pin), str(port))

    def set_analog_output(self, pin: int, value: float, *, current: bool = False) -> None:
        """Set a controller analog output: voltage (V) unless ``current=True`` (A)."""
        self._conn.set_analog_out(int(pin), float(value), bool(current))

    # --- SupportsForceTorque ---
    def get_tcp_wrench(self) -> Wrench:
        """Live generalized force/torque at the TCP (N, N·m, BASE frame), the hand-over signal."""
        f = self._conn.get_tcp_force()
        return Wrench(float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[4]), float(f[5]),
                     frame=Frame.BASE)

    def get_joint_torques(self) -> tuple[float, ...]:
        """Live torque at each joint (newton-metres), base-to-tool order."""
        return tuple(float(t) for t in self._conn.get_joint_torques())

    # --- SupportsRobotStatus + enriched errors ---
    def get_robot_status(self) -> RobotStatus:
        """A snapshot of the controller's live robot mode + safety mode + stop flags + text."""
        return RobotStatus(
            robot_mode=_UR_ROBOT_MODE.get(self._conn.get_robot_mode(), RobotMode.UNKNOWN),
            safety_mode=_UR_SAFETY_MODE.get(self._conn.get_safety_mode(), SafetyMode.UNKNOWN),
            protective_stopped=self._conn.is_protective_stopped(),
            emergency_stopped=self._conn.is_emergency_stopped(),
            message=self._conn.dashboard_safety_status(),
        )

    def recover_from_protective_stop(self) -> bool:
        """Ask the controller (via the dashboard) to release an active protective stop."""
        ok = self._conn.unlock_protective_stop()
        if ok:
            self.logger.info("Requested protective-stop release from the UR dashboard.")
        else:
            self.logger.warning("Protective-stop release unavailable (no dashboard connection).")
        return ok

    def raise_if_stopped(self) -> None:
        """Raise an ENRICHED :class:`RobotEmergencyStop` if the controller is in a stop/fault state."""
        status = self.get_robot_status()
        if status.is_stopped:
            detail = f" — {status.message}" if status.message else ""
            raise RobotEmergencyStop(
                f"UR controller in {status.safety_mode.value} "
                f"(robot_mode={status.robot_mode.value}){detail}"
            )
