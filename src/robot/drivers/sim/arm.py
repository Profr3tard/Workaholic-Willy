"""Isaac Sim-backed :class:`RobotArm` driver.

Guarantees that have held since the skeleton:
1. Importing the module is safe off-workstation (no Isaac symbols at import time; every
   ``isaacsim.*`` import is lazy, inside ``connect()``'s non-mock branch).
2. ``mock_mode=True`` is a pure-Python kinematic mock, no Isaac is touched, motion
   methods update an internal TCP/joint cache. macOS / CI run entirely on this path.
3. Every motion method honours the Phase N typed :class:`MotionResult` contract; reads
   raise typed errors (``RobotConnectionError`` etc.), never leak ``ImportError``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import numpy as np

from src.geometry import Frame, FrameMismatchError, Pose
from src.geometry.quaternion import (
    IDENTITY_QUAT_XYZW,
    conjugate,
    multiply,
    rotate_vector,
)

from ...core import (
    NO_PLAN_FAIL_SAFE_MESSAGE,
    IsaacNotAvailableError,
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
from ...safety import SafetyPreflight
from ...safety.continuous_monitor import ContinuousCollisionMonitor, ContinuousGuardAbort
from ...safety.planning import CuroboPlanClient, CuroboUnavailableError
from .adapter import (
    willy_joints_to_isaac,
    willy_pose_to_isaac,
    isaac_joints_to_willy,
    isaac_pose_to_willy,
    isaac_rotmat_to_wxyz,
)
from ._isaac_protocols import (
    IsaacArmSubset,
    IsaacArticulation,
    IsaacKinematicsSolver,
    IsaacMotionPolicy,
    IsaacRmpFlow,
)
from .config import SimRobotConfig
from .session import IsaacSimSession

__all__ = ["ISAAC_CAPABILITIES", "IsaacRobotArm"]

_LOGGER = logging.getLogger(__name__)


# The end-effector frame, shared by every UR e-series model.
_EE_FRAME = "tool0"

# Settle detection: max joint speed (rad/s) below which the arm is treated as stopped.
# Validated on-box, the UR5e residual velocity at rest is ~0, so 1e-2 rad/s sits well
# above the noise floor and settles a 0.3 rad joint move in ~24 steps (~0.4 s sim).
_SETTLE_VEL_THRESHOLD = 1e-2

# move() success gate (mm). RMPflow alone converges to ~19 mm (reactive steady-state), so
# move() drives RMPflow toward the target then snaps to the exact IK solution (sub-mm);
# this is the generous EXECUTED tolerance the refined TCP must land within.
_MOVE_POS_TOL_MM = 5.0

# Max joint step (rad) when interpolating a move_joint command. The position drive stalls on a
# single large point-to-point command, so move_joint walks waypoints no larger than this.
_MAX_JOINT_STEP_RAD = 0.03

# move() retries IK-resolve + drive this many times before giving up. A large move (e.g. from a
# parked arm) or a load-perturbed move can stall short once; re-resolving IK from the new config and
# driving again recovers it. A converged move (the common case, incl. all of M1) returns first try.
_MOVE_MAX_ATTEMPTS = 3

# Only retry a move if it landed within this far of the target, a near miss (settle/precision) that
# a re-resolve can close. A far miss means a wrong/unreachable IK branch.
_MOVE_RETRY_MAX_ERR_MM = 40.0

# Hard cap on move_joint interpolation waypoints, so a wild/far IK solution can't explode into
# thousands of physics steps. 250 * _MAX_JOINT_STEP_RAD = 7.5 rad covers any sane single move.
_MAX_INTERP_STEPS = 250

# Orientation matters for grasping: an IK branch can hit the target POSITION with the gripper
# rotated (wrong closing axis -> it shoves the object instead of gripping). So _resolve_ik scores
# position + orientation, and move() verifies both. Tolerance + the mm-per-degree trade-off weight.
_MOVE_ORI_TOL_DEG = 6.0
_ORI_ERR_WEIGHT_MM_PER_DEG = 3.0


def _quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    """Smallest rotation angle (degrees) between two XYZW quaternions."""
    dot = abs(float(np.dot(np.asarray(q1, dtype=np.float64), np.asarray(q2, dtype=np.float64))))
    return float(np.degrees(2.0 * np.arccos(min(1.0, dot))))

# RMPflow approach tolerance (m): stop driving the reactive policy once this close, then let
# the IK refine finish the job. ~30 mm is comfortably inside RMPflow's ~19 mm steady state.
_RMP_APPROACH_TOL_M = 0.03

# cuRobo trajectory execution: per waypoint, step until the arm is within this joint tolerance of it (so the
# arm TRACKS the collision-free path), capped at this many physics steps (so a hard-to-reach waypoint can't
# hang the follow). Tracking-to-tolerance is what makes the follow robust across arbitrary cuRobo plans.
_CUROBO_WP_TOL_RAD = 0.03
_CUROBO_MAX_STEPS_PER_WP = 12

# The 6 UR5e joints, by name.
_ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


# Gripper mount transform (Robotiq 2F-85 on the UR5e flange).
_FLANGE_TO_TCP_QUAT_XYZW = np.array([-0.70710678, 0.0, 0.0, 0.70710678], dtype=np.float64)


def _flange_to_tcp(
    flange_pos_mm: np.ndarray,
    flange_quat_xyzw: np.ndarray,
    offset_mm: "tuple[float, float, float]",
    rotation_quat_xyzw: "tuple[float, float, float, float]",
):
    """Map a flange (tool0) pose to the gripper grasp-centre (TCP) pose."""
    flange_quat = np.asarray(flange_quat_xyzw, dtype=np.float64)
    tcp_quat = multiply(flange_quat, np.asarray(rotation_quat_xyzw, dtype=np.float64))
    tcp_pos = np.asarray(flange_pos_mm, dtype=np.float64) + rotate_vector(
        flange_quat, np.asarray(offset_mm, dtype=np.float64)
    )
    return tcp_pos, tcp_quat


def _tcp_to_flange(
    tcp_pos_mm: np.ndarray,
    tcp_quat_xyzw: np.ndarray,
    offset_mm: "tuple[float, float, float]",
    rotation_quat_xyzw: "tuple[float, float, float, float]",
):
    """Map a gripper grasp-centre (TCP) target to the flange (tool0) target for Lula IK."""
    tcp_quat = np.asarray(tcp_quat_xyzw, dtype=np.float64)
    flange_quat = multiply(tcp_quat, conjugate(np.asarray(rotation_quat_xyzw, dtype=np.float64)))
    flange_pos = np.asarray(tcp_pos_mm, dtype=np.float64) - rotate_vector(
        flange_quat, np.asarray(offset_mm, dtype=np.float64)
    )
    return flange_pos, flange_quat



ISAAC_CAPABILITIES = RobotCapabilities(
    vendor="sim",
    model="isaac-sim",
    dof=6,
    supports_joint_move=True,
    supports_linear_move=True,
    supports_async_move=False,
    has_native_fk=True,
    has_native_ik=True,
    has_force_control=False,
    is_simulated=True,
)


class IsaacRobotArm(RobotArm):
    """Isaac-backed :class:`RobotArm` adapter.

    Parameters
    ----------
    config
        :class:`SimRobotConfig` describing the scene, prim paths, and runtime knobs.
        Construction does **not** touch Isaac.
    safety_preflight
        Optional vendor-neutral :class:`SafetyPreflight`. When supplied,
        ``move()`` routes through it before commanding motion.
    """

    def __init__(
        self,
        config: SimRobotConfig,
        *,
        safety_preflight: "SafetyPreflight | None" = None,
    ) -> None:
        self._config = config
        self._session = IsaacSimSession(config)
        self._connected = False
        self._preflight: "SafetyPreflight | None" = safety_preflight

        # Real (non-mock) Isaac handles, populated in connect()
        self._articulation: IsaacArticulation | None = None
        self._arm_subset: IsaacArmSubset | None = None  # ArticulationSubset over the 6 _ARM_JOINT_NAMES
        self._kin_solver: IsaacKinematicsSolver | None = None
        self._rmpflow: IsaacRmpFlow | None = None
        self._amp: IsaacMotionPolicy | None = None
        self._dof_names: list[str] | None = None
        self._natural_aim_seed = False
        self._continuous_monitor: ContinuousCollisionMonitor | None = None
        # Motion-planning seam (CuRobo will plug in here). "ik" (default) == the blind pure-IK straight
        # line, byte-identical. "rmpflow" drives the reactive collision-aware policy toward the pose before
        # the IK snap, avoiding obstacles registered via register_planner_obstacles. Owner-overridable
        # (like _natural_aim_seed / _continuous_monitor) so a runner can flip it before the config plumbs it.
        self._motion_planner: str = config.motion_planner
        # Lazily-spawned cuRobo planning client (process-isolated server; created only when motion_planner ==
        # "curobo" and a move() is issued). None == off == byte-identical; closed in disconnect().
        self._curobo_client: CuroboPlanClient | None = None
        # Opt-in: plan move_joint() through cuRobo instead of interpolating a straight line in joint
        # space. Off by default because it changes the trajectory of EVERY move_joint call.
        self._plan_joint_moves: bool = False
        # CuRobo-everywhere AUTO-FALLBACK: when motion_planner == "curobo" but the cuRobo sidecar/env is not
        # available (no env, missing python, server boot fails), the FIRST move() probes it once, logs a warning,
        # latches this flag, and DEGRADES to the blind-"ik" path.
        self._curobo_unavailable: bool = False
        #: Why it became unavailable, kept so a runner can report the CAUSE and not just the symptom.
        self._curobo_unavailable_reason: str = ""

        # Mock-mode kinematic cache (also the placeholder state before connect). In
        # mock_mode the motion methods update these in place and the reads return them.
        self._tcp: Pose = Pose(
            position_mm=np.array([400.0, 0.0, 300.0], dtype=np.float64),
            quaternion_xyzw=IDENTITY_QUAT_XYZW.copy(),
            frame=Frame.BASE,
            label="isaac-home-placeholder",
        )
        self._home_tcp: Pose = self._tcp
        home = config.home_joint_positions
        self._joints = JointPositions(
            np.asarray(
                home if home is not None else (0.0,) * ISAAC_CAPABILITIES.dof,
                dtype=np.float64,
            )
        )
        self._home_joints = self._joints

    @property
    def mock_mode(self) -> bool:
        """Whether this arm runs the pure-Python kinematic mock (immutable per session)."""
        return self._config.mock_mode

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def capabilities(self) -> RobotCapabilities:
        """Honest capability surface."""
        if self._config.mock_mode:
            return replace(ISAAC_CAPABILITIES, has_native_fk=False, has_native_ik=False)
        return ISAAC_CAPABILITIES

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def session(self) -> IsaacSimSession:
        """Expose the owning :class:`IsaacSimSession` for tests / tooling."""
        return self._session

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Bring the Isaac session up and (non-mock) wrap the articulation + kinematics.

        Raises
        ------
        IsaacNotAvailableError
            On hosts without the Isaac SDK (non-mock path).
        RobotConnectionError
            If ``robot_prim_path`` is not configured for the non-mock path.
        """
        self._session.start()
        if not self._config.mock_mode:
            self._bring_up_articulation()
        self._connected = True

    def _bring_up_articulation(self) -> None:
        """Wrap the UR articulation and build the Lula kinematics solver (non-mock)."""
        if not self._config.robot_prim_path:
            raise RobotConnectionError(
                "IsaacRobotArm.connect (non-mock) requires SimRobotConfig.robot_prim_path."
            )
        from isaacsim.core.prims import SingleArticulation  # type: ignore[import-not-found]
        from isaacsim.robot_motion.motion_generation import (  # type: ignore[import-not-found]
            interface_config_loader,
        )
        from isaacsim.robot_motion.motion_generation.lula.kinematics import (  # type: ignore[import-not-found]
            LulaKinematicsSolver,
        )

        articulation = SingleArticulation(prim_path=self._config.robot_prim_path)
        articulation.initialize()

        # The robot is a COMBINED 12-DoF articulation (UR5e + the 2F-85 gripper variant). Address
        # the 6 arm joints by name via an ArticulationSubset so every arm read/write ignores the
        # gripper DoFs (and is robust to their dof ordering).
        from isaacsim.core.api.articulations import ArticulationSubset  # type: ignore[import-not-found]

        dof_names = list(articulation.dof_names)
        missing = [n for n in _ARM_JOINT_NAMES if n not in dof_names]
        if missing:
            raise IsaacNotAvailableError(
                f"Arm joints {missing} not in the articulation dof_names {dof_names}."
            )
        arm_subset = ArticulationSubset(articulation, list(_ARM_JOINT_NAMES))

        if self._config.home_joint_positions is not None:
            home = np.asarray(self._config.home_joint_positions, dtype=np.float64)
            arm_subset.set_joint_positions(home)         # teleport the state
            arm_subset.apply_action(joint_positions=home)  # AND hold it as the drive target, so the
            self._session.step_n(2)                       # arm does not drift during later stepping

        # Lula solver + RMPflow policy for the configured UR model (de-locked: robot_model selects the Lula
        # supported-config key -- "UR5e" (default) / "UR3e" / ...). All UR e-series share the link/joint names,
        # so only the Lula config (link LENGTHS) differs; the rest of the driver is model-agnostic.
        from src.robot.drivers.sim.robot_models import ur_model_spec

        lula_key = ur_model_spec(self._config.robot_model).lula_key
        lula_cfg = interface_config_loader.load_supported_lula_kinematics_solver_config(lula_key)
        solver = LulaKinematicsSolver(**lula_cfg)
        frames = list(solver.get_all_frame_names())
        if _EE_FRAME not in frames:
            raise IsaacNotAvailableError(
                f"End-effector frame {_EE_FRAME!r} not found in the {lula_key} Lula description "
                f"(frames: {frames})."
            )

        # RMPflow motion policy for the Cartesian move() path. The config already
        # carries end_effector_frame_name='tool0'; ArticulationMotionPolicy binds it to this
        # articulation at the fixed step_dt_s.
        from isaacsim.robot_motion.motion_generation import (  # type: ignore[import-not-found]
            ArticulationMotionPolicy,
        )
        from isaacsim.robot_motion.motion_generation.lula.motion_policies import (  # type: ignore[import-not-found]
            RmpFlow,
        )

        rmp_cfg = interface_config_loader.load_supported_motion_policy_config(lula_key, "RMPflow")
        rmpflow = RmpFlow(**rmp_cfg)
        amp = ArticulationMotionPolicy(
            articulation, rmpflow, default_physics_dt=self._config.step_dt_s
        )

        self._articulation = articulation
        self._arm_subset = arm_subset
        self._kin_solver = solver
        self._rmpflow = rmpflow
        self._amp = amp
        self._dof_names = dof_names

    def disconnect(self) -> None:
        if self._curobo_client is not None:
            self._curobo_client.close()  # shut down the process-isolated cuRobo server
            self._curobo_client = None
        self._articulation = None
        self._arm_subset = None
        self._kin_solver = None
        self._rmpflow = None
        self._amp = None
        self._dof_names = None
        self._session.stop()
        self._connected = False

    # ------------------------------------------------------------------
    # State (live reads in non-mock; cached in mock)
    # ------------------------------------------------------------------

    def get_tcp_pose(self) -> Pose:
        if not self._connected:
            raise RobotConnectionError("IsaacRobotArm is not connected.")
        if self._config.mock_mode or self._arm_subset is None or self._kin_solver is None:
            return self._tcp
        joints = np.asarray(self._arm_subset.get_joint_positions(), dtype=np.float64)
        pos_m, rot = self._kin_solver.compute_forward_kinematics(_EE_FRAME, joints)
        tool0 = isaac_pose_to_willy(np.asarray(pos_m), isaac_rotmat_to_wxyz(rot), label="isaac-tcp")
        tcp_pos, tcp_quat = _flange_to_tcp(
            tool0.position_mm, tool0.quaternion_xyzw,
            self._config.tool_offset_mm, self._config.tool_rotation_quat_xyzw,
        )
        return Pose(position_mm=tcp_pos, quaternion_xyzw=tcp_quat, frame=Frame.BASE, label="isaac-tcp")

    def get_joint_positions(self) -> JointPositions:
        if not self._connected:
            raise RobotConnectionError("IsaacRobotArm is not connected.")
        if self._config.mock_mode or self._arm_subset is None:
            return self._joints
        return isaac_joints_to_willy(self._arm_subset.get_joint_positions())

    # ------------------------------------------------------------------
    # Kinematics
    # ------------------------------------------------------------------

    def _check_dof(self, joints: JointPositions, what: str) -> None:
        n = int(np.asarray(joints.values).reshape(-1).shape[0])
        if n != ISAAC_CAPABILITIES.dof:
            raise RobotMotionRejected(
                f"IsaacRobotArm.{what} expects {ISAAC_CAPABILITIES.dof} joints; got {n}."
            )

    def fk(self, joints: JointPositions) -> Pose:
        if self._config.mock_mode:
            raise IsaacNotAvailableError(
                "IsaacRobotArm.fk requires a non-mock Isaac session; mock_mode has no kinematics."
            )
        if not self._connected or self._kin_solver is None:
            raise RobotConnectionError("IsaacRobotArm.fk requires a connected articulation.")
        self._check_dof(joints, "fk")
        pos_m, rot = self._kin_solver.compute_forward_kinematics(
            _EE_FRAME, willy_joints_to_isaac(joints)
        )
        tool0 = isaac_pose_to_willy(np.asarray(pos_m), isaac_rotmat_to_wxyz(rot), label="isaac-fk")
        tcp_pos, tcp_quat = _flange_to_tcp(
            tool0.position_mm, tool0.quaternion_xyzw,
            self._config.tool_offset_mm, self._config.tool_rotation_quat_xyzw,
        )
        return Pose(position_mm=tcp_pos, quaternion_xyzw=tcp_quat, frame=Frame.BASE, label="isaac-fk")

    def ik(
        self,
        pose: Pose,
        *,
        seed: JointPositions | None = None,
    ) -> JointPositions:
        # Validate the frame BEFORE any conversion (contract parity with other drivers).
        if pose.frame is not Frame.BASE:
            raise FrameMismatchError(
                f"IsaacRobotArm.ik requires Frame.BASE; got {pose.frame!r}."
            )
        if self._config.mock_mode:
            raise IsaacNotAvailableError(
                "IsaacRobotArm.ik requires a non-mock Isaac session; mock_mode has no kinematics."
            )
        if not self._connected or self._kin_solver is None:
            raise RobotConnectionError("IsaacRobotArm.ik requires a connected articulation.")

        # The pose is the TCP (grasp-centre) target; map it through the gripper mount to the
        # flange (tool0) target Lula solves for.
        flange_pos, flange_quat = _tcp_to_flange(
            np.asarray(pose.position_mm, dtype=np.float64),
            np.asarray(pose.quaternion_xyzw, dtype=np.float64),
            self._config.tool_offset_mm, self._config.tool_rotation_quat_xyzw,
        )
        tool0_target = Pose(
            position_mm=flange_pos, quaternion_xyzw=flange_quat, frame=Frame.BASE, label=pose.label
        )
        target_pos_m, target_wxyz = willy_pose_to_isaac(tool0_target)
        kwargs: dict[str, object] = {}
        if seed is not None:
            self._check_dof(seed, "ik seed")
            kwargs["warm_start"] = willy_joints_to_isaac(seed)

        joints, success = self._kin_solver.compute_inverse_kinematics(
            _EE_FRAME, target_pos_m, target_wxyz, **kwargs
        )
        if not success or joints is None:
            raise RobotKinematicsError(
                f"IsaacRobotArm.ik did not converge for pose {pose.label!r}."
            )
        joints = np.asarray(joints, dtype=np.float64).reshape(-1)
        if joints.shape[0] != ISAAC_CAPABILITIES.dof:
            raise RobotKinematicsError(
                f"IK returned {joints.shape[0]} joints; expected {ISAAC_CAPABILITIES.dof}."
            )
        return isaac_joints_to_willy(joints)

    # ------------------------------------------------------------------
    # Motion 
    # ------------------------------------------------------------------

    def reset_safety_continuity(self) -> None:
        """Reset the SafetyPreflight's motion-continuity reference."""
        if self._preflight is not None:
            self._preflight.reset()

    def move_to_joints(
        self,
        joints: JointPositions,
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
    ) -> MotionResult:
        """Move the arm to the specified joint positions."""
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
        if self.mock_mode:
            # Identity mock: commit the commanded joints. TCP is not touched (the mock has
            # no FK); callers needing a consistent TCP use move_linear/move.
            self._joints = joints
            return
        if not self._connected or self._arm_subset is None:
            raise RobotConnectionError(
                "IsaacRobotArm.move_joint requires a connected articulation."
            )
        self._check_dof(joints, "move_joint")
        # Drive only the 6 arm joints, the subset maps them to their indices in the combined
        # articulation, leaving the gripper DoFs untouched.
        target = np.asarray(willy_joints_to_isaac(joints), dtype=np.float64)
        if self._arm_subset is None:
            return
        start = np.asarray(self._arm_subset.get_joint_positions(), dtype=np.float64)
        mon = self._continuous_monitor
        waypoints: list[np.ndarray] | None = None
        if self._plan_joint_moves and self._motion_planner == "curobo":
            planned = self._plan_joint_path(start, target)
            if planned is None:
                # Fail closed. Silently interpolating here would hand back exactly the blind path this
                # option exists to replace, at the moment it is known to be unsafe.
                raise CuroboUnavailableError(
                    f"cuRobo found no collision-free joint path to {np.round(target, 3).tolist()}. "
                    f"Refusing to interpolate blindly -- that is the path the guard already rejects."
                )
            waypoints = [np.asarray(q, dtype=np.float64) for q in planned]

        if waypoints is None:
            n_steps = max(1, int(np.ceil(float(np.max(np.abs(target - start))) / _MAX_JOINT_STEP_RAD)))
            n_steps = min(n_steps, _MAX_INTERP_STEPS)
            waypoints = [start + (target - start) * (i / n_steps) for i in range(1, n_steps + 1)]

        for waypoint in waypoints:
            if mon is not None:  # check the NEXT config BEFORE moving into it; HALT (never apply) on STOP
                verdict = mon.check(waypoint)  # waypoint is already UR-order joints (subset == UR order)
                if verdict.stop:
                    raise ContinuousGuardAbort(verdict)
            self._arm_subset.apply_action(joint_positions=waypoint)
            self._session.step()
        if mon is not None:
            verdict = mon.check(target)
            if verdict.stop:
                raise ContinuousGuardAbort(verdict)
        self._arm_subset.apply_action(joint_positions=target)
        self.wait_until_steady(timeout_s=self._config.settle_timeout_s)

    def move_linear(
        self,
        pose: Pose,
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
    ) -> None:
        if self.mock_mode:
            if pose.frame is not Frame.BASE:
                raise FrameMismatchError(
                    "IsaacRobotArm.move_linear requires Frame.BASE; "
                    f"got {pose.frame!r}."
                )
            self._tcp = pose
            return
        result = self.move(pose, linear=True, vel=velocity, acc=acceleration)
        if result.status is not MotionStatus.EXECUTED:
            raise RobotMotionRejected(
                f"IsaacRobotArm.move_linear failed: {result.status.value} — {result.message}"
            )

    def stop(self) -> None:
        """Best-effort halt: hold the arm at its current configuration."""
        if self.mock_mode or not self._connected or self._arm_subset is None:
            return None
        try:
            current = np.asarray(self._arm_subset.get_joint_positions(), dtype=np.float64)
            self._arm_subset.apply_action(joint_positions=current)  # hold station
        except Exception:  # noqa: BLE001 - stop() must never raise (RobotArm Protocol contract)
            pass
        if self._preflight is not None:
            self._preflight.reset()
        return None

    # ------------------------------------------------------------------
    # High-level pipeline surface
    # ------------------------------------------------------------------

    def is_inside_workspace(self, pose: Pose) -> bool:
        """Always ``True`` the sim driver owns no workspace box."""
        if pose.frame is not Frame.BASE:
            raise FrameMismatchError(
                "IsaacRobotArm.is_inside_workspace requires Frame.BASE; "
                f"got {pose.frame!r}."
            )
        return True

    def move_to(
        self,
        pose: Pose,
        *,
        linear: bool = False,
        vel: float | None = None,
        acc: float | None = None,
        register: bool = True,
    ) -> bool:
        if self.mock_mode:
            if not self._connected or pose.frame is not Frame.BASE:
                return False
            self._tcp = pose
            return True
        return (
            self.move(pose, linear=linear, vel=vel, acc=acc, register=register).status
            is MotionStatus.EXECUTED
        )

    def move_home(self) -> bool:
        if self.mock_mode:
            if not self._connected:
                return False
            self._tcp = self._home_tcp
            self._joints = self._home_joints
            return True
        if not self._connected or self._arm_subset is None or self._home_joints is None:
            return False
        if self._preflight is not None:
            self._preflight.reset()
        self.move_joint(self._home_joints)
        return True

    def wait_until_steady(
        self,
        timeout_s: float = 5.0,
        poll_interval_s: float = 0.02,
    ) -> bool:
        """Step the sim until the arm stops, or ``timeout_s`` (sim seconds) elapses.

        Returns ``True`` once ``max|joint velocity| < _SETTLE_VEL_THRESHOLD`` (rad/s),
        ``False`` on timeout. In mock mode (or before connect) nothing is in flight, so it
        returns ``True`` immediately. ``poll_interval_s`` is accepted for interface parity;
        in sim we evaluate every physics step (the fixed ``step_dt_s`` cadence).
        """
        if self._config.mock_mode or self._arm_subset is None:
            return True
        dt = self._config.step_dt_s if self._config.step_dt_s > 0 else 1.0 / 60.0
        max_steps = max(1, int(timeout_s / dt))
        for _ in range(max_steps):
            self._session.step()
            vel = np.asarray(self._arm_subset.get_joint_velocities(), dtype=np.float64)
            if float(np.max(np.abs(vel))) < _SETTLE_VEL_THRESHOLD:
                return True
        return False

    def _resolve_ik(self, pose: Pose) -> JointPositions:
        """Robustly resolve IK for a TCP target across multiple seeds."""
        current = np.asarray(self.get_joint_positions().values, dtype=np.float64)
        pi = float(np.pi)
        bases = [current]
        if self._natural_aim_seed:
            tx, ty = float(pose.position_mm[0]), float(pose.position_mm[1])
            bases.insert(0, np.array([np.arctan2(ty, tx), -1.6, 1.6, -1.5, -1.57, 0.0], dtype=np.float64))
        if self._config.home_joint_positions is not None:
            bases.append(np.asarray(self._config.home_joint_positions, dtype=np.float64))
        seeds: list[np.ndarray] = []
        for base in bases:
            # Sweep wrist_3 (the free rotation about the approach axis for a top-down grasp): the
            # reachable IK branch can sit at any wrist_3, and Lula only returns the branch nearest
            # the seed, so cover the full circle to be sure we find a reachable, drive-able one.
            for k in range(12):
                s = base.copy()
                s[5] = base[5] + k * (2.0 * pi / 12.0)
                seeds.append(s)
            seeds.append(base + np.array([0.0, 0.0, 0.0, pi, 0.0, 0.0]))  # wrist_1 flip
            seeds.append(base + np.array([0.0, 0.0, 0.0, 0.0, pi, 0.0]))  # wrist_2 flip

        target_pos = np.asarray(pose.position_mm, dtype=np.float64)
        target_quat = np.asarray(pose.quaternion_xyzw, dtype=np.float64)
        best: JointPositions | None = None
        best_err = float("inf")
        for seed in seeds:
            try:
                joints = self.ik(pose, seed=JointPositions(seed))
            except RobotKinematicsError:
                continue
            fk_pose = self.fk(joints)
            pos_err = float(np.linalg.norm(np.asarray(fk_pose.position_mm) - target_pos))
            ori_err = _quat_angle_deg(fk_pose.quaternion_xyzw, target_quat)
            # Combined cost: a branch that hits the position but with the gripper rotated must lose
            # to one that matches both (a wrong closing axis shoves the object instead of gripping).
            combined = pos_err + _ORI_ERR_WEIGHT_MM_PER_DEG * ori_err
            if combined < best_err:
                best_err, best = combined, joints
                if pos_err < _MOVE_POS_TOL_MM and ori_err < _MOVE_ORI_TOL_DEG:
                    break
        if best is None:
            raise RobotKinematicsError(
                f"IsaacRobotArm: no IK seed converged for pose {pose.label!r}."
            )
        # Unwind each revolute joint by +/-2*pi to the branch nearest the current configuration.
        best_arr = np.asarray(best.values, dtype=np.float64)
        two_pi = 2.0 * float(np.pi)
        unwound = best_arr - two_pi * np.round((best_arr - current) / two_pi)
        out_of_range = np.abs(unwound) > two_pi
        unwound[out_of_range] = best_arr[out_of_range]
        return JointPositions(unwound)

    def _drive_rmpflow(self, pose: Pose) -> bool:
        """Drive RMPflow toward a BASE-frame pose (smooth, collision-aware approach).

        Returns ``True`` once the EE is within ``_RMP_APPROACH_TOL_M`` of the target (RMPflow routed
        there, ``move()`` then refines to the exact IK solution; RMPflow's ~19 mm steady-state offset is
        removed by that refine). Returns ``False`` if the ``settle_timeout_s`` step budget elapses without
        converging, the reactive policy is BLOCKED or trapped in a local minimum.
        """
        # Bind the Isaac handles to locals so the non-None narrowing holds across the loop's
        # method calls.
        rmpflow, amp = self._rmpflow, self._amp
        articulation, arm_subset, kin_solver = self._articulation, self._arm_subset, self._kin_solver
        if (
            rmpflow is None or amp is None or articulation is None
            or arm_subset is None or kin_solver is None
        ):
            raise RobotConnectionError(
                "IsaacRobotArm._drive_rmpflow requires a connected articulation."
            )
        # RMPflow's EE is tool0 (the Lula frame); map the TCP target through the gripper mount to
        # the tool0 target so the convergence check below is consistent.
        flange_pos, flange_quat = _tcp_to_flange(
            np.asarray(pose.position_mm, dtype=np.float64),
            np.asarray(pose.quaternion_xyzw, dtype=np.float64),
            self._config.tool_offset_mm, self._config.tool_rotation_quat_xyzw,
        )
        tool0_pose = Pose(
            position_mm=flange_pos, quaternion_xyzw=flange_quat, frame=Frame.BASE, label=pose.label
        )
        target_pos_m, target_wxyz = willy_pose_to_isaac(tool0_pose)
        rmpflow.set_end_effector_target(
            target_position=target_pos_m, target_orientation=target_wxyz
        )
        dt = self._config.step_dt_s if self._config.step_dt_s > 0 else 1.0 / 60.0
        budget = max(1, int(self._config.settle_timeout_s / dt))
        for _ in range(budget):
            rmpflow.update_world()
            action = amp.get_next_articulation_action(dt)
            articulation.apply_action(action)
            self._session.step()
            q = np.asarray(arm_subset.get_joint_positions(), dtype=np.float64)
            pos_m, _ = kin_solver.compute_forward_kinematics(_EE_FRAME, q)
            if float(np.linalg.norm(np.asarray(pos_m) - target_pos_m)) < _RMP_APPROACH_TOL_M:
                return True
        return False

    def register_planner_obstacles(self, obstacles: "Iterable[Any]") -> int:
        """Register scene obstacles into the motion planner's world model (the shared scene→planner seam).

        ``obstacles`` are Isaac core-API objects (e.g. ``FixedCuboid`` / ``DynamicCuboid`` /
        ``DynamicCylinder`` wrapping the bin walls + neighbour-object prims); the RMPflow approach then
        routes AROUND them. The runner builds the objects (this driver never imports Isaac), so they are
        duck-typed ``Any``. Returns the count registered. No-op (returns 0) when no planner is up.
        """
        rmpflow = self._rmpflow
        if rmpflow is None:
            return 0
        added = 0
        for obstacle in obstacles:
            rmpflow.add_obstacle(obstacle)
            added += 1
        return added

    def _guard_self_collision_margin_mm(self) -> float:
        """The clearance to ask the PLANNER for, from this arm's live guard (0.0 == leave it alone).

        Read off the guard rather than re-read from config so a runner that swapped the guard in code is
        still described truthfully. NOTE this is ``planner_margin_mm``, NOT the ``min_distance_mm`` the
        guard enforces: how much margin a planner can absorb depends on how tightly its spheres fit that
        robot. Deriving it from the enforced distance is exactly the mistake that took the UR3e cell from
        10/10 to 0/10, it plans to 6 mm and finds NO plan at 10 mm.
        """
        preflight = self._preflight
        if preflight is None:
            return 0.0
        for guard in getattr(preflight, "guards", ()) or ():
            margin = getattr(guard, "_planner_margin_mm", None)
            if isinstance(margin, (int, float)) and margin > 0.0:
                return float(margin)
        return 0.0

    def _plan_joint_path(
        self, start: "np.ndarray", target: "np.ndarray"
    ) -> list[list[float]] | None:
        """A collision-free joint path ``start`` -> ``target``, or ``None`` if cuRobo cannot find one.

        Returns ``None`` rather than raising for an UNAVAILABLE planner.
        """
        try:
            client = self._get_curobo_client()
        except Exception as exc:  # noqa: BLE001 - no planner is "no plan", reported by the caller
            _LOGGER.warning("planned joint move requested but cuRobo is unavailable: %s", exc)
            return None
        try:
            return client.plan_joint([float(q) for q in start], [float(q) for q in target])
        except CuroboUnavailableError as exc:
            _LOGGER.warning("cuRobo joint planning failed: %s", exc)
            return None

    def _get_curobo_client(self) -> CuroboPlanClient:
        """Lazily spawn + JIT-warm the process-isolated cuRobo planning server (blocks on the first call)."""
        if self._curobo_client is None:
            import os

            from src.robot.drivers.sim.robot_models import curobo_robot_yml
            from src.robot.safety.planning.environment import ENV_CUROBO_ROBOT

            # cuRobo robot config = the CONFIGURED model's {key}.yml.
            robot_yml = curobo_robot_yml(self._config.robot_model)
            env_yml = os.environ.get(ENV_CUROBO_ROBOT)
            if env_yml and env_yml != robot_yml:
                _LOGGER.warning(
                    "%s=%r disagrees with the configured robot_model=%r (-> %r) and is IGNORED: planning a "
                    "cell against another robot's geometry is never safe. Set robot_model instead.",
                    ENV_CUROBO_ROBOT, env_yml, self._config.robot_model, robot_yml,
                )
            # Hand the planner the clearance the SAFETY GUARD will demand of the configuration it returns.
            client = CuroboPlanClient(
                robot_config=robot_yml, self_collision_margin_mm=self._guard_self_collision_margin_mm(),
            )
            client.start()
            self._curobo_client = client
        return self._curobo_client

    def _resolve_motion_planner(self) -> str:
        """The EFFECTIVE planner for this move, with the cuRobo-everywhere auto-fallback.

        Returns ``self._motion_planner`` verbatim unless it is ``"curobo"`` and the cuRobo sidecar/env cannot be
        brought up, then it latches ``_curobo_unavailable`` and returns ``"ik"`` (a one-time probe + warning).
        """
        if self._motion_planner != "curobo" or self.mock_mode:
            return self._motion_planner
        if self._curobo_unavailable:
            return "ik"
        try:
            self._get_curobo_client()  # idempotent: spawns + JIT-warms the server once, else raises
        except CuroboUnavailableError as exc:
            self._curobo_unavailable = True
            # Kept for the caller to READ, not just for a log line to scroll past.
            self._curobo_unavailable_reason = str(exc)
            _LOGGER.warning(
                "cuRobo planner unavailable (%s); falling back to the blind IK path for this arm.", exc,
            )
            return "ik"
        return "curobo"

    @property
    def curobo_degraded(self) -> bool:
        """``True`` when this arm was CONFIGURED for cuRobo and is running blind IK instead."""
        return self._motion_planner == "curobo" and self._curobo_unavailable

    @property
    def curobo_degraded_reason(self) -> str:
        """Why the planner could not start. Empty while nothing has degraded."""
        return self._curobo_unavailable_reason

    def set_curobo_world(self, cuboids: list[dict]) -> int:
        """Register the scene obstacles into cuRobo's collision world (the scene->planner world-model for the
        ``"curobo"`` planner, the cross-process analogue of register_planner_obstacles for RMPflow).
        """
        try:
            return self._get_curobo_client().set_world(cuboids)
        except CuroboUnavailableError:
            return 0

    def _drive_curobo(
        self, pose: "Pose", *, current_joints: "JointPositions | None" = None
    ) -> MotionResult:
        """Plan a collision-free trajectory to ``pose`` with cuRobo and EXECUTE it (planner-owns-final-motion).

        Maps the grasp TCP pose -> tool0 (the cuRobo/Lula EE) + mm/XYZW -> m/WXYZ, sends {current joints, goal}
        to the warm server, and follows the returned joint trajectory. ``EXECUTED`` if the final TCP is within
        tolerance; a fail-safe ``TIMEOUT`` if cuRobo found no collision-free plan; ``CONTROLLER_REJECTED`` if
        the planning server is unavailable. No blind motion on any failure (the planner owns the path).
        """
        if self._arm_subset is None:
            return MotionResult.failed(
                MotionStatus.CONNECTION_ERROR, MotionCommand.MOVE_TO, target_pose=pose,
                message="IsaacRobotArm._drive_curobo requires a connected articulation.",
            )
        # grasp TCP -> tool0 (Lula EE), then mm -> m and XYZW -> WXYZ (the cuRobo goal convention).
        flange_pos_mm, flange_quat_xyzw = _tcp_to_flange(
            np.asarray(pose.position_mm, dtype=np.float64),
            np.asarray(pose.quaternion_xyzw, dtype=np.float64),
            self._config.tool_offset_mm, self._config.tool_rotation_quat_xyzw,
        )
        goal_pos_m = (np.asarray(flange_pos_mm, dtype=np.float64) / 1000.0).tolist()
        fqx, fqy, fqz, fqw = (float(v) for v in flange_quat_xyzw)
        goal_quat_wxyz = [fqw, fqx, fqy, fqz]
        arm_q = np.asarray(self._arm_subset.get_joint_positions(), dtype=np.float64)  # _ARM_JOINT_NAMES order
        try:
            client = self._get_curobo_client()
            start = [float(arm_q[_ARM_JOINT_NAMES.index(n)]) for n in client.joint_names]  # -> server order
            traj = client.plan(start, goal_pos_m, goal_quat_wxyz)
        except CuroboUnavailableError as exc:
            return MotionResult.failed(
                MotionStatus.CONTROLLER_REJECTED, MotionCommand.MOVE_TO, target_pose=pose,
                message=f"cuRobo planner unavailable: {exc}",
            )
        if traj is None:
            # Fail-safe, but opaque: downstream this is a bare TIMEOUT, indistinguishable from a bad grasp.
            _LOGGER.warning(
                "cuRobo found NO plan: TCP target=%s -> flange goal=%s quat_wxyz=%s from start joints=%s "
                "(server order %s); failing safe, no blind motion.",
                np.round(np.asarray(pose.position_mm), 1).tolist(),
                np.round(np.asarray(goal_pos_m) * 1000.0, 1).tolist(),
                [round(v, 4) for v in goal_quat_wxyz],
                np.round(arm_q, 3).tolist(), [round(v, 3) for v in start],
            )
            return MotionResult.failed(
                MotionStatus.TIMEOUT, MotionCommand.MOVE_TO, target_pose=pose,
                message=NO_PLAN_FAIL_SAFE_MESSAGE,
            )
        if self._preflight is not None:
            order = [client.joint_names.index(n) for n in _ARM_JOINT_NAMES]  # server order -> _ARM_JOINT_NAMES
            final_q = JointPositions(np.asarray([traj[-1][i] for i in order], dtype=np.float64))
            rejected = self._preflight_reject(
                pose, target_joints=final_q, current_joints=current_joints,
                skip_guards=frozenset({"ik_quality", "motion_continuity"}),
            )
            if rejected is not None:
                return rejected
        self._execute_curobo_trajectory(traj, client.joint_names, client.dt)
        tcp = self.get_tcp_pose()
        err_mm = float(np.linalg.norm(np.asarray(tcp.position_mm) - np.asarray(pose.position_mm)))
        ori_err_deg = _quat_angle_deg(tcp.quaternion_xyzw, pose.quaternion_xyzw)
        if err_mm <= _MOVE_POS_TOL_MM and ori_err_deg <= _MOVE_ORI_TOL_DEG:
            return MotionResult.executed(
                MotionCommand.MOVE_TO, target_pose=pose,
                message=f"cuRobo trajectory reached target ({err_mm:.1f} mm, {ori_err_deg:.1f}°).",
            )
        _LOGGER.warning(
            "cuRobo trajectory ended %.1f mm / %.1f deg from target=%s (tol %.1f mm, %.1f deg) -- the arm "
            "followed the plan but the plan did not land on the goal.",
            err_mm, ori_err_deg, np.round(np.asarray(pose.position_mm), 1).tolist(),
            _MOVE_POS_TOL_MM, _MOVE_ORI_TOL_DEG,
        )
        return MotionResult.failed(
            MotionStatus.TIMEOUT, MotionCommand.MOVE_TO, target_pose=pose,
            message=f"cuRobo trajectory ended {err_mm:.1f} mm from target (> {_MOVE_POS_TOL_MM} mm).",
        )

    def _execute_curobo_trajectory(
        self, traj: list[list[float]], server_joint_names: list[str], traj_dt: float
    ) -> None:
        """Follow cuRobo's joint trajectory faithfully (no per-waypoint IK / no blind interpolation).

        cuRobo guaranteed the PATH is collision-free, so the arm must TRACK it — not rush through. For each
        waypoint we command it as the drive target and step until the arm actually reaches it (within
        ``_CUROBO_WP_TOL_RAD``) or a per-waypoint step cap, so the arm physically follows the planned path at
        every config (a fixed steps-per-waypoint rush lets the drive lag, and on a contorted plan the arm
        ends far from the goal).
        """
        if self._arm_subset is None:
            return
        idx = [server_joint_names.index(n) for n in _ARM_JOINT_NAMES]  # server order -> _ARM_JOINT_NAMES order
        last_q: np.ndarray | None = None
        for wp in traj:
            last_q = np.asarray([wp[i] for i in idx], dtype=np.float64)
            self._arm_subset.apply_action(joint_positions=last_q)
            for _ in range(_CUROBO_MAX_STEPS_PER_WP):
                self._session.step()
                cur = np.asarray(self._arm_subset.get_joint_positions(), dtype=np.float64)
                if float(np.max(np.abs(cur - last_q))) < _CUROBO_WP_TOL_RAD:
                    break  # reached this waypoint -> advance (keeps the arm ON the planned path)
        if last_q is not None:  # re-assert the final config, then settle so the TCP lands exactly
            self._arm_subset.apply_action(joint_positions=last_q)
        self.wait_until_steady(self._config.settle_timeout_s)

    def move(
        self,
        pose: Pose,
        *,
        linear: bool = False,
        vel: float | None = None,
        acc: float | None = None,
        register: bool = True,
    ) -> MotionResult:
        """Typed move non-mock Cartesian motion via RMPflow + IK refine."""
        if pose.frame is not Frame.BASE:
            return MotionResult.failed(
                MotionStatus.INVALID_TARGET,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message=(
                    "IsaacRobotArm.move requires Frame.BASE; got "
                    f"{pose.frame!r}."
                ),
            )
        if not self._connected:
            return MotionResult.failed(
                MotionStatus.CONNECTION_ERROR,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message="IsaacRobotArm is not connected.",
            )
        # mock_mode: identity kinematics. Still run the preflight (a pose-/stub-based guard can reject)
        if self.mock_mode:
            rejected = self._preflight_reject(pose, target_joints=None, current_joints=None)
            if rejected is not None:
                return rejected
            self._tcp = pose
            return MotionResult.executed(
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message="mock_mode",
            )
        if self._articulation is None or self._kin_solver is None or self._rmpflow is None:
            return MotionResult.failed(
                MotionStatus.CONNECTION_ERROR,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message="IsaacRobotArm.move: articulation/motion policy not initialized (call connect()).",
            )
        try:
            target_joints = self._resolve_ik(pose)
        except RobotKinematicsError as exc:
            return MotionResult.failed(
                MotionStatus.IK_FAILED,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message=f"IsaacRobotArm.move: IK failed: {exc}",
            )
        try:
            current_joints = self.get_joint_positions()
        except Exception:  # noqa: BLE001 - telemetry hiccup: the IK-jump check skips on None
            current_joints = None
        # CuRobo-everywhere: resolve the EFFECTIVE planner once (curobo -> ik auto-fallback if the sidecar/env is
        # unavailable), then route on it, so a "curobo" default is safe wherever the cuRobo env is absent.
        planner = self._resolve_motion_planner()
        # cuRobo plans a smooth, in-limits, collision-free JOINT trajectory itself, so the motion_continuity
        # guard mis-fires across cuRobo moves: it memoises the PRE-RESOLVED IK (used only for this preflight),
        # but cuRobo EXECUTES a different (2π-equivalent) IK branch, so the next move's pre-resolved IK steps
        # ~360° vs the stale memo and the guard false-rejects. cuRobo owns continuity -> clear the memo first
        # (the same reason _JOINT_MOVE_SKIP_GUARDS skips continuity for a deliberate joint command).
        if planner == "curobo":
            if self._preflight is not None:
                self._preflight.reset()
            return self._drive_curobo(pose, current_joints=current_joints)
        rejected = self._preflight_reject(pose, target_joints=target_joints, current_joints=current_joints)
        if rejected is not None:
            return rejected
        if planner == "rmpflow" and self._rmpflow is not None:
            if not self._drive_rmpflow(pose):
                return MotionResult.failed(
                    MotionStatus.TIMEOUT,
                    MotionCommand.MOVE_TO,
                    target_pose=pose,
                    message=(
                        "RMPflow approach did not converge (obstacle-blocked / local minimum); "
                        "failing safe rather than blind-snapping through the obstacle."
                    ),
                )
        return self._drive_to_target(pose, target_joints)

    def _drive_to_target(
        self,
        pose: "Pose",
        target_joints: "JointPositions | None",
    ) -> MotionResult:
        """Drive to ``pose`` in joint space, retrying IK on non-convergence."""
        # Drive in joint space, retrying on non-convergence. A large move (e.g. from a parked arm) or a
        # load-perturbed move can stall short the first time
        err_mm = float("inf")
        for _attempt in range(_MOVE_MAX_ATTEMPTS):
            if target_joints is None:
                try:
                    target_joints = self._resolve_ik(pose)
                except RobotKinematicsError as exc:
                    return MotionResult.failed(
                        MotionStatus.IK_FAILED,
                        MotionCommand.MOVE_TO,
                        target_pose=pose,
                        message=f"IsaacRobotArm.move: IK failed — {exc}",
                    )
            self.move_joint(target_joints)
            tcp = self.get_tcp_pose()
            err_mm = float(
                np.linalg.norm(np.asarray(tcp.position_mm) - np.asarray(pose.position_mm))
            )
            ori_err_deg = _quat_angle_deg(tcp.quaternion_xyzw, pose.quaternion_xyzw)
            if err_mm <= _MOVE_POS_TOL_MM and ori_err_deg <= _MOVE_ORI_TOL_DEG:
                return MotionResult.executed(
                    MotionCommand.MOVE_TO,
                    target_pose=pose,
                    message=f"IK + move_joint reached target ({err_mm:.1f} mm, {ori_err_deg:.1f}°).",
                )
            if err_mm > _MOVE_RETRY_MAX_ERR_MM:
                break  # far miss: a re-resolve of the same target won't recover, fail fast
            target_joints = None  # near miss: re-resolve IK from the new configuration next attempt
        # A non-converging move is the single most opaque failure this driver can produce: it surfaces
        # as a bare TIMEOUT, which reads as "the grasp was bad" when it actually means the arm could not
        # reach the pose the grasp asked for.
        _LOGGER.warning(
            "move did NOT converge: %.1f mm from target (tol %.1f mm) after %d attempt(s); "
            "target=%s stalled at joints=%s",
            err_mm, _MOVE_POS_TOL_MM, _MOVE_MAX_ATTEMPTS,
            np.round(np.asarray(pose.position_mm), 1).tolist(),
            np.round(np.asarray(self.get_joint_positions().values), 3).tolist(),
        )
        return MotionResult.failed(
            MotionStatus.TIMEOUT,
            MotionCommand.MOVE_TO,
            target_pose=pose,
            message=f"IsaacRobotArm.move did not converge ({err_mm:.1f} mm > {_MOVE_POS_TOL_MM} mm).",
        )

    def _preflight_reject(
        self,
        pose: "Pose",
        *,
        target_joints: "JointPositions | None" = None,
        current_joints: "JointPositions | None" = None,
        skip_guards: "frozenset[str]" = frozenset(),
    ) -> "MotionResult | None":
        """Run the safety preflight for a Cartesian target."""
        if self._preflight is None:
            return None
        ctx = self._preflight.context_for_pose(
            pose,
            command=MotionCommand.MOVE_TO,
            target_joints=target_joints,
            current_joints=current_joints,
            arm=self,
        )
        decision = self._preflight.evaluate(ctx, skip_guards=skip_guards)
        return SafetyPreflight.as_motion_result(
            decision, MotionCommand.MOVE_TO, target_pose=pose,
        )
