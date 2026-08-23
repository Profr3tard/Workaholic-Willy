"""cuRobo-planned motion for the REAL UR arm (plan via safety.planning, execute via ur_rtde).

Real-hardware counterpart of the Isaac sim driver's cuRobo path
(:meth:`src.robot.drivers.sim.arm.IsaacRobotArm._drive_curobo`): ask the
process-isolated cuRobo planner
(:class:`~src.robot.safety.planning.CuroboPlanClient`) for a global
collision-free joint trajectory to a Cartesian goal, then EXECUTE that trajectory on
the UR controller waypoint-by-waypoint via ``ur_rtde`` ``moveJ``.

FAIL-CLOSED: if the cuRobo env/service is unavailable, or no collision-free plan
exists, this does NOT fall back to blind IK, it returns a typed failure
(:attr:`MotionStatus.CONTROLLER_REJECTED` / :attr:`MotionStatus.TIMEOUT`) so a real
cell never moves on an unplanned path.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from src.geometry import Frame, Pose
from src.robot.constants import UR_CUROBO_LOG_FILE, create_robot_logger
from src.robot.core import MotionCommand, MotionResult, MotionStatus
from src.robot.safety.planning import CuroboPlanClient, CuroboUnavailableError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .connection import URConnection

__all__ = ["CuroboUrPlanner", "UR_ARM_JOINT_NAMES"]

#: UR arm joint order (base -> wrist_3): the order ``ur_rtde`` reports ``getActualQ`` and expects ``moveJ``.
UR_ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


class CuroboUrPlanner:
    """Plan a UR joint trajectory with cuRobo and execute it over ``ur_rtde`` (fail-closed).

    Parameters
    ----------
    connection
        A connected :class:`~src.robot.drivers.ur.connection.URConnection`
        (``is_connected`` / ``get_joint_positions`` / ``moveJ``).
    client_factory
        Builds the :class:`CuroboPlanClient` (injected as a fake in tests).
    vel / acc
        Default joint speed / acceleration for each executed ``moveJ`` waypoint.
    """

    def __init__(
        self,
        connection: URConnection,
        *,
        client_factory: Callable[[], CuroboPlanClient] = CuroboPlanClient,
        vel: float | None = None,
        acc: float | None = None,
    ) -> None:
        self._conn = connection
        self._client_factory = client_factory
        self._vel = vel
        self._acc = acc
        self._client: CuroboPlanClient | None = None
        self.logger = create_robot_logger("CuroboUrPlanner", UR_CUROBO_LOG_FILE)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _client_or_start(self) -> CuroboPlanClient:
        """Lazily spawn + JIT-warm the cuRobo server (raises CuroboUnavailableError if the env is missing)."""
        if self._client is None:
            client = self._client_factory()
            client.start()
            self._client = client
        return self._client

    def set_world(self, cuboids: list[dict]) -> int:
        """Register scene obstacles into cuRobo's collision world; 0 if the planner is unavailable."""
        try:
            return self._client_or_start().set_world(cuboids)
        except CuroboUnavailableError:
            return 0

    def close(self) -> None:
        """Shut down the planning server (idempotent)."""
        if self._client is not None:
            self._client.close()
            self._client = None

    # ------------------------------------------------------------------
    # Planning + execution
    # ------------------------------------------------------------------

    def plan(self, pose: Pose) -> list[list[float]] | None:
        """Plan tool0 -> ``pose`` (BASE) with cuRobo. Returns the joint trajectory in UR order, or ``None``.

        Raises
        ------
        CuroboUnavailableError
            If the cuRobo env/service cannot be brought up (caller must fail closed).
        """
        goal_pos_m = (np.asarray(pose.position_mm, dtype=np.float64) / 1000.0).tolist()
        qx, qy, qz, qw = (float(v) for v in pose.quaternion_xyzw)
        goal_quat_wxyz = [qw, qx, qy, qz]

        current_ur = [float(v) for v in self._conn.get_joint_positions()]
        client = self._client_or_start()
        start = self._to_client_order(current_ur, client.joint_names)
        traj = client.plan(start, goal_pos_m, goal_quat_wxyz)
        if not traj:
            return None
        return [self._to_ur_order(list(wp), client.joint_names) for wp in traj]

    def execute(
        self,
        traj_ur: list[list[float]],
        pose: Pose,
        *,
        vel: float | None = None,
        acc: float | None = None,
    ) -> MotionResult:
        """Execute a UR-order joint trajectory waypoint-by-waypoint via ``moveJ``."""
        v = vel if vel is not None else self._vel
        a = acc if acc is not None else self._acc
        try:
            for waypoint in traj_ur:
                ok = self._conn.moveJ(list(waypoint), vel=v, acc=a)
                if not ok:
                    return MotionResult.failed(
                        MotionStatus.CONTROLLER_REJECTED,
                        MotionCommand.MOVE_TO,
                        target_pose=pose,
                        message="UR moveJ rejected a cuRobo trajectory waypoint.",
                    )
        except (RuntimeError, OSError) as exc:
            return MotionResult.failed(
                MotionStatus.CONNECTION_ERROR,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message=f"UR moveJ raised executing the cuRobo trajectory: {exc}",
                exception=exc,
            )
        self.logger.info("cuRobo trajectory executed on UR: %d waypoints", len(traj_ur))
        return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose, message="curobo")

    def move(self, pose: Pose, *, vel: float | None = None, acc: float | None = None) -> MotionResult:
        """Plan tool0 -> ``pose`` (BASE) with cuRobo and execute it. Fail-closed on any failure."""
        if pose.frame is not Frame.BASE:
            return MotionResult.failed(
                MotionStatus.INVALID_TARGET,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message=f"CuroboUrPlanner requires Frame.BASE; got {pose.frame!r}",
            )
        if not self._conn.is_connected:
            return MotionResult.failed(
                MotionStatus.CONNECTION_ERROR,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message="CuroboUrPlanner requires an open UR connection.",
            )
        try:
            traj_ur = self.plan(pose)
        except CuroboUnavailableError as exc:
            return MotionResult.failed(
                MotionStatus.CONTROLLER_REJECTED,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message=f"cuRobo planner unavailable: {exc}",
                exception=exc,
            )
        if not traj_ur:
            return MotionResult.failed(
                MotionStatus.TIMEOUT,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message="cuRobo found no collision-free plan; failing safe (no blind motion).",
            )
        return self.execute(traj_ur, pose, vel=vel, acc=acc)

    # ------------------------------------------------------------------
    # Joint-order remap (planner <--> UR)
    # ------------------------------------------------------------------

    @staticmethod
    def _to_client_order(ur_joints: list[float], client_names: list[str]) -> list[float]:
        """Reorder UR-order joints into the planner's joint order (identity if names are unknown)."""
        if not client_names or set(client_names) != set(UR_ARM_JOINT_NAMES):
            return [float(v) for v in ur_joints]
        idx = {name: i for i, name in enumerate(UR_ARM_JOINT_NAMES)}
        return [float(ur_joints[idx[name]]) for name in client_names]

    @staticmethod
    def _to_ur_order(client_joints: list[float], client_names: list[str]) -> list[float]:
        """Reorder planner-order joints back into UR joint order (identity if names are unknown)."""
        if not client_names or set(client_names) != set(UR_ARM_JOINT_NAMES):
            return [float(v) for v in client_joints]
        pos = {name: i for i, name in enumerate(client_names)}
        return [float(client_joints[pos[name]]) for name in UR_ARM_JOINT_NAMES]
