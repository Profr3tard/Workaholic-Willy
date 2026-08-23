"""
Dummy / sim arm driver — pure-Python implementation of
:class:`src.robot.core.RobotArm`.

Useful for:

* Unit tests that exercise pipelines / planners without real hardware.
* Offline development of new pipelines on a laptop.
* Smoke-testing the vendor-neutral surface itself.

The driver maintains an in-memory ``(joints, tcp_pose)`` state. There
is no kinematic model: ``move_joint`` simply records the new joint
vector, ``move_linear`` records the new pose, ``fk`` returns the last
recorded pose, ``ik`` returns the last recorded joints.
"""

from __future__ import annotations

import numpy as np

from src.geometry import Frame, FrameMismatchError, Pose
from src.geometry.quaternion import IDENTITY_QUAT_XYZW

from ...core import (
    JointPositions,
    MotionCommand,
    MotionResult,
    MotionStatus,
    RobotArm,
    RobotCapabilities,
    RobotConnectionError,
    RobotMotionRejected,
)

__all__ = ["DUMMY_CAPABILITIES", "DummyRobotArm"]


DUMMY_CAPABILITIES = RobotCapabilities(
    vendor="dummy",
    model="dummy-6dof",
    dof=6,
    supports_joint_move=True,
    supports_linear_move=True,
    supports_async_move=False,
    has_native_fk=False,
    has_native_ik=False,
    has_force_control=False,
    is_simulated=True,
)


class DummyRobotArm(RobotArm):
    """Lightweight in-memory arm satisfying :class:`RobotArm`.

    Parameters
    ----------
    dof : int
        Degrees of freedom for the synthetic arm. Default 6 (UR-shaped).
    initial_pose : Pose, optional
        TCP pose returned by :meth:`get_tcp_pose` until something else
        is commanded. Must be in :attr:`Frame.BASE`. Defaults to
        position ``(400, 0, 300) mm`` with identity orientation.
    """

    def __init__(
        self,
        *,
        dof: int = 6,
        initial_pose: Pose | None = None,
    ) -> None:
        self._capabilities = (
            DUMMY_CAPABILITIES
            if dof == DUMMY_CAPABILITIES.dof
            else RobotCapabilities(
                vendor="dummy", model=f"dummy-{dof}dof", dof=dof,
                is_simulated=True,
                supports_joint_move=True, supports_linear_move=True,
                supports_async_move=False,
                has_native_fk=False, has_native_ik=False,
                has_force_control=False,
            )
        )
        self._connected = False
        self._joints = JointPositions(np.zeros(dof, dtype=np.float64))
        if initial_pose is None:
            initial_pose = Pose(
                position_mm=np.array([400.0, 0.0, 300.0]),
                quaternion_xyzw=IDENTITY_QUAT_XYZW,
                frame=Frame.BASE,
                label="dummy-home",
            )
        if initial_pose.frame is not Frame.BASE:
            raise FrameMismatchError(
                f"DummyRobotArm initial_pose must be in Frame.BASE; got {initial_pose.frame!r}."
            )
        self._tcp = initial_pose
        self._home_pose = initial_pose

    # ---- introspection ----

    @property
    def capabilities(self) -> RobotCapabilities:
        return self._capabilities

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ---- lifecycle ----

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    # ---- state ----

    def get_tcp_pose(self) -> Pose:
        if not self._connected:
            raise RobotConnectionError("DummyRobotArm is not connected.")
        return self._tcp

    def get_joint_positions(self) -> JointPositions:
        if not self._connected:
            raise RobotConnectionError("DummyRobotArm is not connected.")
        return self._joints

    # ---- motion ----

    def move_to_joints(
        self,
        joints: JointPositions,
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
    ) -> MotionResult:
        """The dummy carries no preflight, so it simply drives (see Protocol)."""
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
        if not self._connected:
            raise RobotConnectionError("DummyRobotArm is not connected.")
        if joints.dof != self._capabilities.dof:
            raise RobotMotionRejected(
                f"DummyRobotArm expected {self._capabilities.dof} DoF, got {joints.dof}."
            )
        self._joints = joints

    def move_linear(
        self,
        pose: Pose,
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
    ) -> None:
        if not self._connected:
            raise RobotConnectionError("DummyRobotArm is not connected.")
        if pose.frame is not Frame.BASE:
            raise FrameMismatchError(
                f"DummyRobotArm.move_linear requires Frame.BASE; got {pose.frame!r}."
            )
        self._tcp = pose

    def stop(self) -> None:
        # No async motion in flight, nothing to interrupt.
        pass

    # ---- kinematics (stubbed) ----

    def fk(self, joints: JointPositions) -> Pose:
        # Joints are recorded but not used
        # FK returns the last commanded pose.
        if joints.dof != self._capabilities.dof:
            raise RobotMotionRejected(
                f"DummyRobotArm.fk expected {self._capabilities.dof} DoF, got {joints.dof}."
            )
        return self._tcp

    def ik(
        self,
        pose: Pose,
        *,
        seed: JointPositions | None = None,
    ) -> JointPositions:
        if pose.frame is not Frame.BASE:
            raise FrameMismatchError(
                f"DummyRobotArm.ik requires Frame.BASE; got {pose.frame!r}."
            )
        return self._joints

    # ---- high-level pipeline surface ------------------------------------

    def is_inside_workspace(self, pose: Pose) -> bool:
        """Dummy has no workspace box always accept BASE-frame poses."""
        if pose.frame is not Frame.BASE:
            raise FrameMismatchError(
                f"DummyRobotArm.is_inside_workspace requires Frame.BASE; got {pose.frame!r}."
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
        """Record ``pose`` as the new TCP pose. Always succeeds.

        ``register`` is accepted for protocol compatibility but ignored:
        the dummy driver maintains no diversity history.
        """
        if not self._connected:
            raise RobotConnectionError("DummyRobotArm is not connected.")
        if pose.frame is not Frame.BASE:
            raise FrameMismatchError(
                f"DummyRobotArm.move_to requires Frame.BASE; got {pose.frame!r}."
            )
        self._tcp = pose
        return True

    def move_home(self) -> bool:
        """Reset to the configured home TCP pose. Always succeeds."""
        if not self._connected:
            raise RobotConnectionError("DummyRobotArm is not connected.")
        self._tcp = self._home_pose
        self._joints = JointPositions(np.zeros(self._capabilities.dof, dtype=np.float64))
        return True

    def wait_until_steady(
        self,
        timeout_s: float = 5.0,
        poll_interval_s: float = 0.02,
    ) -> bool:
        """Pure-Python sim has no motion in flight; settle is instant."""
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
        """Typed counterpart of :meth:`move_to`.

        Dummy driver has no workspace box and no real controller, so
        successful moves always return :attr:`MotionStatus.EXECUTED`.
        Connection / frame faults are reported as typed failures
        rather than raised.
        """
        if pose.frame is not Frame.BASE:
            return MotionResult.failed(
                MotionStatus.INVALID_TARGET,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message=f"DummyRobotArm.move requires Frame.BASE; got {pose.frame!r}",
            )
        if not self._connected:
            return MotionResult.failed(
                MotionStatus.CONNECTION_ERROR,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message="DummyRobotArm is not connected",
            )
        self._tcp = pose
        return MotionResult.executed(
            MotionCommand.MOVE_TO, target_pose=pose,
        )
