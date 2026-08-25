"""Vendor-neutral execution policy for approach, grasp, and retreat.

Owns the robot and gripper choreography for an executed grasp while the
orchestrator selects the candidate. Optionally verifies object detection
when the gripper provides the corresponding capability; otherwise the grasp
is trusted after the close command.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import cast

import numpy as np

from src.geometry import Frame, Pose
from src.geometry.quaternion import from_rotation_matrix
from src.robot.core import (
    Gripper,
    MotionCommand,
    MotionResult,
    MotionStatus,
    ObjectDetectingGripper,
    RobotArm,
)
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint

__all__ = [
    "GraspExecutionPolicy",
    "PolicyOutcome",
    "PolicyReport",
]


class PolicyOutcome(str, Enum):
    """Terminal status of :meth:`GraspExecutionPolicy.execute`."""

    EXECUTED = "executed"
    """Approach -> close -> retreat completed; object held (or trusted)."""

    OBJECT_NOT_DETECTED = "object_not_detected"
    """Close succeeded mechanically but the gripper reports an empty jaw."""

    MOTION_FAILED = "motion_failed"
    """:meth:`RobotArm.move_to` raised. The exception is in the report."""

    CAMERA_FRAME_REJECTED = "camera_frame_rejected"
    """Fail-closed guard: the policy refused to execute a camera-frame
    :class:`GraspPoint` because ``require_base_frame_grasp`` is enabled
    and no :class:`FrameResolver` was wired into the upstream
    orchestrator. No motion was commanded.
    """

    APPROACH_PATH_BLOCKED = "approach_path_blocked"
    """Every ranked candidate's approach/retreat SWEEP collided with a neighbour in the scene
    obstacle cloud (the swept-volume validator), so no candidate was executed.
    """


@dataclass(frozen=True, slots=True)
class PolicyReport:
    """Outcome of a :meth:`GraspExecutionPolicy.execute` call.

    Attributes
    ----------
    outcome
        Terminal :class:`PolicyOutcome`.
    waypoints
        Ordered tuple of :class:`Pose` instances the policy commanded.
    object_detected
        :data:`None` when the gripper has no detection capability or no
        gripper was configured; otherwise the reported boolean.
    error
        Optional :class:`Exception` instance for :attr:`PolicyOutcome.MOTION_FAILED`.
    motion_status
        Categorical :class:`MotionStatus` from the typed
        :meth:`RobotArm.move` surface. ``None`` when the driver only
        exposes the legacy bool ``move_to`` path.
        Populated whenever the driver implements the
        typed move contract, including the success case
        (:attr:`MotionStatus.EXECUTED`) so consumers can confirm a
        clean execution without checking ``outcome`` alone.
    motion_message
        Optional human-readable detail forwarded from the underlying
        :class:`MotionResult.message`. Empty string when not provided.
    """

    outcome: PolicyOutcome
    waypoints: tuple[Pose, ...] = ()
    object_detected: bool | None = None
    error: Exception | None = None
    motion_status: MotionStatus | None = None
    motion_message: str = ""


@dataclass
class GraspExecutionPolicy:
    """Approach / grasp / retreat strategy.

    Parameters
    ----------
    arm
        Any :class:`RobotArm` driver.
    gripper
        Optional :class:`Gripper`. When :data:`None`, the policy only
        drives the arm.
    standoff_mm
        Distance from the grasp point along the *reverse* approach axis
        for the pre-grasp waypoint. The policy interpolates linearly
        from pre-grasp to the grasp point in :attr:`approach_steps`
        waypoints, inclusive of both endpoints.
    retreat_mm
        Vertical lift (along world +Z) commanded after closing.
    approach_steps
        Number of inclusive waypoints between pre-grasp and the grasp
        point. Must be ``>= 2``.
    pre_open_width_mm
        Optional jaw width commanded *before* the approach when a gripper
        is present. ``None`` skips the pre-open command.
    close_width_mm
        Jaw width commanded at the grasp point. When :data:`None`, falls
        back to ``max(gripper.min_width_mm, grasp.grip_width_mm - 1.0)``
        at execution time.
    close_speed
        Optional gripper speed in ``[0.0, 1.0]``.
    close_force_n
        Optional gripper force in newtons.
    """

    arm: RobotArm
    gripper: Gripper | None = None
    standoff_mm: float = 80.0
    retreat_mm: float = 100.0
    approach_steps: int = 4
    # Split the post-close vertical lift into ``retreat_steps`` interpolated waypoints, each driven
    # + settled (move_joint settles after every move).
    retreat_steps: int = 1
    pre_open_width_mm: float | None = None
    close_width_mm: float | None = None
    # Squeeze margin (mm) used ONLY when ``close_width_mm is None``: the jaw closes to
    # ``max(min_width, grasp.grip_width_mm - close_squeeze_mm)`` so the close ADAPTS to each object's
    # measured grip width (a firm squeeze below the surface holds it) without per-object tuning.
    close_squeeze_mm: float = 1.0
    close_speed: float | None = None
    close_force_n: float | None = None
    # Opt-in fail-closed frame guard. When ``True`` the policy refuses
    # to execute a camera-frame :class:`GraspPoint` and returns
    # :attr:`PolicyOutcome.CAMERA_FRAME_REJECTED` without commanding any motion.
    require_base_frame_grasp: bool = False
    # Pre-move steady-state temporal gate (DwellSafetyConfig). When True the policy blocks on
    # ``arm.wait_until_steady(steady_timeout_s)`` BEFORE every commanded move and FAILS CLOSED (no
    # motion) on timeout.
    require_steady_before_motion: bool = False
    steady_timeout_s: float = 5.0
    # Opt-in grasp-orientation alignment for SYMMETRIC top-down grasps. When True the
    # policy rigidly yaws each grasp's orientation about base-Z so the gripper CLOSING axis falls on the
    # trackable base-X (keeping position + the ~vertical approach).
    align_closing_to_base_x: bool = False
    # When True the approach is driven as a SINGLE goal (the grasp pose) so the GLOBAL
    # planner (cuRobo) plans the whole collision-free descent ITSELF, instead of tracking the interpolated
    # standoff->grasp waypoints.
    planner_owns_approach: bool = False

    def __post_init__(self) -> None:
        if self.approach_steps < 2:
            raise ValueError(
                f"approach_steps must be >= 2, got {self.approach_steps}"
            )
        if self.standoff_mm < 0.0:
            raise ValueError(f"standoff_mm must be >= 0, got {self.standoff_mm}")
        if self.retreat_mm < 0.0:
            raise ValueError(f"retreat_mm must be >= 0, got {self.retreat_mm}")
        if self.retreat_steps < 1:
            raise ValueError(f"retreat_steps must be >= 1, got {self.retreat_steps}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, grasp: GraspPoint) -> PolicyReport:
        """Drive the arm + gripper through the full grasp sequence."""
        # Fail-closed guard: when enabled, a camera-frame grasp is
        # never executed. Doing so would silently move the arm to a
        # camera-frame pose.
        if self.require_base_frame_grasp and grasp.frame is not GraspFrame.BASE:
            return PolicyReport(
                outcome=PolicyOutcome.CAMERA_FRAME_REJECTED,
                waypoints=(),
                motion_message=(
                    "refused to execute a camera-frame grasp with no "
                    "FrameResolver wired (require_base_frame_grasp=True)"
                ),
            )
        waypoints = self._build_waypoints(grasp)

        # Pre-open the gripper before driving the approach so the jaws
        # are clear once we reach the grasp point.
        if self.gripper is not None and self.pre_open_width_mm is not None:
            self.gripper.set_width_mm(
                float(self.pre_open_width_mm),
                speed=self.close_speed,
                force=None,
            )

        commanded: list[Pose] = []
        last_status: MotionStatus | None = None
        last_message: str = ""
        approach = waypoints[:-self.retreat_steps]  # all but the retreat lift(s)
        if self.planner_owns_approach:
            # cuRobo plans the full collision-free descent to the grasp itself -> drive ONLY the grasp (the
            # last approach waypoint), skipping the interpolated standoff/approach segments the blind-IK path needs.
            approach = approach[-1:]
        for pose in approach:
            try:
                result = self._drive_to(pose)
            except Exception as exc:  # noqa: BLE001 - propagate via report
                return PolicyReport(
                    outcome=PolicyOutcome.MOTION_FAILED,
                    waypoints=tuple(commanded),
                    error=exc,
                    motion_status=last_status,
                    motion_message=last_message,
                )
            if result is not None:
                last_status = result.status
                last_message = result.message
                if not result.ok:
                    return PolicyReport(
                        outcome=PolicyOutcome.MOTION_FAILED,
                        waypoints=tuple(commanded),
                        error=cast("Exception | None", result.exception),
                        motion_status=result.status,
                        motion_message=result.message,
                    )
            commanded.append(pose)

        # Close the gripper at the grasp point.
        object_detected: bool | None = None
        if self.gripper is not None:
            target_width = self._resolve_close_width(grasp)
            self.gripper.set_width_mm(
                target_width,
                speed=self.close_speed,
                force=self.close_force_n,
            )
            if isinstance(self.gripper, ObjectDetectingGripper):
                object_detected = bool(self.gripper.is_object_detected())
                if not object_detected:
                    return PolicyReport(
                        outcome=PolicyOutcome.OBJECT_NOT_DETECTED,
                        waypoints=tuple(commanded),
                        object_detected=False,
                        motion_status=last_status,
                        motion_message=last_message,
                    )

        # Retreat: command each interpolated lift waypoint (retreat_steps; default 1 = the single full lift).
        for pose in waypoints[-self.retreat_steps:]:
            try:
                result = self._drive_to(pose)
            except Exception as exc:  # noqa: BLE001 - propagate via report
                return PolicyReport(
                    outcome=PolicyOutcome.MOTION_FAILED,
                    waypoints=tuple(commanded),
                    object_detected=object_detected,
                    error=exc,
                    motion_status=last_status,
                    motion_message=last_message,
                )
            if result is not None:
                last_status = result.status
                last_message = result.message
                if not result.ok:
                    return PolicyReport(
                        outcome=PolicyOutcome.MOTION_FAILED,
                        waypoints=tuple(commanded),
                        object_detected=object_detected,
                        error=cast("Exception | None", result.exception),
                        motion_status=result.status,
                        motion_message=result.message,
                    )
            commanded.append(pose)

        return PolicyReport(
            outcome=PolicyOutcome.EXECUTED,
            waypoints=tuple(commanded),
            object_detected=object_detected,
            motion_status=last_status,
            motion_message=last_message,
        )

    # ------------------------------------------------------------------
    # Motion adapter
    # ------------------------------------------------------------------

    def _drive_to(self, pose: Pose) -> MotionResult | None:
        """Drive ``arm`` to ``pose`` via the best available surface."""
        # Pre-move steady gate (fail-closed temporal check, OUTSIDE the per-target SafetyPreflight
        # pipeline).
        if self.require_steady_before_motion:
            wait_fn = getattr(self.arm, "wait_until_steady", None)
            if callable(wait_fn) and not wait_fn(float(self.steady_timeout_s)):
                return MotionResult.failed(
                    MotionStatus.TIMEOUT,
                    MotionCommand.OTHER,
                    target_pose=pose,
                    message=(
                        "pre-move steady gate timed out after "
                        f"{self.steady_timeout_s:.2f}s (require_steady_before_motion)"
                    ),
                )
        typed_move = getattr(self.arm, "move", None)
        if callable(typed_move):
            return typed_move(pose)
        self.arm.move_to(pose)
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_close_width(self, grasp: GraspPoint) -> float:
        """Pick the jaw width to command at the grasp point."""
        if self.close_width_mm is not None:
            return float(self.close_width_mm)
        # Adaptive: squeeze ``close_squeeze_mm`` below the predicted grip width (default 1 mm = the legacy
        # hug) but never below the gripper's minimum opening. A larger margin clamps thin objects firmly.
        gripper = self.gripper
        min_w = getattr(gripper, "min_width_mm", 0.0) or 0.0
        return float(max(min_w, float(grasp.grip_width_mm) - float(self.close_squeeze_mm)))

    def _build_waypoints(self, grasp: GraspPoint) -> tuple[Pose, ...]:
        """Build the [approach_0..approach_N-1, retreat] waypoint sequence."""
        closing = np.asarray(grasp.axis, dtype=np.float64)
        approach_unit = np.asarray(grasp.approach, dtype=np.float64)
        target = np.asarray(grasp.position, dtype=np.float64)
        if self.align_closing_to_base_x:  # yaw the close onto the reachable base-X sign for this side
            prefer_plus_x = float(target[1]) < 0.0  # UR5e: -Y reaches with +X close, +Y with -X
            closing, approach_unit = _yaw_axes_to_base_x(closing, approach_unit, prefer_plus_x=prefer_plus_x)
        quat = _quaternion_from_axes(closing, approach_unit)
        pre = target - approach_unit * float(self.standoff_mm)
        frame = Frame(grasp.frame.value)
        poses: list[Pose] = []
        for index, fraction in enumerate(np.linspace(0.0, 1.0, num=self.approach_steps)):
            position = pre + (target - pre) * float(fraction)
            poses.append(
                Pose(
                    position_mm=position,
                    quaternion_xyzw=quat.copy(),
                    frame=frame,
                    label=f"approach_{index:02d}",
                )
            )
        # Retreat: ``retreat_steps`` interpolated vertical lifts from the grasp point to grasp + retreat_mm
        # (k=1..N).
        for k in range(1, self.retreat_steps + 1):
            fraction = float(k) / float(self.retreat_steps)
            retreat_position = target + np.array([0.0, 0.0, float(self.retreat_mm) * fraction])
            label = "retreat" if self.retreat_steps == 1 else f"retreat_{k:02d}"
            poses.append(
                Pose(
                    position_mm=retreat_position,
                    quaternion_xyzw=quat.copy(),
                    frame=frame,
                    label=label,
                )
            )
        return tuple(poses)


def _quaternion_from_axes(closing: np.ndarray, approach: np.ndarray) -> np.ndarray:
    """Unit XYZW quaternion from a grasp's closing + approach axes.

    Rotation columns are ``[closing, binormal, approach]`` with the binormal recovered from
    ``approach x closing`` so the matrix is right-handed and orthonormal.
    """
    closing = np.asarray(closing, dtype=np.float64)
    approach = np.asarray(approach, dtype=np.float64)
    binormal = np.cross(approach, closing)
    binormal_norm = float(np.linalg.norm(binormal))
    if binormal_norm < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    binormal /= binormal_norm
    R = np.column_stack([closing, binormal, approach])
    return np.asarray(from_rotation_matrix(R), dtype=np.float64)


def _grasp_point_to_quaternion(grasp: GraspPoint) -> np.ndarray:
    """Rebuild a unit XYZW quaternion from a ``GraspPoint``'s frame axes (closing + approach)."""
    return _quaternion_from_axes(
        np.asarray(grasp.axis, dtype=np.float64), np.asarray(grasp.approach, dtype=np.float64)
    )


def _yaw_axes_to_base_x(
    closing: np.ndarray, approach: np.ndarray, *, prefer_plus_x: bool
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rigidly yaw ``(closing, approach)`` about base-Z so the CLOSING axis points along base **+X**
    (``prefer_plus_x``) or base **-X**, keeping a ~vertical approach.
    """
    closing = np.asarray(closing, dtype=np.float64)
    approach = np.asarray(approach, dtype=np.float64)
    ch = np.array([closing[0], closing[1], 0.0], dtype=np.float64)
    n = float(np.linalg.norm(ch))
    if n < 1e-6:  # closing axis is (near-)vertical -> no meaningful horizontal yaw
        return closing, approach
    ang = float(np.arctan2(ch[1], ch[0]))
    target = 0.0 if prefer_plus_x else np.pi  # base +X or base -X (the reachable side)
    theta = target - ang
    c, s = float(np.cos(theta)), float(np.sin(theta))
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ closing, rz @ approach
