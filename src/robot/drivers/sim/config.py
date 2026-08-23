"""Pure-Python configuration dataclasses for the Isaac sim driver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "SimCameraConfig",
    "SimRobotConfig",
]


CameraMountingMode = Literal["eye_to_hand", "eye_in_hand"]


@dataclass(frozen=True, slots=True)
class SimCameraConfig:
    """One camera attached to the simulated scene.

    Attributes
    ----------
    prim_path
        Simulator prim path that owns the camera sensor.
    mounting_mode
        ``"eye_to_hand"`` for cameras fixed in the world frame;
        ``"eye_in_hand"`` for cameras attached to the end-effector. The
        :class:`backend.src.robot.execution.calibration.CalibrationRoutine`
        consumes this to pick the matching calibrator (ETH vs EIH).
    """

    prim_path: str
    mounting_mode: CameraMountingMode = "eye_to_hand"


@dataclass(frozen=True, slots=True)
class SimRobotConfig:
    """Top-level config block for one Isaac-backed robot cell.

    Attributes
    ----------
    backend
        Always ``"isaac"`` for now. Reserved as a discriminator so a
        future PyBullet / MuJoCo backend can live in the same slot.
    enabled
        Master switch. When ``False`` the registry SHOULD raise rather
        than silently spin up an Isaac session.
    scene
        Scene identifier or USD path. Driver opens this on connect.
    robot_prim_path
        Simulator prim path of the articulation Willy drives.
    gripper_prim_path
        Optional prim path of the gripper articulation / asset.
    cameras
        Mapping of camera name -> :class:`SimCameraConfig`. Empty when
        the slice runs without simulated cameras.
    home_joint_positions
        Explicit home joints. Driver defaults to scene-defined home
        when omitted.
    step_dt_s
        Deterministic simulator step (seconds). Driver authors MUST use
        this for both motion stepping and ``wait_until_steady``.
    settle_timeout_s
        Maximum sim time (seconds) the driver will wait for a steady
        signal before returning :attr:`MotionStatus.TIMEOUT`.
    motion_planner
        Approach-phase planner: ``"ik"`` (default, the blind pure-IK
        straight line, byte-identical) or ``"rmpflow"`` (drive the
        reactive collision-aware policy first, then IK-snap). The seam a
        CuRobo planner plugs into later.
    headless
        Run Isaac without a viewport. Recommended for CI / batch runs.
    mock_mode
        When ``True`` the driver runs as a pure-Python kinematic mock.
    """

    backend: Literal["isaac"] = "isaac"
    enabled: bool = False
    # Which UR model this sim cell drives (de-locks the historically UR5e-hardcoded driver).
    robot_model: str = "ur5e"
    scene: str | None = None
    robot_prim_path: str | None = None
    gripper_prim_path: str | None = None
    cameras: dict[str, SimCameraConfig] = field(default_factory=dict)
    home_joint_positions: tuple[float, ...] | None = None
    step_dt_s: float = 1.0 / 60.0
    settle_timeout_s: float = 5.0
    tool_offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tool_rotation_quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    # CuRobo is the only planner that can drive the RMPflow policy, so it is the default.
    motion_planner: Literal["ik", "rmpflow", "curobo"] = "curobo"
    headless: bool = True
    mock_mode: bool = False
