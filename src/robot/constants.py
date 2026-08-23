"""Shared constants for the robot package."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:  # pragma: no cover - typing only
    from logging import Logger

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
ROBOT_LOG_FILE: Final[str] = "robot.log"
ROBOT_LOG_DIR: Final[str] = "logs/backend/robot"
#: Per-module sub-logs, the SECOND sink alongside the aggregate ``robot.log``.
ROBOT_MODULES_LOG_DIR: Final[str] = "logs/backend/robot/modules"

# Per-module log filenames, each subsystem logs to its own file here AND to the aggregate robot.log.
UR_CONNECTION_LOG_FILE: Final[str] = "ur_connection.log"
UR_ARM_LOG_FILE: Final[str] = "ur_arm.log"
UR_MOTION_LOG_FILE: Final[str] = "ur_motion.log"
UR_CUROBO_LOG_FILE: Final[str] = "ur_curobo.log"
KUKA_ARM_LOG_FILE: Final[str] = "kuka_arm.log"
KUKA_EKI_LOG_FILE: Final[str] = "kuka_eki.log"
SIM_ARM_LOG_FILE: Final[str] = "sim_arm.log"
GRIPPER_LOG_FILE: Final[str] = "gripper.log"
SAFETY_PREFLIGHT_LOG_FILE: Final[str] = "safety_preflight.log"
SAFETY_WORKSPACE_LOG_FILE: Final[str] = "safety_workspace.log"
POSE_PROVIDER_LOG_FILE: Final[str] = "pose_provider.log"
CALIBRATION_LOG_FILE: Final[str] = "calibration.log"
GRASP_CALCULATOR_LOG_FILE: Final[str] = "grasp_calculator.log"
MASK_ANALYZER_LOG_FILE: Final[str] = "mask_analyzer.log"
RUNTIME_PICK_LOG_FILE: Final[str] = "runtime_pick.log"

# -- driver selection + host readiness -------------------------------------------------------------
#: Which vendor factory was registered / instantiated.
DRIVER_REGISTRY_LOG_FILE: Final[str] = "driver_registry.log"
#: The fail-early vendor-SDK gate (``drivers/doctor.py``).
DRIVER_DOCTOR_LOG_FILE: Final[str] = "driver_doctor.log"

# -- digital-I/O bench (the numbers that become gripper config) ------------------------------------
#: Every pin driven and every transition timed on a real cell.
UR_IO_BENCH_LOG_FILE: Final[str] = "ur_io_bench.log"
#: The operator CLI in front of the bench, who asked for what, and every refusal.
UR_IO_CLI_LOG_FILE: Final[str] = "ur_io_cli.log"

# -- grippers --------------------------------------------------------------------------------------
GRIPPER_REGISTRY_LOG_FILE: Final[str] = "gripper_registry.log"
#: The two I/O end-effectors.
JAW_IO_GRIPPER_LOG_FILE: Final[str] = "jaw_io_gripper.log"
VACUUM_GRIPPER_LOG_FILE: Final[str] = "vacuum_gripper.log"

# -- execution / composition root ------------------------------------------------------------------
#: Who owned the cell and when, an ownership trail that outlives the process that held the lock.
CELL_LOCK_LOG_FILE: Final[str] = "cell_lock.log"
IK_SERVICE_LOG_FILE: Final[str] = "ik_service.log"
#: What ``from_robot_config`` actually built + which advanced overlays ended up live.
GRASP_BUILDERS_LOG_FILE: Final[str] = "grasp_builders.log"
GRASP_RECORD_LOG_FILE: Final[str] = "grasp_record_logging.log"
GRASP_LATENCY_LOG_FILE: Final[str] = "grasp_latency.log"
RL_SHADOW_LOG_FILE: Final[str] = "rl_shadow.log"

# -- motion-stack externals ------------------------------------------------------------------------
#: The process-isolated cuRobo sidecar: spawn, warm-up cost, and every plan it refused.
CUROBO_CLIENT_LOG_FILE: Final[str] = "curobo_client.log"
#: Which exact-mesh collision engine resolved (Coal / python-fcl / none).
PLANNING_ENVIRONMENT_LOG_FILE: Final[str] = "planning_environment.log"
#: The deep health check that LOADS every engine
PLANNING_DOCTOR_LOG_FILE: Final[str] = "planning_doctor.log"


def create_robot_logger(name: str, sub_file: str, level: int = logging.INFO) -> Logger:
    """Create a logger for a robot subsystem, with the standard file sinks."""
    from src.utility.log_cfg import create_logger

    return create_logger(
        name,
        sub_file,
        level=level,
        log_dir=ROBOT_MODULES_LOG_DIR,
        aggregate_file=ROBOT_LOG_FILE,
        aggregate_dir=ROBOT_LOG_DIR,
    )

# ----------------------------------------------------------------------
# Kinematics defaults
# ----------------------------------------------------------------------
HOME_JOINTS_DEFAULT: Final[tuple[float, ...]] = (
    0.0,
    -1.5707963267948966,  # -pi/2
    0.0,
    -1.5707963267948966,  # -pi/2
    0.0,
    0.0,
)


def home_joints_default() -> list[float]:
    """Return a fresh, mutable copy of :data:`HOME_JOINTS_DEFAULT`."""
    return list(HOME_JOINTS_DEFAULT)
