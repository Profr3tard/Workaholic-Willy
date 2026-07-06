# core — vendor-neutral robot contract layer

`backend.src.robot.core` defines the typed contracts every robot/gripper driver and every
higher pipeline must agree on: the `RobotArm`/`Gripper` Protocols, the vendor enums, the
motion-result and joint-vector value objects, the capability descriptor, and the
vendor-neutral error hierarchy. In the strict downward stack
(`config → geometry → perception → robot/core → drivers → safety → grasping → rl → replay`)
it sits just above `geometry`: it imports `Pose`/`Frame` from `geometry` but is otherwise a
leaf — **no vendor SDK may ever be imported here**, and nothing in `core` imports `drivers`,
`safety`, or anything above it.

## Responsibilities

- Own the abstract `RobotArm` and `Gripper` driver surfaces (runtime-checkable `Protocol`s).
- Own the canonical `RobotVendor` / `GripperVendor` `StrEnum`s used by config + the driver registries.
- Own the typed motion outcome contract: `MotionResult`, `MotionStatus`, `MotionCommand`.
- Own the typed joint-vector value object `JointPositions` (radians, validated, immutable).
- Own the declarative `RobotCapabilities` feature descriptor drivers advertise for introspection.
- Own the vendor-neutral `RobotError` exception hierarchy that drivers translate vendor faults into.

## Public API

Exported from `backend.src.robot.core` (`__init__.py`):

- `RobotArm` — runtime-checkable `Protocol` for arm drivers. Surface: `capabilities`,
  `is_connected` (properties); `connect()`/`disconnect()`/`stop()`; `get_tcp_pose() -> Pose`,
  `get_joint_positions() -> JointPositions`; `move_joint(joints, *, velocity, acceleration)`,
  `move_linear(pose, *, velocity, acceleration)`; `fk(joints) -> Pose`, `ik(pose, *, seed) -> JointPositions`;
  high-level pipeline helpers `is_inside_workspace(pose) -> bool`,
  `move_to(pose, *, linear, vel, acc, register) -> bool`, `move_home() -> bool`,
  `wait_until_steady(timeout_s, poll_interval_s) -> bool`; and the typed surface
  `move(pose, *, linear, vel, acc, register) -> MotionResult` and
  `move_to_joints(joints, *, velocity, acceleration) -> MotionResult`.
- `Gripper` — runtime-checkable `Protocol` for parallel-jaw grippers: `is_connected`,
  `min_width_mm`, `max_width_mm` (properties); `connect()`/`disconnect()`/`activate()`;
  `set_width_mm(width_mm, *, speed, force)`, `get_width_mm() -> float`.
- `ObjectDetectingGripper` — opt-in `Protocol` extension adding `is_object_detected() -> bool`,
  consumed by the grasp execution policy for post-close verification.
- `RobotVendor(StrEnum)` — `UR`, `KUKA`, `FRANKA`, `ROS2`, `SIM`, `DUMMY`; plus
  `RobotVendor.from_string(value)` (case-insensitive coercion, raises on unknown).
- `GripperVendor(StrEnum)` — `ROBOTIQ`, `FRANKA_HAND`, `SCHUNK`, `DUMMY`, `NONE`; plus
  `GripperVendor.from_string(value)`.
- `MotionStatus(StrEnum)` — closed outcome set; `EXECUTED` is the only success value.
- `MotionCommand(StrEnum)` — `MOVE_TO`, `MOVE_HOME`, `MOVE_JOINTS`, `OTHER`.
- `MotionResult` — frozen dataclass `(status, command, target_pose=None, target_joints=None,
  message="", exception=None)`. Property `ok`; truthy via `__bool__`; constructors
  `MotionResult.executed(...)`, `MotionResult.failed(status, ...)`, `MotionResult.from_bool(ok, ...)`.
- `JointPositions(values)` — immutable radians vector; `.dof`, `.values` (read-only ndarray),
  `len`/`iter`/`getitem`, `np.asarray()` support, exact `__eq__`/`__hash__`, `.tolist()`,
  `JointPositions.from_list(values)`.
- `RobotCapabilities` — frozen, slotted descriptor:
  `(vendor, model="", dof=6, supports_joint_move=True, supports_linear_move=True,
  supports_async_move=False, has_native_fk=False, has_native_ik=False,
  has_force_control=False, is_simulated=False)`; validates a lowercase/whitespace-free `vendor`
  and `dof > 0`.
- `RobotError` and subclasses: `RobotConnectionError`, `RobotKinematicsError`,
  `RobotMotionRejected`, `RobotSingularityRisk` (subclass of `RobotMotionRejected`),
  `RobotEmergencyStop`.

## How it fits

- **Depends on:** `backend.src.geometry` (`Pose`, `Frame`, `FrameMismatchError`) and `numpy`.
  Nothing else inside `willy_backend`. `Pose`/`Frame` are *defined in `geometry`*, not here —
  `core` only consumes and frame-tags them. Poses are millimetres + canonical XYZW quaternions,
  frame-tagged; `get_tcp_pose()`/`fk()` return `Frame.BASE`; joints are radians.
- **Used by:** every arm driver (`drivers/ur`, `drivers/kuka`, `drivers/sim`, …), every gripper
  (`grippers/`), the safety pipeline (`safety/preflight.py`, `decision.py`, `singularity.py`,
  `guard.py`), the grasping layer (`grasping/pick_loop.py`, `execution_policy.py`,
  `recovery_orchestrator.py`), the execution/runtime facade
  (`execution/runtime_pick.py`, `execution/calibration.py`, `execution/autonomous_grasp/service.py`),
  and the Isaac sim runners (`willy_sim/run_m1_pick.py`, `run_m2_pick.py`).

## Key files

- `robot_arm.py` — `RobotArm` Protocol (bool surface + typed `move`).
- `gripper.py` — `Gripper` + `ObjectDetectingGripper` Protocols.
- `motion_result.py` — `MotionStatus`, `MotionCommand`, `MotionResult`.
- `joint_positions.py` — `JointPositions` value object.
- `capabilities.py` — `RobotCapabilities` descriptor.
- `vendor.py` / `gripper_vendor.py` — `RobotVendor` / `GripperVendor` enums.
- `errors.py` — `RobotError` hierarchy.
- `__init__.py` — re-exports the public surface listed above.

## Status & gaps (HONEST)

- **Two motion surfaces coexist.** The bool helpers (`move_to`, `move_home`,
  `is_inside_workspace`, `wait_until_steady`) and the typed `move() -> MotionResult`
  both live on `RobotArm`; the typed surface is preferred for runtime/calibration but the bool
  shims are deliberately kept while callers are migrated. This is intentional, not dead code.
- **`MotionStatus` has 15 members, not 10.** Beyond the 10 base outcomes
  (`EXECUTED`, `WORKSPACE_REJECTED`, `IK_FAILED`, `CONTROLLER_REJECTED`, `TIMEOUT`,
  `CONNECTION_ERROR`, `UNSUPPORTED`, `INVALID_TARGET`, `CANCELLED`, `UNKNOWN`) there are 5
  safety-guard categories: `JOINT_LIMIT_REJECTED`, `IK_QUALITY_REJECTED`,
  `SELF_COLLISION_REJECTED`, `PAYLOAD_REJECTED`, `CONTINUITY_REJECTED`, produced by
  `safety.SafetyPreflight`. (The previous README undercounted this.)
- **`MotionResult.from_bool` default is `CONTROLLER_REJECTED`.** A bare `False` with no richer
  cause is mislabeled as a controller refusal even when the real cause was a workspace/IK
  rejection — callers must pass `failure_status=` explicitly to get an accurate category. Known
  mislabel; flagged in the project register, not yet fixed (needs the bool callers migrated).
- **Force/impedance control is an unbuilt slot.** `RobotCapabilities.has_force_control` exists and
  defaults `False`; there is **no method on the `RobotArm` Protocol for force/admittance control**.
  Reserved for a future force-control surface.
- **No async / streaming / multi-arm contract.** `move*` are blocking. `supports_async_move` is a
  flag with no Protocol method behind it.
- **Frame/error coupling is partial.** `move_linear`/`ik`/`move_to`/`is_inside_workspace` are
  documented to require `Frame.BASE` (and `move_linear`/`ik` to raise `FrameMismatchError`), but
  enforcement lives in each driver — `core` only states the contract.
- **Driver maturity (advertised via the enums but lives in `drivers/`, not here):** `ur` and `sim`
  are the exercised backends; `sim` (Isaac) is validated for M1 known-pose pick 10/10, M2
  real-vision pick ~8/10, eye-in-hand wrist-camera pick 10/10. `kuka` is unvalidated on real
  hardware. `franka` and `ros2` are **empty driver slots** despite being enum members.
  `GripperVendor.FRANKA_HAND` / `SCHUNK` are reserved (no driver); `dummy`/`none` are the
  test/no-op paths. `core` itself defines none of these — it only names them.
- **No web/server/ROS-node code here**, and `fastapi`/`uvicorn` are dead deps elsewhere in the
  repo — neither touches this package.

## CLI / examples

No `python -m` entry point — this is a pure contract library. There is no dedicated
`tests/test_core_*` module; the contracts are exercised indirectly by the driver and pipeline
suites (e.g. `tests/test_motion_result.py`, `tests/test_robot_boundaries.py`,
`tests/test_sim_driver_skeleton.py`, `tests/test_runtime_pick.py`).
