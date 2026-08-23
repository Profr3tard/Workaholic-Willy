"""Robot calibration event protocol."""

from __future__ import annotations

from typing import Any, Final, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Canonical event-type names
# ---------------------------------------------------------------------------

class RobotCalibrationEvent:
    """Namespace of canonical event-type strings for hand-eye calibration."""

    MOVING_TO_POSE: Final[str] = "robot_calibration_moving_to_pose"
    POSE_REJECTED: Final[str] = "robot_calibration_pose_rejected"
    MARKER_DETECTED: Final[str] = "robot_calibration_marker_detected"
    POSE_ACCEPTED: Final[str] = "robot_calibration_pose_accepted"

    #: Tuple of every canonical event name, handy for tests / docs.
    ALL: Final[tuple[str, ...]] = (
        MOVING_TO_POSE,
        POSE_REJECTED,
        MARKER_DETECTED,
        POSE_ACCEPTED,
    )


# ---------------------------------------------------------------------------
# Listener protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class RobotCalibrationEventListener(Protocol):
    """Callable signature accepted by :class:`CalibrationRoutine`."""

    def __call__(self, event_type: str, data: dict[str, Any]) -> None: ...


# ---------------------------------------------------------------------------
# Drift + OOD watchdog event surface
# ---------------------------------------------------------------------------

class RobotWatchdogEvent:
    """Namespace of canonical event-type strings for the watchdog.

    Events are emitted by :class:`AutonomousGraspService` when the
    watchdog's per-tick state transitions across one of the following
    edges:

    * ``DRIFT_DETECTED`` drift_severity escalated from NONE/LOW to
      MODERATE-or-higher.
    * ``OOD_DETECTED`` ood_flagged transitioned from False to True.
    * ``DEGRADED_MODE_ENGAGED`` degraded_mode_active transitioned
      from False to True.
    * ``BLOCK_AUTO_TRIGGERED`` an enforced BLOCK_AUTO action fired
      and the service short-circuited the decision loop.
    * ``SLO_BREACH`` the runtime's rolling p95 for one
      of the locked latency stages (decision / ranking / fusion)
      crossed its configured SLO budget.

    Emission is best-effort and never blocks the grasp loop. Listener
    exceptions are swallowed so a buggy listener cannot abort a pick.
    """

    DRIFT_DETECTED: Final[str] = "robot_watchdog_drift_detected"
    OOD_DETECTED: Final[str] = "robot_watchdog_ood_detected"
    DEGRADED_MODE_ENGAGED: Final[str] = "robot_watchdog_degraded_mode_engaged"
    BLOCK_AUTO_TRIGGERED: Final[str] = "robot_watchdog_block_auto_triggered"
    SLO_BREACH: Final[str] = "robot_watchdog_slo_breach"

    #: Tuple of every canonical event name, handy for tests / docs.
    ALL: Final[tuple[str, ...]] = (
        DRIFT_DETECTED,
        OOD_DETECTED,
        DEGRADED_MODE_ENGAGED,
        BLOCK_AUTO_TRIGGERED,
        SLO_BREACH,
    )


@runtime_checkable
class RobotWatchdogEventListener(Protocol):
    """Callable signature accepted by :class:`AutonomousGraspService`."""

    def __call__(self, event_type: str, data: dict[str, Any]) -> None: ...
