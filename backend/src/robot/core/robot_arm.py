"""
:class:`RobotArm` — vendor-neutral abstract surface for a manipulator.

Every driver in :mod:`backend.src.robot.drivers` (UR, KUKA, Franka, ROS 2,
sim, dummy, …) MUST implement this Protocol. Pipelines and planners
depend only on this interface; they never import a driver class
directly.

Numerics contract
-----------------
* All :class:`Pose` arguments / returns live in millimetres + canonical
  XYZW quaternions, frame-tagged. ``get_tcp_pose()`` returns a Pose with
  :attr:`Frame.BASE`.
* Joint positions are :class:`JointPositions` (radians, ``float64``).
* Velocity / acceleration units follow the driver's natural convention
  (UR uses rad/s and rad/s² for joint moves, m/s and m/s² for Cartesian
  moves) — pass through unchanged. Higher-level planners are expected to
  consult :attr:`RobotArm.capabilities` for unit details when they need
  them.

The Protocol is :func:`runtime_checkable` so tests can verify a
candidate driver via ``isinstance(driver, RobotArm)`` without needing
nominal subclassing — but every real driver SHOULD still inherit from
the Protocol explicitly to get static-type help.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from backend.src.geometry import Pose

from .capabilities import RobotCapabilities
from .joint_positions import JointPositions
from .motion_result import MotionResult

__all__ = ["RobotArm"]


@runtime_checkable
class RobotArm(Protocol):
    """Vendor-neutral arm interface."""

    # ---- introspection --------------------------------------------------

    @property
    def capabilities(self) -> RobotCapabilities:
        """Static feature flags for this driver."""
        ...

    @property
    def is_connected(self) -> bool:
        """``True`` while a live link to the controller is open."""
        ...

    # ---- lifecycle ------------------------------------------------------

    def connect(self) -> None:
        """Open the link to the controller. Idempotent."""
        ...

    def disconnect(self) -> None:
        """Close the link. Idempotent — safe to call multiple times."""
        ...

    # ---- state ----------------------------------------------------------

    def get_tcp_pose(self) -> Pose:
        """Current TCP pose, tagged ``Frame.BASE``."""
        ...

    def get_joint_positions(self) -> JointPositions:
        """Current joint configuration."""
        ...

    # ---- motion ---------------------------------------------------------

    def move_joint(
        self,
        joints: JointPositions,
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
    ) -> None:
        """
        Joint-space move to ``joints``.

        Raises
        ------
        RobotMotionRejected
            If a workspace / safety pre-check denies the move.
        RobotConnectionError
            If the link is not open.
        """
        ...

    def move_to_joints(
        self,
        joints: JointPositions,
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
    ) -> MotionResult:
        """Typed joint-space move — the fail-closed counterpart of :meth:`move_joint`.

        Routes the commanded joints through the ``SafetyPreflight`` DESTINATION guards (joint-limit /
        self-collision / payload / static IK-quality incl. singularity) and then drives. A commanded
        joint move (park / home / scan pose) is a deliberate trajectory RESTART, so the step-size checks
        (motion-continuity + the IK-jump check) and the pose-only workspace guard are exempted, and the
        continuity reference is reset around it. Unlike the void :meth:`move_joint`, a guard rejection is
        returned as a typed :class:`MotionResult` (the matching :class:`MotionStatus`) rather than raised.
        A driver with no preflight wired simply drives and returns ``EXECUTED``.
        """
        ...

    def move_linear(
        self,
        pose: Pose,
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
    ) -> None:
        """
        Cartesian move to ``pose`` (must be in :attr:`Frame.BASE`).

        Raises
        ------
        FrameMismatchError
            If ``pose.frame`` is not :attr:`Frame.BASE`.
        RobotMotionRejected
            If a workspace / safety pre-check denies the move.
        RobotKinematicsError
            If no IK solution exists.
        """
        ...

    def stop(self) -> None:
        """Emergency stop. Always safe to call."""
        ...

    # ---- kinematics -----------------------------------------------------

    def fk(self, joints: JointPositions) -> Pose:
        """Forward kinematics. Returns a TCP :class:`Pose` in :attr:`Frame.BASE`."""
        ...

    def ik(
        self,
        pose: Pose,
        *,
        seed: JointPositions | None = None,
    ) -> JointPositions:
        """
        Inverse kinematics.

        Parameters
        ----------
        pose
            Target TCP pose; must be in :attr:`Frame.BASE`.
        seed
            Optional joint seed for nearest-solution selection.

        Raises
        ------
        FrameMismatchError
            If ``pose.frame`` is not :attr:`Frame.BASE`.
        RobotKinematicsError
            If no IK solution exists.
        """
        ...

    # ---- high-level pipeline surface (vendor-neutral) -------------------
    #
    # These helpers exist so pipelines and the API layer never need to
    # touch vendor-specific symbols (URPose, the workspace ``guard``
    # attribute, …). Every driver is expected to implement them with
    # *bool* semantics — pre-check first, return ``False`` instead of
    # raising — so pipelines can keep their try-and-retry control flow.

    def is_inside_workspace(self, pose: Pose) -> bool:
        """
        Pre-flight check: would ``pose`` be accepted by the driver's
        workspace / safety policy?

        ``pose.frame`` must be :attr:`Frame.BASE`. Drivers that have no
        configured workspace MUST return ``True``.
        """
        ...

    def move_to(
        self,
        pose: Pose,
        *,
        linear: bool = False,
        vel: float | None = None,
        acc: float | None = None,
        register: bool = True,
    ) -> bool:
        """
        High-level "go there" command for pipelines.

        Returns ``True`` on success, ``False`` if the move was rejected
        by a workspace / safety check or by the controller. Drivers MUST
        NOT raise for the ordinary "out of workspace" / "IK failed"
        cases — those are signalled by the bool return — but MAY raise
        for connection errors.

        ``linear=True`` requests a Cartesian (straight-line) move when
        the driver supports it; otherwise drivers fall back to a
        joint-space move.

        ``register=False`` tells the driver **not** to add ``pose`` to
        any internal diversity / sampling history. Calibration routines
        and other per-pose orchestrators that maintain their own
        bookkeeping should pass ``register=False`` so the driver's
        long-running guard does not double-count.
        """
        ...

    def move_home(self) -> bool:
        """
        Move to the driver's configured home configuration.

        Returns ``True`` on success, ``False`` on rejection.
        """
        ...

    def wait_until_steady(
        self,
        timeout_s: float = 5.0,
        poll_interval_s: float = 0.02,
    ) -> bool:
        """
        Block until the controller reports the arm has come to rest.

        Returns ``True`` if the arm settled within ``timeout_s``,
        ``False`` on timeout. Drivers without an explicit "steady"
        signal MAY implement this as ``time.sleep(timeout_s); return True``
        --- the exact semantics are vendor-specific. Calibration
        routines call this between each commanded move and the next
        capture.
        """
        ...

    # ---- typed motion surface -------------------------------------------
    #
    # The bool-returning :meth:`move_to` / :meth:`move_home` collapse
    # every failure into a single ``False``, which leaves orchestrators
    # and calibration unable to distinguish a workspace rejection from
    # an IK failure, a controller refusal, or a connection fault.
    # :meth:`move` is the typed surface that replaces that pattern; the
    # bool methods remain as compatibility shims and are being migrated off.

    def move(
        self,
        pose: Pose,
        *,
        linear: bool = False,
        vel: float | None = None,
        acc: float | None = None,
        register: bool = True,
    ) -> MotionResult:
        """
        Typed "go there" command for runtime orchestration.

        Returns a :class:`MotionResult` that classifies the outcome
        into one of :class:`MotionStatus` so callers can react without
        log scraping. Drivers MUST NOT raise for the ordinary
        workspace / IK / controller-refusal cases — those are signalled
        through :attr:`MotionResult.status`. Drivers MAY raise on
        connection / transport faults, but SHOULD prefer to return
        :attr:`MotionStatus.CONNECTION_ERROR` so callers can stay on
        the typed code path.

        ``register=False`` tells the driver **not** to add ``pose`` to
        any internal diversity / sampling history — see
        :meth:`move_to` for the semantics.
        """
        ...
