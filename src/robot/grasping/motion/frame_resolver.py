"""Typed contract for resolving the current camera-to-base transform.

Provides a vendor-neutral ``FrameResolver`` used to obtain
``T_cam_to_base`` at perception time, preventing stale or missing transforms
from being used for grasp execution. Supports fail-closed operation when no
valid transform is available and avoids dependencies on hardware drivers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Union, runtime_checkable

import numpy as np

from src.geometry import Frame, Pose, Transform
from src.robot.core import RobotArm
from src.robot.grasping.types.perception import PerceptionFrame

__all__ = [
    "EyeInHandFrameResolver",
    "FrameResolutionFailure",
    "FrameResolver",
    "IdentityFrameResolver",
    "RESOLVE_REASON_BAD_TRANSFORM",
    "RESOLVE_REASON_EXCEPTION",
    "RESOLVE_REASON_NONE_RETURNED",
    "RESOLVE_REASON_NO_RESOLVER",
    "RESOLVE_REASON_WRONG_FRAME",
    "StaticCameraToBaseResolver",
    "resolve_or_none",
]


@runtime_checkable
class FrameResolver(Protocol):
    """Resolve ``T_cam_to_base`` for a given :class:`PerceptionFrame`."""

    def camera_to_base_for_frame(
        self,
        frame: PerceptionFrame,
        *,
        arm: RobotArm,
    ) -> Transform: ...


# ---------------------------------------------------------------------------
# Built-in resolvers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StaticCameraToBaseResolver:
    """Eye-to-hand resolver: a fixed ``T_cam_to_base``."""

    transform: Transform

    def __post_init__(self) -> None:
        if (
            self.transform.from_frame is not Frame.CAMERA
            or self.transform.to_frame is not Frame.BASE
        ):
            raise ValueError(
                "StaticCameraToBaseResolver.transform must be "
                "Transform(CAMERA -> BASE); got "
                f"{self.transform.from_frame.name} -> "
                f"{self.transform.to_frame.name}"
            )

    def camera_to_base_for_frame(
        self,
        frame: PerceptionFrame,  # noqa: ARG002 - unused for eye-to-hand
        *,
        arm: RobotArm,  # noqa: ARG002 - unused for eye-to-hand
    ) -> Transform:
        return self.transform


@dataclass(frozen=True, slots=True)
class EyeInHandFrameResolver:
    """Eye-in-hand resolver: compose live TCP pose with camera-on-tool.

    The camera is rigidly mounted to the flange/TCP. The static
    calibration ``t_cam_to_tool`` (``CAMERA -> TOOL``) is supplied at
    construction. At resolution time the resolver:

    1. reads the current TCP pose from the arm
       (:meth:`RobotArm.get_tcp_pose`),
    2. reinterprets that pose as ``T_tool_to_base`` (the TCP pose in
       base-frame *is* the rigid transform mapping tool-frame data
       into the base frame),
    3. composes ``T_cam_to_tool @ T_tool_to_base`` to obtain the live
       ``T_cam_to_base``.

    Failure modes are explicit:

    * Wrong calibration frames raise :class:`ValueError` at
      construction.
    * A TCP pose that is not in :attr:`Frame.BASE` raises
      :class:`ValueError` at resolution. This catches the common
      footgun of a driver returning a tool-frame TCP read.
    """

    t_cam_to_tool: Transform

    def __post_init__(self) -> None:
        if (
            self.t_cam_to_tool.from_frame is not Frame.CAMERA
            or self.t_cam_to_tool.to_frame is not Frame.TOOL
        ):
            raise ValueError(
                "EyeInHandFrameResolver.t_cam_to_tool must be "
                "Transform(CAMERA -> TOOL); got "
                f"{self.t_cam_to_tool.from_frame.name} -> "
                f"{self.t_cam_to_tool.to_frame.name}"
            )

    def camera_to_base_for_frame(
        self,
        frame: PerceptionFrame,  # noqa: ARG002 - frame not needed for the transform itself
        *,
        arm: RobotArm,
    ) -> Transform:
        tcp_pose = arm.get_tcp_pose()
        if not isinstance(tcp_pose, Pose):
            raise TypeError(
                "RobotArm.get_tcp_pose() must return a Pose; got "
                f"{type(tcp_pose).__name__}"
            )
        if tcp_pose.frame is not Frame.BASE:
            raise ValueError(
                "EyeInHandFrameResolver requires arm.get_tcp_pose() in "
                f"Frame.BASE; got {tcp_pose.frame.name}. This usually "
                "means the driver returned a tool-frame reading by "
                "mistake."
            )
        # The TCP pose in base coordinates is a rigid transform that
        # maps tool-frame data into base-frame data: T_tool_to_base.
        t_tool_to_base = Transform.from_matrix(
            np.asarray(tcp_pose.to_matrix(), dtype=np.float64),
            from_frame=Frame.TOOL,
            to_frame=Frame.BASE,
        )
        # Transform.compose semantics: ``a.compose(b)`` returns a
        # transform from ``a.from_frame`` to ``b.to_frame`` i.e.
        # apply ``a`` first, then ``b``. So composing
        # ``T_cam_to_tool`` then ``T_tool_to_base`` yields the
        # desired ``T_cam_to_base``.
        return self.t_cam_to_tool.compose(t_tool_to_base)


@dataclass(frozen=True, slots=True)
class IdentityFrameResolver:
    """Explicit "camera frame == base frame" resolver."""

    def camera_to_base_for_frame(
        self,
        frame: PerceptionFrame,  # noqa: ARG002 - unused
        *,
        arm: RobotArm,  # noqa: ARG002 - unused
    ) -> Transform:
        return Transform.identity(from_frame=Frame.CAMERA, to_frame=Frame.BASE)


# ---------------------------------------------------------------------------
# Typed convenience: non-raising resolution wrapper for shadow callers
# ---------------------------------------------------------------------------

# Reason strings frozen wire contract. Consumers (multi-view
# fusion, future telemetry) must match against these constants rather
# than the human-readable ``message``.
RESOLVE_REASON_NO_RESOLVER = "no_resolver"
RESOLVE_REASON_EXCEPTION = "exception"
RESOLVE_REASON_NONE_RETURNED = "none_returned"
RESOLVE_REASON_BAD_TRANSFORM = "bad_transform"
RESOLVE_REASON_WRONG_FRAME = "wrong_frame"


@dataclass(frozen=True, slots=True)
class FrameResolutionFailure:
    """Typed failure carrier returned by :func:`resolve_or_none`.

    ``reason`` is one of the ``RESOLVE_REASON_*`` module-level
    constants and is the stable wire field. ``message`` is a
    human-readable diagnostic intended only for logs / telemetry --
    do not branch on it.
    """

    reason: str
    message: str = ""


def resolve_or_none(
    resolver: Optional[FrameResolver],
    frame: PerceptionFrame,
    *,
    arm: RobotArm,
) -> Union[Transform, FrameResolutionFailure]:
    """Best-effort, non-raising wrapper around :meth:`FrameResolver.camera_to_base_for_frame`.

    Returns either:

    * a :class:`Transform` (``CAMERA -> BASE``) on success, or
    * a :class:`FrameResolutionFailure` describing why no transform
      could be produced.

    Critically, this helper never raises.
    """

    if resolver is None:
        return FrameResolutionFailure(
            reason=RESOLVE_REASON_NO_RESOLVER,
            message="no frame resolver wired",
        )
    try:
        t = resolver.camera_to_base_for_frame(frame, arm=arm)
    except Exception as exc:  # noqa: BLE001 - intentional broad guard for shadow paths
        return FrameResolutionFailure(
            reason=RESOLVE_REASON_EXCEPTION,
            message=f"{type(exc).__name__}: {exc}",
        )
    if t is None:
        return FrameResolutionFailure(
            reason=RESOLVE_REASON_NONE_RETURNED,
            message="resolver returned None",
        )
    if not isinstance(t, Transform):
        return FrameResolutionFailure(
            reason=RESOLVE_REASON_BAD_TRANSFORM,
            message=f"resolver returned {type(t).__name__}, expected Transform",
        )
    if t.from_frame is not Frame.CAMERA or t.to_frame is not Frame.BASE:
        return FrameResolutionFailure(
            reason=RESOLVE_REASON_WRONG_FRAME,
            message=(
                f"resolver returned Transform({t.from_frame.name} -> "
                f"{t.to_frame.name}); expected CAMERA -> BASE"
            ),
        )
    return t
