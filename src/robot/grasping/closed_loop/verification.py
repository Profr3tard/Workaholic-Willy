"""Post-grasp verification.

Defines the typed contract and default implementations for verifying that an
executed grasp acquired an object.

Verification runs after ``GraspExecutionPolicy.execute`` succeeds. The
service provides the executed grasp, gripper state, jaw-width snapshots,
optional post-lift perception, and target identity to a ``GraspVerifier``,
which returns a ``GraspVerificationReport``. Failed verification, and
inconclusive verification when ``fail_closed`` is enabled, maps to
``AutonomousGraspOutcome.VERIFICATION_FAILED``.

This module performs no motion and preserves the existing
``ObjectDetectingGripper`` close-time check as the legacy trust path.
Verifiers are stateless and do not reimplement vendor-specific detection.

Public API:
    VerificationOutcome, GraspVerificationPolicy,
    GraspVerificationContext, GraspVerificationReport, GraspVerifier,
    NoOpVerifier, ObjectDetectingGripperVerifier,
    WidthDeltaGripperVerifier, VisionTargetDisplacementVerifier,
    CompositeGraspVerifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Mapping, Optional, Protocol, runtime_checkable

from src.robot.core import Gripper, ObjectDetectingGripper
from src.robot.grasping.types.grasp_point import GraspPoint
from src.robot.grasping.closed_loop.refinement import (
    IoUCentroidTargetTracker,
    TargetIdentity,
    TargetTracker,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.robot.grasping.types.perception import PerceptionFrame


__all__ = [
    "CompositeGraspVerifier",
    "GraspVerificationContext",
    "GraspVerificationPolicy",
    "GraspVerificationReport",
    "GraspVerifier",
    "NoOpVerifier",
    "ObjectDetectingGripperVerifier",
    "VerificationOutcome",
    "VisionTargetDisplacementVerifier",
    "WidthDeltaGripperVerifier",
]


class VerificationOutcome(StrEnum):
    """Terminal status of a single :meth:`GraspVerifier.verify` call.

    The four-value taxonomy is deliberate:

    * :attr:`PASSED` the verifier observed positive evidence the
      grasp succeeded.
    * :attr:`FAILED` the verifier observed positive evidence the
      grasp failed .
    * :attr:`INCONCLUSIVE` the verifier could not collect the
      evidence it needs to decide 
    * :attr:`SKIPPED` the policy itself is disabled.
    """

    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class GraspVerificationPolicy:
    """Bounded operator configuration for post-grasp verification.

    Provides conservative defaults for typical bin-picking with a parallel-jaw
    gripper. Inconclusive verification remains explicit; ``fail_closed`` controls
    whether it is treated as a verification failure.

    Attributes:
        enabled:
            Master switch. If ``False``, verification is skipped and the legacy
            ``ObjectDetectingGripper`` trust path is used.
        require_object_detected:
            Whether a reported ``is_object_detected() == False`` is a failure.
            If ``False``, the result is ``INCONCLUSIVE`` instead.
        width_delta_min_mm:
            Minimum jaw width above the gripper minimum opening. Defaults to
            ``2.0`` mm; smaller values indicate an empty grasp.
        width_delta_max_mm:
            Optional maximum jaw width above the commanded close width.
            ``None`` disables the upper-bound check.
        post_lift_vision_check:
            Whether to acquire a post-lift perception frame for vision-based
            verification.
        vision_displacement_iou_max:
            Maximum allowed IoU between the original target and a post-lift
            candidate. Defaults to ``0.2``.
        fail_closed:
            If ``True`` (default), ``INCONCLUSIVE`` is mapped to verification
            failure. If ``False``, inconclusive picks may still succeed.
    """

    enabled: bool = False
    require_object_detected: bool = False
    width_delta_min_mm: float = 2.0
    width_delta_max_mm: Optional[float] = None
    post_lift_vision_check: bool = False
    vision_displacement_iou_max: float = 0.2
    fail_closed: bool = True

    def __post_init__(self) -> None:
        if self.width_delta_min_mm < 0.0:
            raise ValueError(
                "width_delta_min_mm must be non-negative; "
                f"got {self.width_delta_min_mm}"
            )
        if (
            self.width_delta_max_mm is not None
            and self.width_delta_max_mm < 0.0
        ):
            raise ValueError(
                "width_delta_max_mm must be non-negative when set; "
                f"got {self.width_delta_max_mm}"
            )
        if not 0.0 <= self.vision_displacement_iou_max <= 1.0:
            raise ValueError(
                "vision_displacement_iou_max must be in [0, 1]; "
                f"got {self.vision_displacement_iou_max}"
            )


@dataclass(frozen=True, slots=True)
class GraspVerificationContext:
    """Frozen aggregate of everything a verifier needs."""

    grasp: GraspPoint
    policy: GraspVerificationPolicy
    gripper: Optional[Gripper] = None
    pre_close_width_mm: Optional[float] = None
    post_close_width_mm: Optional[float] = None
    commanded_close_width_mm: Optional[float] = None
    post_lift_frame: Optional["PerceptionFrame"] = None
    target_identity: Optional[TargetIdentity] = None


@dataclass(frozen=True, slots=True)
class GraspVerificationReport:
    """Frozen aggregate result of a single verifier call.

    Attributes
    ----------
    outcome
        Terminal :class:`VerificationOutcome`.
    reason
        Short machine-readable string explaining the decision.
    telemetry
        Free-form key/value bag.
    """

    outcome: VerificationOutcome
    reason: str = ""
    telemetry: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class GraspVerifier(Protocol):
    """Vendor-neutral post-grasp verifier."""

    def verify(
        self, context: GraspVerificationContext
    ) -> GraspVerificationReport:
        ...


@dataclass(frozen=True, slots=True)
class NoOpVerifier:
    """Always reports :attr:`VerificationOutcome.PASSED`."""

    def verify(
        self, context: GraspVerificationContext
    ) -> GraspVerificationReport:
        return GraspVerificationReport(
            outcome=VerificationOutcome.PASSED,
            reason="noop",
            telemetry={"verifier": "noop"},
        )


@dataclass(frozen=True, slots=True)
class ObjectDetectingGripperVerifier:
    """Delegate to :meth:`ObjectDetectingGripper.is_object_detected`.

    * Gripper missing / does not advertise the capability ->
      :attr:`VerificationOutcome.INCONCLUSIVE`.
    * Capability returns :data:`True` -> :attr:`VerificationOutcome.PASSED`.
    * Capability returns :data:`False`:
      * with :attr:`GraspVerificationPolicy.require_object_detected` ->
        :attr:`VerificationOutcome.FAILED`,
      * without it -> :attr:`VerificationOutcome.INCONCLUSIVE` so the
        operator can pair this verifier with a vision check that has
        the final say.
    """

    def verify(
        self, context: GraspVerificationContext
    ) -> GraspVerificationReport:
        gripper = context.gripper
        if gripper is None or not isinstance(gripper, ObjectDetectingGripper):
            return GraspVerificationReport(
                outcome=VerificationOutcome.INCONCLUSIVE,
                reason="no_object_detection_capability",
                telemetry={"verifier": "object_detecting_gripper"},
            )
        try:
            detected = bool(gripper.is_object_detected())
        except Exception as exc:  # noqa: BLE001 - keep verifier total
            return GraspVerificationReport(
                outcome=VerificationOutcome.INCONCLUSIVE,
                reason="object_detection_query_raised",
                telemetry={
                    "verifier": "object_detecting_gripper",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
        if detected:
            return GraspVerificationReport(
                outcome=VerificationOutcome.PASSED,
                reason="gripper_object_detected",
                telemetry={
                    "verifier": "object_detecting_gripper",
                    "object_detected": True,
                },
            )
        if context.policy.require_object_detected:
            return GraspVerificationReport(
                outcome=VerificationOutcome.FAILED,
                reason="gripper_object_not_detected",
                telemetry={
                    "verifier": "object_detecting_gripper",
                    "object_detected": False,
                },
            )
        return GraspVerificationReport(
            outcome=VerificationOutcome.INCONCLUSIVE,
            reason="gripper_reports_empty_but_policy_does_not_require",
            telemetry={
                "verifier": "object_detecting_gripper",
                "object_detected": False,
            },
        )


@dataclass(frozen=True, slots=True)
class WidthDeltaGripperVerifier:
    """Verify a grasp by inspecting the post-close jaw width.

    Inputs required:

    * a :class:`Gripper` on the context (to read ``min_width_mm``),
    * :attr:`GraspVerificationContext.post_close_width_mm`.

    Either of those missing -> :attr:`VerificationOutcome.INCONCLUSIVE`.
    """

    def verify(
        self, context: GraspVerificationContext
    ) -> GraspVerificationReport:
        gripper = context.gripper
        if gripper is None:
            return GraspVerificationReport(
                outcome=VerificationOutcome.INCONCLUSIVE,
                reason="no_gripper",
                telemetry={"verifier": "width_delta"},
            )
        if context.post_close_width_mm is None:
            return GraspVerificationReport(
                outcome=VerificationOutcome.INCONCLUSIVE,
                reason="no_post_close_width_sample",
                telemetry={"verifier": "width_delta"},
            )
        # The PHYSICAL closed width, not the policy floor.
        _closed = getattr(gripper, "closed_width_mm", None)
        min_width = float(_closed) if _closed is not None else float(
            getattr(gripper, "min_width_mm", 0.0) or 0.0
        )
        post = float(context.post_close_width_mm)
        min_engaged_width = min_width + float(
            context.policy.width_delta_min_mm
        )
        telemetry: dict[str, Any] = {
            "verifier": "width_delta",
            "post_close_width_mm": post,
            "min_width_mm": min_width,
            "min_engaged_width_mm": min_engaged_width,
        }
        if post <= min_engaged_width:
            return GraspVerificationReport(
                outcome=VerificationOutcome.FAILED,
                reason="jaws_collapsed_to_minimum",
                telemetry=telemetry,
            )
        upper_bound = context.policy.width_delta_max_mm
        commanded = context.commanded_close_width_mm
        if upper_bound is not None and commanded is not None:
            ceiling = float(commanded) + float(upper_bound)
            telemetry["upper_bound_mm"] = ceiling
            if post > ceiling:
                return GraspVerificationReport(
                    outcome=VerificationOutcome.FAILED,
                    reason="jaws_did_not_close_enough",
                    telemetry=telemetry,
                )
        return GraspVerificationReport(
            outcome=VerificationOutcome.PASSED,
            reason="width_within_bounds",
            telemetry=telemetry,
        )


@dataclass(frozen=True, slots=True)
class VisionTargetDisplacementVerifier:
    """Verify a grasp by checking the target left the workspace."""

    tracker: TargetTracker = field(default_factory=IoUCentroidTargetTracker)

    def verify(
        self, context: GraspVerificationContext
    ) -> GraspVerificationReport:
        frame = context.post_lift_frame
        identity = context.target_identity
        if frame is None or identity is None:
            return GraspVerificationReport(
                outcome=VerificationOutcome.INCONCLUSIVE,
                reason="no_post_lift_frame_or_target_identity",
                telemetry={"verifier": "vision_target_displacement"},
            )
        if not frame.segmentations:
            # No candidates at all -> target gone.
            return GraspVerificationReport(
                outcome=VerificationOutcome.PASSED,
                reason="post_lift_frame_has_no_segmentations",
                telemetry={
                    "verifier": "vision_target_displacement",
                    "match_iou": 0.0,
                },
            )
        # We *want* the tracker to fail to find the target.
        match = self.tracker.match(
            identity,
            frame,
            iou_threshold=context.policy.vision_displacement_iou_max,
        )
        if match is None:
            return GraspVerificationReport(
                outcome=VerificationOutcome.PASSED,
                reason="target_no_longer_visible",
                telemetry={
                    "verifier": "vision_target_displacement",
                    "iou_threshold": context.policy.vision_displacement_iou_max,
                },
            )
        idx, iou = match
        return GraspVerificationReport(
            outcome=VerificationOutcome.FAILED,
            reason="target_still_visible",
            telemetry={
                "verifier": "vision_target_displacement",
                "match_segmentation_index": int(idx),
                "match_iou": float(iou),
                "iou_threshold": context.policy.vision_displacement_iou_max,
            },
        )


@dataclass(frozen=True, slots=True)
class CompositeGraspVerifier:
    """Combine several verifiers under a typed aggregation rule.

    Aggregation rules
    -----------------
    * ``"all_must_pass"`` (default): the composite reports
      :attr:`VerificationOutcome.PASSED` only when every child returns
      :attr:`VerificationOutcome.PASSED`. The first
      :attr:`VerificationOutcome.FAILED` short-circuits the chain.
      :attr:`VerificationOutcome.INCONCLUSIVE` children are treated as
      passes unless :attr:`require_all_conclusive` is :data:`True`, in
      which case they are treated as failures.
    * ``"any_pass"``: PASSED is reported as soon as any child returns
      :attr:`VerificationOutcome.PASSED`. If every child is
      :attr:`VerificationOutcome.INCONCLUSIVE` the composite reports
      :attr:`VerificationOutcome.INCONCLUSIVE`; otherwise it reports
      :attr:`VerificationOutcome.FAILED`.
    """

    verifiers: tuple[GraspVerifier, ...]
    rule: str = "all_must_pass"
    require_all_conclusive: bool = False

    def __post_init__(self) -> None:
        if not self.verifiers:
            raise ValueError(
                "CompositeGraspVerifier requires at least one child verifier"
            )
        if self.rule not in {"all_must_pass", "any_pass"}:
            raise ValueError(
                "CompositeGraspVerifier.rule must be 'all_must_pass' or "
                f"'any_pass'; got {self.rule!r}"
            )

    def verify(
        self, context: GraspVerificationContext
    ) -> GraspVerificationReport:
        child_reports: list[GraspVerificationReport] = [
            child.verify(context) for child in self.verifiers
        ]
        child_telemetry = tuple(
            {
                "outcome": str(r.outcome),
                "reason": r.reason,
                "telemetry": dict(r.telemetry),
            }
            for r in child_reports
        )
        if self.rule == "all_must_pass":
            for report in child_reports:
                if report.outcome is VerificationOutcome.FAILED:
                    return GraspVerificationReport(
                        outcome=VerificationOutcome.FAILED,
                        reason=f"child_failed:{report.reason}",
                        telemetry={
                            "verifier": "composite",
                            "rule": self.rule,
                            "children": child_telemetry,
                        },
                    )
                if (
                    self.require_all_conclusive
                    and report.outcome is VerificationOutcome.INCONCLUSIVE
                ):
                    return GraspVerificationReport(
                        outcome=VerificationOutcome.FAILED,
                        reason=f"child_inconclusive:{report.reason}",
                        telemetry={
                            "verifier": "composite",
                            "rule": self.rule,
                            "children": child_telemetry,
                        },
                    )
            return GraspVerificationReport(
                outcome=VerificationOutcome.PASSED,
                reason="all_children_passed_or_inconclusive",
                telemetry={
                    "verifier": "composite",
                    "rule": self.rule,
                    "children": child_telemetry,
                },
            )
        # "any_pass"
        any_pass = any(
            r.outcome is VerificationOutcome.PASSED for r in child_reports
        )
        if any_pass:
            return GraspVerificationReport(
                outcome=VerificationOutcome.PASSED,
                reason="at_least_one_child_passed",
                telemetry={
                    "verifier": "composite",
                    "rule": self.rule,
                    "children": child_telemetry,
                },
            )
        all_inconclusive = all(
            r.outcome is VerificationOutcome.INCONCLUSIVE
            for r in child_reports
        )
        return GraspVerificationReport(
            outcome=(
                VerificationOutcome.INCONCLUSIVE
                if all_inconclusive
                else VerificationOutcome.FAILED
            ),
            reason=(
                "all_children_inconclusive"
                if all_inconclusive
                else "no_child_passed"
            ),
            telemetry={
                "verifier": "composite",
                "rule": self.rule,
                "children": child_telemetry,
            },
        )
