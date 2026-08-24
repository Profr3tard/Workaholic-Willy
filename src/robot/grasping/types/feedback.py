"""Structured, vendor-neutral feedback for grasp calculation.

Wraps the existing ``list[GraspPoint]`` result with a typed reason for
empty candidate lists, enabling retries, fallbacks, operator feedback,
and stage-level telemetry without breaking the historical API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from src.robot.grasping.types.grasp_point import GraspPoint

__all__ = [
    "GraspFailureReason",
    "GraspResult",
]


class GraspFailureReason(StrEnum):
    """Reason codes explaining why the grasp pipeline yielded no result."""

    EMPTY_MASK = "empty_mask"
    MASK_TOO_SMALL = "mask_too_small"
    NO_VALID_DEPTH = "no_valid_depth"
    LOW_DEPTH_CONFIDENCE = "low_depth_confidence"
    LOW_MASK_CONFIDENCE = "low_mask_confidence"
    NO_CANDIDATES_GENERATED = "no_candidates_generated"
    ALL_COLLIDED = "all_collided"
    ALL_OUT_OF_WORKSPACE = "all_out_of_workspace"
    ALL_TABLE_CONFLICT = "all_table_conflict"
    IK_FAILED = "ik_failed"
    NO_VALID_GRASP = "no_valid_grasp"
    RESCAN_RECOMMENDED = "rescan_recommended"
    TRY_NEXT_CANDIDATE = "try_next_candidate"
    ACTIVE_PERCEPTION_RECOMMENDED = "active_perception_recommended"
    # Mask topology too risky for a stable parallel-jaw grasp (rings,
    # U-shapes, handles).
    TOPOLOGY_RISK_REJECTED = "topology_risk_rejected"
    # Operator-supplied semantic policy denied this candidate (label
    # not in allow-list, in deny-list, or below confidence threshold).
    SEMANTIC_REJECTED = "semantic_rejected"
    # Target mask is heavily occluded by other detected objects: the
    # fraction of the target's convex hull covered by neighbouring
    # masks exceeded the configured ``heavy_occlusion_threshold``.
    HEAVY_OCCLUSION = "heavy_occlusion"
    # The target was classified as a non-rigid object (cable, cloth,
    # bag) and the configured ``DeformableHandlingStrategy`` refused
    # to plan a parallel-jaw grasp.
    DEFORMABLE_ROUTING_REQUIRED = "deformable_routing_required"
    # Two-scan pre-grasp refinement: the target the initial grasp was
    # computed on could not be re-identified in the refined frame
    # (no matching segmentation passed the tracker's threshold).
    TARGET_LOST_DURING_REFINE = "target_lost_during_refine"
    # Refinement produced a grasp whose position / orientation /
    # grip-width delta exceeded the configured correction bounds.
    REFINEMENT_DIVERGED = "refinement_diverged"
    # The motion planner refused to route to an otherwise valid grasp and
    # NOTHING MOVED (see ``NO_PLAN_FAIL_SAFE_MESSAGE``).
    MOTION_PLAN_REFUSED = "motion_plan_refused"
    # The CONTROLLER itself cannot move: protective stop, emergency stop, a safety-mode stop, or
    # simply powered off.
    CONTROLLER_NOT_OPERATIONAL = "controller_not_operational"
    # A hard target label was requested and perception returned segmentations, but NOT ONE of them
    # carried that label. Distinct from every reason above: nothing is wrong with the scene, the
    # grasp or the cell, the operator asked for an object this frame does not contain, or the
    # detector named it something else.
    TARGET_LABEL_NOT_FOUND = "target_label_not_found"


@dataclass(frozen=True, slots=True)
class GraspResult:
    """Structured result of a single ``GraspCalculator.compute`` call.

    Fields
    ------
    candidates
        Ranked grasp candidates (best first). Empty when no valid grasp
        was found; ``reasons`` will then be non-empty.
    reasons
        Ordered, deduplicated failure / recommendation codes. Empty for
        a successful call.
    telemetry
        Stage counters from the calculator. Same shape as
        ``GraspCalculator.last_telemetry``.
    depth_confidence
        Fraction of pixels under the mask with finite, positive depth.
        ``None`` if the calculator returned before this could be
        computed (e.g. empty mask).
    mask_confidence
        Optional segmentation score copied from the
        ``SegmentationResult.score`` boundary field when present.
    top_score
        Score of the best candidate, or ``0.0`` when empty.
    metadata
        Free-form extra data (debug image hash, calculator id, ...).
    """

    candidates: tuple[GraspPoint, ...] = ()
    reasons: tuple[GraspFailureReason, ...] = ()
    telemetry: dict[str, Any] = field(default_factory=dict)
    depth_confidence: float | None = None
    mask_confidence: float | None = None
    top_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """``True`` when at least one valid candidate is available."""
        return bool(self.candidates)

    @property
    def best(self) -> GraspPoint | None:
        """Highest-ranked candidate or ``None`` when the result is empty."""
        return self.candidates[0] if self.candidates else None

    def with_reasons(self, *reasons: GraspFailureReason) -> GraspResult:
        """Return a copy with additional reasons appended (deduplicated)."""
        seen: dict[GraspFailureReason, None] = dict.fromkeys(self.reasons)
        for reason in reasons:
            seen[reason] = None
        return GraspResult(
            candidates=self.candidates,
            reasons=tuple(seen.keys()),
            telemetry=self.telemetry,
            depth_confidence=self.depth_confidence,
            mask_confidence=self.mask_confidence,
            top_score=self.top_score,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dictionary."""
        return {
            "candidates": [grasp.to_dict() for grasp in self.candidates],
            "reasons": [reason.value for reason in self.reasons],
            "telemetry": dict(self.telemetry),
            "depth_confidence": self.depth_confidence,
            "mask_confidence": self.mask_confidence,
            "top_score": self.top_score,
            "metadata": dict(self.metadata),
            "is_success": self.is_success,
        }
