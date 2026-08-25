"""Typed, passive per-stage latency tracker for grasp pipeline SLO metrics.

Records wall-clock spans for decision, ranking, and fusion stages using
monotonic time. The tracker never raises, blocks, or performs I/O; snapshots
are JSON-safe and use the canonical telemetry field names. Unentered stages
are omitted, and repeated stages use the last recorded span.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Optional


__all__ = (
    "LatencyStage",
    "LatencySpan",
    "LatencyTracker",
    "STAGE_FIELD_NAMES",
)


class LatencyStage(str, Enum):
    """Canonical per-attempt stages tracked by the SLO gate."""

    DECISION = "decision"
    RANKING = "ranking"
    FUSION = "fusion"


#: Map each stage to its locked telemetry-catalog ``extra`` field name.
STAGE_FIELD_NAMES: dict[LatencyStage, str] = {
    LatencyStage.DECISION: "decision_latency_ms",
    LatencyStage.RANKING: "ranking_latency_ms",
    LatencyStage.FUSION: "fusion_latency_ms",
}


@dataclass(frozen=True, slots=True)
class LatencySpan:
    """A single closed span recorded by the tracker."""

    stage: LatencyStage
    elapsed_ms: float

    def __post_init__(self) -> None:
        if not isinstance(self.stage, LatencyStage):
            raise TypeError(
                f"LatencySpan.stage must be a LatencyStage; got "
                f"{type(self.stage).__name__}"
            )
        if not isinstance(self.elapsed_ms, (int, float)):
            raise TypeError(
                "LatencySpan.elapsed_ms must be a finite number"
            )
        value = float(self.elapsed_ms)
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("LatencySpan.elapsed_ms must be finite")
        if value < 0.0:
            raise ValueError("LatencySpan.elapsed_ms must be non-negative")
        # Normalise int → float so downstream JSON encoding is stable.
        object.__setattr__(self, "elapsed_ms", value)


@dataclass(slots=True)
class LatencyTracker:
    """Mutable accumulator of per-stage spans for a single attempt"""

    spans: dict[LatencyStage, LatencySpan] = field(default_factory=dict)

    @contextmanager
    def span(self, stage: LatencyStage) -> Iterator[None]:
        """Time a code block and record it under ``stage``."""

        if not isinstance(stage, LatencyStage):
            raise TypeError(
                "LatencyTracker.span requires a LatencyStage; "
                f"got {type(stage).__name__}"
            )
        start_ns = time.monotonic_ns()
        try:
            yield
        finally:
            elapsed_ns = time.monotonic_ns() - start_ns
            elapsed_ms = elapsed_ns / 1_000_000.0
            # Clamp to non-negative to guard against monotonic_ns
            # quirks on very fast spans (some platforms report 0).
            if elapsed_ms < 0.0:
                elapsed_ms = 0.0
            self.spans[stage] = LatencySpan(
                stage=stage, elapsed_ms=elapsed_ms
            )

    def record(self, stage: LatencyStage, elapsed_ms: float) -> None:
        """Record a span computed by the caller."""

        self.spans[stage] = LatencySpan(stage=stage, elapsed_ms=elapsed_ms)

    def get(self, stage: LatencyStage) -> Optional[float]:
        """Return the recorded elapsed milliseconds for ``stage``."""

        span = self.spans.get(stage)
        return None if span is None else span.elapsed_ms

    def snapshot(self) -> dict[str, float]:
        """Return a fresh ``{field_name: elapsed_ms}`` dict."""

        return {
            STAGE_FIELD_NAMES[stage]: span.elapsed_ms
            for stage, span in self.spans.items()
        }
