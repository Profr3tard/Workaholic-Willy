"""Continuous (per-control-step) software collision-AVOIDANCE monitor.

HONESTY (safety-critical): this is a SOFTWARE collision-AVOIDANCE layer, NOT "the safety system".
It REDUCES collision risk in simulation. The real-hardware safety GUARANTEE is independent, certified
functional safety — hardware E-stop circuits + ISO 10218 / ISO/TS 15066 / ISO 13849 — which runs
INDEPENDENTLY of this software. A green sim result is NOT "commercially safe".
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

from ._fcl_self_collision import make_backend
from ._ur_kinematics import ur_link_transforms_mm

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ._fcl_self_collision import MeshSelfCollisionBackend


class ContinuousGuardAbort(RuntimeError):
    """Raised mid-motion when the continuous guard stops a move (collision margin / fail-safe).

    The motion HALTS at the last cleared waypoint (never continues past a STOP). Carries the
    :class:`MonitorVerdict` so the caller can log the pair/distance/reason of the safe abort.
    """

    def __init__(self, verdict: MonitorVerdict) -> None:
        self.verdict = verdict
        super().__init__(
            f"continuous guard STOP: {verdict.status}"
            + (f" {verdict.pair} {verdict.distance_mm:.3f}mm" if verdict.pair else "")
        )


class MonitorStatus(StrEnum):
    """A check is OK only if it ran in budget and every pair stayed clear of the margin."""

    OK = "ok"
    COLLISION = "collision"      # a monitored pair fell below the margin -> STOP
    TIMEOUT = "timeout"          # the check overran max_check_ms -> STOP (fail-safe: can't keep up)
    UNAVAILABLE = "unavailable"  # kinematics/backend could not produce a result -> STOP (fail-closed)


@dataclass(frozen=True, slots=True)
class ContinuousGuardProfile:
    """The named safety profile (opt-in). ``enabled=False`` (default) == byte-identical (no guard)."""

    enabled: bool = False
    margin_mm: float = 8.0      # STOP at this clearance BEFORE contact. Keep well under the ~19.6 mm
    #                             natural-grasp wrist clearance or a GOOD pick is false-stopped.
    max_check_ms: float = 12.0  # fail-safe budget: a check slower than this -> STOP (cannot keep up at
    #                             the control rate). 12 ms ~= a 60-80 Hz monitor.


@dataclass(frozen=True, slots=True)
class MonitorVerdict:
    """The outcome of one per-control-step check. ``stop`` is True for anything but OK."""

    status: MonitorStatus
    elapsed_ms: float
    pair: str | None = None
    distance_mm: float | None = None

    @property
    def stop(self) -> bool:
        return self.status is not MonitorStatus.OK


class ContinuousCollisionMonitor:
    """Per-step mesh collision-avoidance with a distance margin + a self fail-safe (fail-closed)."""

    def __init__(
        self,
        backend: MeshSelfCollisionBackend,
        model: str,
        yaw_deg: float,
        fixtures: Sequence[object],
        profile: ContinuousGuardProfile,
    ) -> None:
        self._backend = backend
        self._model = model
        self._yaw = float(yaw_deg)
        self._fixtures = tuple(fixtures)
        self._profile = profile
        self.engine = getattr(backend, "engine", "fcl")

    @classmethod
    def from_model(
        cls,
        model: str,
        yaw_deg: float,
        fixtures: Sequence[object],
        profile: ContinuousGuardProfile,
        mesh_dir: str | None = None,
    ) -> ContinuousCollisionMonitor | None:
        """Build the monitor, or ``None`` if the mesh backend is unavailable (no engine / bundle)."""
        backend = make_backend(model, mesh_dir)
        if backend is None:
            return None
        return cls(backend, model, yaw_deg, fixtures, profile)

    @property
    def profile(self) -> ContinuousGuardProfile:
        return self._profile

    def check(self, joints_rad: Sequence[float] | np.ndarray) -> MonitorVerdict:
        """Check one configuration. Fail-closed: any failure to produce a clear result -> STOP."""
        t0 = time.perf_counter()
        try:
            transforms = ur_link_transforms_mm(self._model, np.asarray(joints_rad, dtype=np.float64))
            if transforms is None:
                return MonitorVerdict(MonitorStatus.UNAVAILABLE, (time.perf_counter() - t0) * 1000.0)
            hit = self._backend.evaluate(
                transforms, self._yaw, self._fixtures, self._profile.margin_mm, broadphase=True
            )
        except Exception:  # noqa: BLE001 - fail-closed: any backend error is a STOP, never a pass-through
            return MonitorVerdict(MonitorStatus.UNAVAILABLE, (time.perf_counter() - t0) * 1000.0)
        elapsed = (time.perf_counter() - t0) * 1000.0
        if hit is not None:
            pair, dmm = hit
            return MonitorVerdict(MonitorStatus.COLLISION, elapsed, pair, dmm)
        if elapsed > self._profile.max_check_ms:
            # the check completed but too slowly to trust at the control rate -> fail-safe STOP/HOLD
            return MonitorVerdict(MonitorStatus.TIMEOUT, elapsed)
        return MonitorVerdict(MonitorStatus.OK, elapsed)
