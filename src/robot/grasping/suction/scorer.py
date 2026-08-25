"""Suction scoring protocol and analytical physics-based implementation.

Defines the ``SuctionScorer`` seam used by candidate synthesis and provides
``AnalyticalSuctionScorer`` as the dependency-free default. Scoring combines
seal formation and wrench resistance from geometry and statics. Learned
scorers can implement the same protocol when licence-clean training data is
available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from src.robot.grasping.suction.seal import SealConfig, SealResult, evaluate_seal
from src.robot.grasping.suction.wrench import (
    WrenchConfig,
    WrenchResult,
    evaluate_wrench_resistance,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

__all__ = [
    "SuctionQuality",
    "SuctionScorer",
    "AnalyticalSuctionScorer",
]


@dataclass(frozen=True, slots=True)
class SuctionQuality:
    """Combined suction quality at one contact (``quality`` ∈ [0, 1] ranking key; ``source`` names the backend)."""

    quality: float
    source: str
    seal: SealResult | None = None
    wrench: WrenchResult | None = None


@runtime_checkable
class SuctionScorer(Protocol):
    """Scores a single suction contact. Implemented by the analytical and learned backends."""

    def prepare_scene(
        self,
        rgb: np.ndarray | None,
        depth_m: np.ndarray,
        intrinsics: np.ndarray,
    ) -> None:
        """
        Per-scene setup before scoring: no-op for the analytical backend;
        the learned backend runs its RGB-D network ONCE here and caches the heatmap.
        """
        ...

    def score(
        self,
        contact_point_mm: np.ndarray,
        approach: np.ndarray,
        points_mm: np.ndarray,
        *,
        surface_normal: np.ndarray | None = None,
        payload_mass_g: float | None = None,
        com_mm: np.ndarray | None = None,
        gravity_dir: np.ndarray | None = None,
    ) -> SuctionQuality: ...


@dataclass(frozen=True, slots=True)
class AnalyticalSuctionScorer:
    """
    The sim-validatable default: ``quality = seal_score x wrench_resist_score``,
    or ``quality = seal`` when no payload is supplied.
    """

    seal_config: SealConfig = field(default_factory=SealConfig)
    wrench_config: WrenchConfig = field(default_factory=WrenchConfig)

    def prepare_scene(
        self, rgb: np.ndarray | None, depth_m: np.ndarray, intrinsics: np.ndarray
    ) -> None:
        """No-op: the analytical model scores directly from the point cloud."""
        return None

    def score(
        self,
        contact_point_mm: np.ndarray,
        approach: np.ndarray,
        points_mm: np.ndarray,
        *,
        surface_normal: np.ndarray | None = None,
        payload_mass_g: float | None = None,
        com_mm: np.ndarray | None = None,
        gravity_dir: np.ndarray | None = None,
    ) -> SuctionQuality:
        seal = evaluate_seal(
            contact_point_mm, approach, points_mm,
            surface_normal=surface_normal, config=self.seal_config,
        )
        wrench: WrenchResult | None = None
        quality = seal.seal_score
        if payload_mass_g is not None and com_mm is not None:
            wrench = evaluate_wrench_resistance(
                contact_point_mm, approach,
                payload_mass_g=payload_mass_g, com_mm=com_mm, config=self.wrench_config,
                gravity_dir=gravity_dir,
            )
            quality = seal.seal_score * wrench.resist_score
        return SuctionQuality(quality=float(quality), source="analytical", seal=seal, wrench=wrench)
