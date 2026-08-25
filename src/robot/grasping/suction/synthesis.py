"""Synthesize ranked suction grasp candidates from surface geometry.

Back-projects the object mask to a point cloud, estimates surface normals
and curvature, selects flat contact points, and scores them through a
``SuctionScorer``. Candidates are ranked by quality and can be transformed
to BASE. Pure NumPy; execution/attach is handled by the robot or simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from src.robot.grasping.geometry.normals import (
    NormalEstimationConfig,
    estimate_surface_normals,
)
from src.robot.grasping.geometry.pointcloud import masked_point_cloud
from src.robot.grasping.types.grasp_point import GraspFrame
from src.robot.grasping.suction.scorer import AnalyticalSuctionScorer, SuctionScorer

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.geometry import Transform

__all__ = ["SuctionGrasp", "SuctionConfig", "synthesize_suction_grasps"]


@dataclass(frozen=True, slots=True)
class SuctionGrasp:
    """
    A single ranked suction candidate
    (contact ``position_mm`` + unit ``approach``, ``quality`` ∈ [0, 1] ranking key, in :attr:`frame`).
    """

    position_mm: np.ndarray
    approach: np.ndarray
    seal_score: float
    quality: float
    frame: GraspFrame
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SuctionConfig:
    """Suction synthesis parameters."""

    scorer: SuctionScorer = field(default_factory=AnalyticalSuctionScorer)
    normals: NormalEstimationConfig | None = None
    max_eval_candidates: int = 80      # surface points scored, caps the cost
    max_results: int = 5
    min_quality: float = 0.2           # candidates below this combined quality are dropped
    min_cloud_points: int = 16

    def __post_init__(self) -> None:
        if self.max_eval_candidates < 1:
            raise ValueError("max_eval_candidates must be >= 1")
        if self.max_results < 1:
            raise ValueError("max_results must be >= 1")
        if not 0.0 <= self.min_quality <= 1.0:
            raise ValueError("min_quality must be in [0, 1]")


def synthesize_suction_grasps(
    segmentation: Any,
    depth_map: np.ndarray,
    intrinsics: np.ndarray,
    *,
    rgb: np.ndarray | None = None,
    camera_to_base: "Transform | None" = None,
    payload_mass_g: float | None = None,
    config: SuctionConfig | None = None,
) -> list[SuctionGrasp]:
    """
    Synthesize up to ``config.max_results`` ranked suction candidates for one segmented object
    (BASE frame with ``camera_to_base``, else CAMERA; ``payload_mass_g`` enables the wrench term;
    ``rgb`` feeds only the learned scorer).
    """
    cfg = config or SuctionConfig()
    mask = np.asarray(getattr(segmentation, "mask", segmentation), dtype=bool)
    cloud = np.asarray(
        masked_point_cloud(mask, depth_map, intrinsics, unit="mm").points_mm, dtype=np.float64
    )
    if cloud.shape[0] < cfg.min_cloud_points:
        return []

    # Per-scene scorer setup: a no-op for the analytical scorer, and the hook where an image-based one
    # would run its network ONCE. In Meters!!!
    cfg.scorer.prepare_scene(
        rgb, np.asarray(depth_map, dtype=np.float64) / 1000.0, np.asarray(intrinsics, dtype=np.float64)
    )

    sn = estimate_surface_normals(cloud, cfg.normals)
    normals = np.asarray(sn.normals, dtype=np.float64)
    valid = np.asarray(sn.valid_mask, dtype=bool)
    cand = np.flatnonzero(valid)
    if cand.size == 0:
        return []
    centroid = cloud.mean(axis=0)  # visible-surface CoM proxy for the wrench term
    # UNIFORM spatial subsample for full-surface coverage, capped at max_eval_candidates. The
    # seal model then does the real ranking; high-curvature points simply score low and drop out.
    stride = max(1, cand.size // cfg.max_eval_candidates)
    cand = cand[::stride]

    R = t = None
    frame = GraspFrame.CAMERA
    # The wrench term needs gravity in the SCORING frame (the cloud frame). World-down is BASE -Z; in the
    # camera frame that is R_base_to_cam @ [0,0,-1] = Rᵀ @ [0,0,-1] (the cloud is scored in camera frame, then
    # the winners are transformed to base). Without a transform we assume a gravity-aligned cloud frame.
    gravity_dir = np.array([0.0, 0.0, -1.0])
    if camera_to_base is not None:
        m = np.asarray(camera_to_base.to_matrix(), dtype=np.float64)
        R, t = m[:3, :3], m[:3, 3]
        frame = GraspFrame.BASE
        gravity_dir = R.T @ np.array([0.0, 0.0, -1.0])

    scored: list[tuple[float, float, SuctionGrasp]] = []
    for j in cand:
        p = cloud[j]
        nrm = normals[j]
        nrm_norm = float(np.linalg.norm(nrm))
        if nrm_norm < 1e-9:
            continue
        approach = -nrm / nrm_norm  # press INTO the surface, anti-parallel to the (camera-facing) normal
        q = cfg.scorer.score(
            p, approach, cloud,
            surface_normal=nrm, payload_mass_g=payload_mass_g, com_mm=centroid,
            gravity_dir=gravity_dir,
        )
        if q.quality < cfg.min_quality:
            continue
        cdist = float(np.linalg.norm(p - centroid))

        pos = p if R is None else (R @ p + t)
        appr = approach if R is None else (R @ approach)
        meta: dict[str, Any] = {"source": q.source}
        if q.seal is not None:
            meta.update(
                max_deformation_mm=float(q.seal.max_deformation_mm),
                perimeter_support=float(q.seal.perimeter_support),
                normal_alignment=float(q.seal.normal_alignment),
            )
        if q.wrench is not None:
            meta.update(
                wrench_feasible=bool(q.wrench.feasible),
                wrench_resist=float(q.wrench.resist_score),
            )
        scored.append(
            (
                q.quality,
                cdist,
                SuctionGrasp(
                    position_mm=np.asarray(pos, dtype=np.float64),
                    approach=np.asarray(appr, dtype=np.float64),
                    seal_score=float(q.seal.seal_score) if q.seal is not None else float(q.quality),
                    quality=float(q.quality),
                    frame=frame,
                    metadata=meta,
                ),
            )
        )

    # Rank by quality (desc); break ties toward the CoM (asc centroid distance) among equally sealable
    # contacts the most central is the most robust (lowest wrench moment) + makes the result deterministic.
    scored.sort(key=lambda kv: (-kv[0], kv[1]))
    return [g for _q, _d, g in scored[: cfg.max_results]]
