"""Fuse multi-view target localizations into one visibility-weighted BASE point.

Combines each active camera's BASE-frame target centroid using its visible pixel
count as weight. Views with no target (``centroid_base_mm=None`` or
``visible_px=0``) contribute nothing, allowing visible cameras to compensate for
occlusion or limited view coverage.

Pure, deterministic, NumPy-only logic with no camera, simulator, or perception
dependencies. Inputs are reduced in order, preserving stable results for a
fixed input.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["ViewLocalization", "fuse_scene_points_base", "fuse_view_localizations"]


@dataclass(frozen=True, slots=True)
class ViewLocalization:
    """One camera's BASE-frame estimate of the target centroid plus its visibility weight.

    ``centroid_base_mm`` is the target's BASE-frame XYZ (mm), or ``None`` when the camera did not see
    it; ``visible_px`` is the visible mask pixel count.
    """

    name: str
    centroid_base_mm: "np.ndarray | None"
    visible_px: int

    @property
    def saw_target(self) -> bool:
        """True iff this view contributes to the fusion (a non-None centroid AND >0 visible pixels)."""
        return self.centroid_base_mm is not None and self.visible_px > 0


def fuse_view_localizations(views: Sequence[ViewLocalization]) -> "np.ndarray | None":
    """Visibility-weighted mean of the per-view BASE-frame target centroids."""
    pts: list[np.ndarray] = []
    weights: list[float] = []
    for v in views:
        if v.saw_target:
            pts.append(np.asarray(v.centroid_base_mm, dtype=np.float64).reshape(3))
            weights.append(float(v.visible_px))
    if not pts:
        return None
    w = np.asarray(weights, dtype=np.float64)
    return (np.stack(pts) * w[:, None]).sum(axis=0) / w.sum()


def fuse_scene_points_base(
    per_camera_clouds: Sequence["np.ndarray | None"],
) -> "np.ndarray | None":
    """Concatenate per-camera BASE-frame neighbour/scene point clouds into ONE fused scene cloud (mm)."""
    clouds: list[np.ndarray] = []
    for c in per_camera_clouds:
        if c is None:
            continue
        arr = np.asarray(c, dtype=np.float64).reshape(-1, 3)
        if arr.size > 0:
            clouds.append(arr)
    if not clouds:
        return None
    return np.vstack(clouds)
