"""Determines a conservative support plane for grasp collision checking.

Uses the higher of the declared support height and the target's lowest
observed point, fusing target footprints across all available camera views.
This avoids underestimating the support plane for stacked objects while
remaining deterministic and independent of vendor SDKs or perception models.
All values are in millimetres.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.geometry import Frame

from .table_collision import SupportPlane

__all__ = ["SupportResolution", "resolve_support_plane"]


class SupportResolution:
    """The plane, and why it is where it is. The reason travels with the number, deliberately."""

    __slots__ = ("plane", "declared_mm", "observed_mm", "source")

    def __init__(self, plane: SupportPlane, declared_mm: float,
                 observed_mm: float | None, source: str) -> None:
        self.plane = plane
        self.declared_mm = declared_mm
        self.observed_mm = observed_mm
        self.source = source

    @property
    def height_mm(self) -> float:
        return float(self.plane.offset_mm)

    def as_telemetry(self) -> dict:
        return {"support_height_mm": round(self.height_mm, 2),
                "support_declared_mm": round(self.declared_mm, 2),
                "support_observed_mm": (None if self.observed_mm is None
                                        else round(self.observed_mm, 2)),
                "support_source": self.source}


def resolve_support_plane(
    *,
    declared_height_mm: float = 0.0,
    container_floor_mm: float | None = None,
    normal: Sequence[float] = (0.0, 0.0, 1.0),
    target_clouds_base_mm: Sequence[np.ndarray] | None = None,
    refine_from_target: bool = True,
) -> SupportResolution:
    """The support plane for one grasp target.

    ``container_floor_mm`` is the optional inside floor of a bin or tray: give it and it replaces the
    workspace height, because a KLT standing on the table raises what its contents rest on by the
    thickness of its own floor.

    ``target_clouds_base_mm`` is one cloud per camera that identified this target (see
    :mod:`~src.robot.grasping.multiview.association` for how a camera decides that it did).

    The plane's offset is a height along ``normal``, so a tilted tray works: the offset is the declared
    surface projected onto the normal, and the observation is the lowest point measured ALONG it.
    """
    unit = np.asarray(normal, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(unit))
    if length < 1e-12:
        raise ValueError("support normal must not be zero")
    unit = unit / length

    declared = float(container_floor_mm if container_floor_mm is not None else declared_height_mm)
    observed: float | None = None
    if refine_from_target and target_clouds_base_mm:
        lows = [float((np.asarray(c, dtype=np.float64).reshape(-1, 3) @ unit).min())
                for c in target_clouds_base_mm
                if np.asarray(c).size]
        if lows:
            # The LOWEST over the cameras, not an average: a camera that cannot see the base reports a
            # height that is too high, and averaging in a wrong number spreads the error instead of
            # discarding it. Whichever camera saw furthest down is the one that saw the truth.
            observed = min(lows)

    if observed is None:
        return SupportResolution(SupportPlane(normal=unit, offset_mm=declared, frame=Frame.BASE),
                                 declared, None, "declared")
    # The HIGHER wins. Never lower: the declared surface is the floor of what is possible, and an
    # observation below it is depth noise looking through the table.
    if observed > declared:
        return SupportResolution(SupportPlane(normal=unit, offset_mm=observed, frame=Frame.BASE),
                                 declared, observed, "observed")
    return SupportResolution(SupportPlane(normal=unit, offset_mm=declared, frame=Frame.BASE),
                             declared, observed, "declared")
