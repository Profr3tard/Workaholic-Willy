"""Analytical suction wrench-resistance model.

Evaluates whether a sealed suction contact can resist the object's gravity
wrench using vacuum force, friction, and cup bending resistance. Produces a
feasibility margin based on mass and CoM offset. Pure NumPy, with no learned
weights or simulator dependencies; real-world seal and material effects
remain hardware-dependent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["WrenchConfig", "WrenchResult", "evaluate_wrench_resistance"]

_G_MPS2 = 9.81


@dataclass(frozen=True, slots=True)
class WrenchConfig:
    """Vacuum-cup wrench-resistance parameters (SI-ish, with mm/g at the boundary)."""

    vacuum_kpa: float = 50.0        # gauge vacuum pressure (a typical industrial cup pulls 40–70 kPa)
    cup_radius_mm: float = 15.0
    friction_coef: float = 0.5      # rubber cup on a dry surface
    # Max resistible moment as a fraction of F_vac · r_cup (the elastic ring's restoring moment). 1.0 is a
    # reasonable cup; smaller = a stiffer/smaller cup that tips more easily.
    moment_arm_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.vacuum_kpa <= 0.0:
            raise ValueError(f"vacuum_kpa must be > 0, got {self.vacuum_kpa}")
        if self.cup_radius_mm <= 0.0:
            raise ValueError(f"cup_radius_mm must be > 0, got {self.cup_radius_mm}")
        if self.friction_coef < 0.0:
            raise ValueError(f"friction_coef must be >= 0, got {self.friction_coef}")
        if self.moment_arm_factor <= 0.0:
            raise ValueError(f"moment_arm_factor must be > 0, got {self.moment_arm_factor}")

    @property
    def vacuum_force_n(self) -> float:
        """Max pull-off force ``F_vac = P · A`` (N)."""
        area_m2 = np.pi * (self.cup_radius_mm / 1000.0) ** 2
        return float(self.vacuum_kpa * 1000.0 * area_m2)


@dataclass(frozen=True, slots=True)
class WrenchResult:
    """Wrench-resistance evaluation at one contact (``resist_score`` = tightest of the pull/shear/moment margins)."""

    resist_score: float
    feasible: bool
    vacuum_force_n: float
    required_pull_n: float
    shear_n: float
    required_moment_nmm: float
    max_moment_nmm: float


def evaluate_wrench_resistance(
    contact_point_mm: np.ndarray,
    approach: np.ndarray,
    *,
    payload_mass_g: float,
    com_mm: np.ndarray,
    config: WrenchConfig | None = None,
    gravity_dir: np.ndarray | None = None,
) -> WrenchResult:
    """Evaluate whether a suction cup can hold a payload against gravity.

    Uses the contact point, approach direction, payload mass, CoM, and gravity
    direction to assess wrench resistance.
    """
    cfg = config or WrenchConfig()
    p = np.asarray(contact_point_mm, dtype=np.float64).reshape(3)
    a = np.asarray(approach, dtype=np.float64).reshape(3)
    a = a / float(np.linalg.norm(a))
    com = np.asarray(com_mm, dtype=np.float64).reshape(3)
    g = np.asarray(gravity_dir if gravity_dir is not None else [0.0, 0.0, -1.0], dtype=np.float64)
    g = g / float(np.linalg.norm(g))

    mass_kg = max(float(payload_mass_g), 0.0) / 1000.0
    f_grav = mass_kg * _G_MPS2  # N, gravity magnitude
    f_vac = cfg.vacuum_force_n

    # Gravity force decomposed at the contact. The cup seals on the face whose
    # outward normal is -approach, since it presses into the surface along approach.
    # The pull-off component is gravity along +approach; the remaining component
    # acts as shear in the contact plane.
    grav_vec = f_grav * g
    pull_off = max(0.0, float(np.dot(grav_vec, a)))    # component pulling the part off the cup (N)
    shear = float(np.linalg.norm(grav_vec - np.dot(grav_vec, a) * a))  # tangential component (N)

    # Moment about the contact from gravity acting at the CoM: |(com - p) × F_grav|.
    lever = (com - p) / 1000.0  # m
    moment_nm = float(np.linalg.norm(np.cross(lever, grav_vec)))
    moment_nmm = moment_nm * 1000.0
    max_moment_nmm = f_vac * cfg.cup_radius_mm * cfg.moment_arm_factor

    # Feasibility ratios (limit / demand); inf when there is no demand.
    def _ratio(limit: float, demand: float) -> float:
        if demand <= 1e-9:
            return float("inf")
        return limit / demand

    ratios = [
        _ratio(f_vac, pull_off),                    # pull-off vs vacuum
        _ratio(cfg.friction_coef * f_vac, shear),   # shear vs friction
        _ratio(max_moment_nmm, moment_nmm),         # tilt vs elastic moment
    ]
    tightest = min(ratios)
    resist_score = float(np.clip(tightest, 0.0, 1.0))
    feasible = bool(tightest >= 1.0)
    return WrenchResult(
        resist_score=resist_score,
        feasible=feasible,
        vacuum_force_n=f_vac,
        required_pull_n=pull_off,
        shear_n=shear,
        required_moment_nmm=moment_nmm,
        max_moment_nmm=max_moment_nmm,
    )
