"""
Hardcoded Universal Robots Denavit-Hartenberg parameters used by the
self-collision guard to derive per-link transforms.

DH convention: standard (Craig-style, alpha/a/d/theta_offset with the
joint variable added to ``theta_offset``). Lengths in metres internally;
the consumer scales to mm for the capsule pipeline.

Sources
-------
Universal Robots official kinematic data, accessible from the support
site under "DH parameters for calculations of kinematics and dynamics":
https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/

Honesty notes
-------------
* Only the standard 6-DoF UR models are bundled. Anything else returns
  ``None`` and the self-collision guard treats arm-vs-arm link checks
  as UNAVAILABLE (it still runs tool capsule + fixture checks).
* DH theta-offsets are zero for UR the controller's ``q`` already
  corresponds to the canonical zero-pose ``(0, 0, 0, 0, 0, 0)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "URDhRow",
    "UR_DH_TABLES_M",
    "ur_link_origins_mm",
    "ur_link_transforms_mm",
]


@dataclass(frozen=True, slots=True)
class URDhRow:
    """One row of standard UR DH parameters (lengths in metres)."""

    a_m: float
    d_m: float
    alpha_rad: float


# Per-model 6-row DH tables (metres + radians). See the UR support page
# linked above for the canonical values. Each row corresponds to a
# joint; the link length is mostly the ``a`` for that row and the ``d``
# for the next row.

_UR3_DH: tuple[URDhRow, ...] = (
    URDhRow(0.0, 0.1519, 1.570796327),
    URDhRow(-0.24365, 0.0, 0.0),
    URDhRow(-0.21325, 0.0, 0.0),
    URDhRow(0.0, 0.11235, 1.570796327),
    URDhRow(0.0, 0.08535, -1.570796327),
    URDhRow(0.0, 0.0819, 0.0),
)

_UR3E_DH: tuple[URDhRow, ...] = (
    URDhRow(0.0, 0.15185, 1.570796327),
    URDhRow(-0.24355, 0.0, 0.0),
    URDhRow(-0.2132, 0.0, 0.0),
    URDhRow(0.0, 0.13105, 1.570796327),
    URDhRow(0.0, 0.08535, -1.570796327),
    URDhRow(0.0, 0.0921, 0.0),
)

_UR5_DH: tuple[URDhRow, ...] = (
    URDhRow(0.0, 0.089159, 1.570796327),
    URDhRow(-0.425, 0.0, 0.0),
    URDhRow(-0.39225, 0.0, 0.0),
    URDhRow(0.0, 0.10915, 1.570796327),
    URDhRow(0.0, 0.09465, -1.570796327),
    URDhRow(0.0, 0.0823, 0.0),
)

_UR5E_DH: tuple[URDhRow, ...] = (
    URDhRow(0.0, 0.1625, 1.570796327),
    URDhRow(-0.425, 0.0, 0.0),
    URDhRow(-0.3922, 0.0, 0.0),
    URDhRow(0.0, 0.1333, 1.570796327),
    URDhRow(0.0, 0.0997, -1.570796327),
    URDhRow(0.0, 0.0996, 0.0),
)

_UR10_DH: tuple[URDhRow, ...] = (
    URDhRow(0.0, 0.1273, 1.570796327),
    URDhRow(-0.612, 0.0, 0.0),
    URDhRow(-0.5723, 0.0, 0.0),
    URDhRow(0.0, 0.163941, 1.570796327),
    URDhRow(0.0, 0.1157, -1.570796327),
    URDhRow(0.0, 0.0922, 0.0),
)

_UR10E_DH: tuple[URDhRow, ...] = (
    URDhRow(0.0, 0.1807, 1.570796327),
    URDhRow(-0.6127, 0.0, 0.0),
    URDhRow(-0.57155, 0.0, 0.0),
    URDhRow(0.0, 0.17415, 1.570796327),
    URDhRow(0.0, 0.11985, -1.570796327),
    URDhRow(0.0, 0.11655, 0.0),
)

_UR16E_DH: tuple[URDhRow, ...] = (
    URDhRow(0.0, 0.1807, 1.570796327),
    URDhRow(-0.4784, 0.0, 0.0),
    URDhRow(-0.36, 0.0, 0.0),
    URDhRow(0.0, 0.17415, 1.570796327),
    URDhRow(0.0, 0.11985, -1.570796327),
    URDhRow(0.0, 0.11655, 0.0),
)


UR_DH_TABLES_M: dict[str, tuple[URDhRow, ...]] = {
    "ur3": _UR3_DH,
    "ur3e": _UR3E_DH,
    "ur5": _UR5_DH,
    "ur5e": _UR5E_DH,
    "ur10": _UR10_DH,
    "ur10e": _UR10E_DH,
    "ur16e": _UR16E_DH,
}


def _dh_transform(theta: float, row: URDhRow) -> np.ndarray:
    """Standard 4x4 DH transform (metres)."""
    ct = float(np.cos(theta))
    st = float(np.sin(theta))
    ca = float(np.cos(row.alpha_rad))
    sa = float(np.sin(row.alpha_rad))
    a = row.a_m
    d = row.d_m
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)


def ur_link_origins_mm(model: str, joints_rad: np.ndarray) -> list[np.ndarray] | None:
    """Per-joint frame origins for a UR model, in mm and the robot base frame.

    Returns a list of length ``len(joints_rad) + 1`` where index 0 is
    the base-frame origin ``[0,0,0]`` and the trailing entry is the
    flange (TCP-without-tool) origin. Returns ``None`` if ``model`` is
    not in the bundled DH table.
    """
    table = UR_DH_TABLES_M.get(model.lower())
    if table is None:
        return None
    if joints_rad.shape[0] != len(table):
        return None
    T = np.eye(4, dtype=np.float64)
    origins_m: list[np.ndarray] = [T[:3, 3].copy()]
    for theta, row in zip(joints_rad, table, strict=True):
        T = T @ _dh_transform(float(theta), row)
        origins_m.append(T[:3, 3].copy())
    return [o * 1000.0 for o in origins_m]


def ur_link_transforms_mm(model: str, joints_rad: np.ndarray) -> list[np.ndarray] | None:
    """Full per-link 4x4 transforms for a UR model (rotation + translation), translation in MM."""
    table = UR_DH_TABLES_M.get(model.lower())
    if table is None or joints_rad.shape[0] != len(table):
        return None
    T = np.eye(4, dtype=np.float64)
    out: list[np.ndarray] = [T.copy()]
    for theta, row in zip(joints_rad, table, strict=True):
        T = T @ _dh_transform(float(theta), row)
        M = T.copy()
        M[:3, 3] = M[:3, 3] * 1000.0  # metres -> mm in the translation column only
        out.append(M)
    return out
