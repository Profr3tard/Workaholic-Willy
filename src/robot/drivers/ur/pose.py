"""
URPose: pose representation and conversions for Universal Robots.

A UR TCP pose is ``[x, y, z, rx, ry, rz]`` where translations are in
**metres** and rotations are axis-angle (radians). Internally this
package always uses **millimetres** to match the OpenCV calibration
pipeline (ChArUco detector, hand-eye calibration, marker poses).

This is the central numerics contract:

* **Inside Workaholic-Willy** mm + axis-angle rad. Every ``URPose`` instance,
  every 4x4 matrix produced by :meth:`to_T`, every distance returned by
  :meth:`distance_to`.
* **At the UR boundary** m + axis-angle rad. Conversions happen in
  :meth:`to_ur_list` / :meth:`from_ur_list`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2 as cv
import numpy as np

from src.utility.io import dump_json, load_json


@dataclass(frozen=True, slots=True)
class URPose:
    """
    Immutable robot TCP pose.

    All translations are in **millimetres**, rotations are axis-angle
    (radians). The UR controller uses metres; conversion happens at
    the boundary (see ``to_ur_list`` / ``from_ur_list``).
    """

    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float
    label: str = ""

    def to_ur_list(self) -> list[float]:
        """Return ``[x_m, y_m, z_m, rx, ry, rz]`` with metres."""
        return [
            self.x / 1000.0,
            self.y / 1000.0,
            self.z / 1000.0,
            self.rx,
            self.ry,
            self.rz,
        ]

    @classmethod
    def from_ur_list(cls, tcp: list[float], label: str = "") -> URPose:
        """
        Create from UR-format ``[x_m, y_m, z_m, rx, ry, rz]``.
        Converts metres to millimetres internally.
        """
        if len(tcp) != 6:
            raise ValueError(
                f"UR TCP list must have 6 elements, got {len(tcp)}: {tcp!r}"
            )
        x, y, z, rx, ry, rz = tcp
        if not all(np.isfinite(v) for v in tcp):
            raise ValueError(f"UR TCP list must be finite, got {tcp!r}")
        return cls(
            x=x * 1000.0,
            y=y * 1000.0,
            z=z * 1000.0,
            rx=rx,
            ry=ry,
            rz=rz,
            label=label,
        )

    def to_T(self) -> np.ndarray:
        """Convert to 4x4 homogeneous transform (mm, rotation matrix)."""
        rvec = np.array([self.rx, self.ry, self.rz], dtype=np.float64)
        R, _ = cv.Rodrigues(rvec)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = [self.x, self.y, self.z]
        return T

    @classmethod
    def from_T(cls, T: np.ndarray, label: str = "") -> URPose:
        """Create from a 4x4 homogeneous matrix (mm)."""
        T = np.asarray(T)
        if T.shape != (4, 4):
            raise ValueError(f"T must have shape (4, 4), got {T.shape}")
        R = T[:3, :3].astype(np.float64)
        t = T[:3, 3]
        rvec, _ = cv.Rodrigues(R)
        rvec = rvec.flatten()
        return cls(
            x=float(t[0]),
            y=float(t[1]),
            z=float(t[2]),
            rx=float(rvec[0]),
            ry=float(rvec[1]),
            rz=float(rvec[2]),
            label=label,
        )

    @property
    def position_mm(self) -> np.ndarray:
        """Translation as (3,) array in mm."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    def distance_to(self, other: URPose) -> float:
        """Euclidean distance in mm between two poses."""
        return float(np.linalg.norm(self.position_mm - other.position_mm))

    def angle_to(self, other: URPose) -> float:
        """Rotation difference in degrees between two poses."""
        R1 = self.to_T()[:3, :3]
        R2 = other.to_T()[:3, :3]
        cos = np.clip((np.trace(R1.T @ R2) - 1.0) / 2.0, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos)))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> URPose:
        return cls(**d)

    @staticmethod
    def save_list(poses: list[URPose], path: str | Path) -> Path:
        """Write a list of poses to a JSON file."""
        path = Path(path)
        dump_json([p.to_dict() for p in poses], path)
        return path

    @staticmethod
    def load_list(path: str | Path) -> list[URPose]:
        """Load a list of poses from a JSON file."""
        data = load_json(path)
        return [URPose.from_dict(d) for d in data]
