"""Derive the tool frame a UR controller is ACTUALLY using with no new vendor SDK surface."""

from __future__ import annotations

import math

import numpy as np

from src.geometry.matrix import invert_homogeneous
from src.robot.safety._ur_kinematics import ur_link_transforms_mm

__all__ = ["ToolFrameMismatch", "derive_active_tool_frame", "tool_frame_matrix", "compare_tool_frames"]


class ToolFrameMismatch(ValueError):
    """The controller's derived tool frame disagrees with what config declares."""


def tool_frame_matrix(
    offset_mm: tuple[float, float, float],
    rotation_quat_xyzw: tuple[float, float, float, float],
) -> np.ndarray:
    """The declared flange -> TCP transform as a 4x4 (translation in mm)."""
    from src.geometry.quaternion import to_rotation_matrix

    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = to_rotation_matrix(np.asarray(rotation_quat_xyzw, dtype=np.float64))
    t[:3, 3] = np.asarray(offset_mm, dtype=np.float64)
    return t


def derive_active_tool_frame(
    model: str,
    joints_rad: "np.ndarray | list[float]",
    controller_tcp_mm: np.ndarray,
) -> np.ndarray | None:
    """Flange -> TCP as the CONTROLLER currently has it, or ``None`` if ``model`` has no DH table.

    Parameters
    ----------
    model
        UR model key (``"ur3e"``, ``"ur5e"``, ...) selects the bundled DH table.
    joints_rad
        The joint vector the controller's FK was evaluated at.
    controller_tcp_mm
        4x4 base -> active-TCP, translation in mm, i.e. the controller's own FK for those joints.
    """
    q = np.asarray(joints_rad, dtype=np.float64)
    links = ur_link_transforms_mm(model, q)
    if links is None:
        return None  # unknown model or joint-count mismatch, the caller decides what that means
    base_to_flange = links[-1]
    return invert_homogeneous(base_to_flange) @ np.asarray(controller_tcp_mm, dtype=np.float64)


def compare_tool_frames(observed: np.ndarray, declared: np.ndarray) -> tuple[float, float]:
    """``(translation_mm, rotation_deg)`` between two flange->TCP transforms."""
    d_t = float(np.linalg.norm(np.asarray(observed)[:3, 3] - np.asarray(declared)[:3, 3]))
    r = np.asarray(observed)[:3, :3] @ np.asarray(declared)[:3, :3].T
    cos = (float(np.trace(r)) - 1.0) / 2.0
    d_r = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
    return d_t, d_r
