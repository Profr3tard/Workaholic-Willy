"""Shared implementation for eye-hand calibrators."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from backend.src.geometry.matrix import invert_homogeneous

from backend.src.calibration.exceptions import CalibrationDataError, CalibrationSolveError
from backend.src.calibration.quality import DEFAULT_BANDS_MM, QualityBandsMm
from backend.src.calibration.solver import HandEyeAXXB

from .dataset import EyeHandDataset, EyeHandSample
from .types import EyeHandCalibrationSettings

__all__ = ["BaseEyeHandCalibrator"]


def _rotation_angle_deg(T_first: np.ndarray, T_second: np.ndarray) -> float:
    rotation_delta = T_first[:3, :3].T @ T_second[:3, :3]
    cosine = float(np.clip((np.trace(rotation_delta) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _poses_are_diverse(
    T_first: np.ndarray,
    T_second: np.ndarray,
    *,
    min_distance_mm: float,
    min_angle_deg: float,
) -> bool:
    distance = float(np.linalg.norm(T_first[:3, 3] - T_second[:3, 3]))
    angle = _rotation_angle_deg(T_first, T_second)
    return distance > min_distance_mm or angle > min_angle_deg


def _rotation_axis_from_relative(relative_transform: np.ndarray) -> np.ndarray | None:
    rotation = relative_transform[:3, :3]
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < math.radians(1.0):
        return None
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(axis))
    if norm == 0.0:
        return None
    return axis / norm


class BaseEyeHandCalibrator:
    """Base class shared by the two explicit hand-eye workflows."""

    def __init__(
        self,
        *,
        settings: EyeHandCalibrationSettings | object | None = None,
        dataset: EyeHandDataset | None = None,
        solver: HandEyeAXXB | None = None,
        bands: QualityBandsMm = DEFAULT_BANDS_MM,
    ) -> None:
        if settings is None:
            self.settings = EyeHandCalibrationSettings()
        elif isinstance(settings, EyeHandCalibrationSettings):
            self.settings = settings
        else:
            self.settings = EyeHandCalibrationSettings.from_config(settings)
        self.dataset = dataset if dataset is not None else EyeHandDataset()
        self.solver = solver if solver is not None else HandEyeAXXB()
        self.bands = bands

    def add_sample(
        self,
        T_base_to_tool: np.ndarray,
        T_cam_to_marker: np.ndarray | None,
        marker_id: int = 0,
    ) -> bool:
        """Validate and append a sample, returning False for skipped detections."""
        if T_cam_to_marker is None:
            return False
        sample = EyeHandSample(
            T_base_to_tool=T_base_to_tool,
            T_cam_to_marker=T_cam_to_marker,
            marker_id=marker_id,
        )
        if len(self.dataset) > 0:
            has_diverse_pose = all(
                _poses_are_diverse(
                    existing.T_base_to_tool,
                    sample.T_base_to_tool,
                    min_distance_mm=self.settings.min_distance_mm,
                    min_angle_deg=self.settings.min_angle_deg,
                )
                for existing in self.dataset.iter_samples()
            )
            if not has_diverse_pose:
                return False
        self.dataset.add_sample(sample)
        return True

    def save_dataset(self, path: str | Path) -> Path:
        return self.dataset.save(path)

    def load_dataset(self, path: str | Path) -> int:
        self.dataset = EyeHandDataset.load(path)
        return len(self.dataset)

    def _sample_matrices(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        return self.dataset.base_to_tool_matrices(), self.dataset.cam_to_marker_matrices()

    def _assert_ready(self) -> None:
        if len(self.dataset) < self.settings.min_samples:
            raise CalibrationDataError(
                f"too few samples: got {len(self.dataset)}, need >= {self.settings.min_samples}"
            )
        T_base_to_tool_list = self.dataset.base_to_tool_matrices()
        relative_robot_motions = HandEyeAXXB.relative_motions(T_base_to_tool_list)
        axes = [
            axis
            for axis in (
                _rotation_axis_from_relative(relative)
                for relative in relative_robot_motions
            )
            if axis is not None
        ]
        if len(axes) < 2:
            raise CalibrationDataError(
                "sample set needs at least two meaningful robot rotation motions"
            )
        axis_rank = int(np.linalg.matrix_rank(np.vstack(axes), tol=0.1))
        if axis_rank < 2:
            raise CalibrationDataError(
                "sample set needs robot rotations around at least two independent axes"
            )

    def _solve_axxb(
        self,
        A_mats: list[np.ndarray],
        B_mats: list[np.ndarray],
    ) -> tuple[np.ndarray, float, float]:
        try:
            transform_matrix, rmse = self.solver.solve(A_mats, B_mats)
        except CalibrationDataError:
            raise
        except Exception as exc:
            raise CalibrationSolveError("AX=XB solve failed") from exc
        residuals = [
            float(np.linalg.norm(A @ transform_matrix - transform_matrix @ B, "fro"))
            for A, B in zip(A_mats, B_mats)
        ]
        max_error = max(residuals) if residuals else float(rmse)
        return transform_matrix, float(rmse), float(max_error)

    @staticmethod
    def _inverse(matrix: np.ndarray) -> np.ndarray:
        return invert_homogeneous(matrix)