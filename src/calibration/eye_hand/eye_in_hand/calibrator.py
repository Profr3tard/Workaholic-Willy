"""Tool-mounted-camera eye-in-hand calibration workflow."""

from __future__ import annotations

from src.geometry import Frame, Transform

from ..common import BaseEyeHandCalibrator
from ..types import EyeHandCalibrationResult, MountingMode

__all__ = ["EyeInHandCalibrator"]


class EyeInHandCalibrator(BaseEyeHandCalibrator):
    """Solve ``T_cam_to_tool`` for a camera mounted on the robot tool."""

    def calibrate(self) -> EyeHandCalibrationResult:
        """Return a typed ``Transform(CAMERA -> TOOL)`` result.

        Refuses with :class:`CalibrationDataError` unless the dataset holds
        ``min_samples`` poses rotating about two independent axes.
        """
        self._assert_ready()
        T_base_to_tool_list, T_cam_to_marker_list = self._sample_matrices()
        pair_count = len(T_base_to_tool_list) - 1

        A_mats = [
            self._inverse(T_base_to_tool_list[index + 1]) @ T_base_to_tool_list[index]
            for index in range(pair_count)
        ]
        B_mats = [
            T_cam_to_marker_list[index + 1] @ self._inverse(T_cam_to_marker_list[index])
            for index in range(pair_count)
        ]
        T_cam_to_tool, rmse_mm, max_error_mm = self._solve_axxb(A_mats, B_mats)
        transform = Transform.from_matrix(
            T_cam_to_tool,
            from_frame=Frame.CAMERA,
            to_frame=Frame.TOOL,
        )
        return EyeHandCalibrationResult.from_solver(
            mode=MountingMode.EYE_IN_HAND,
            transform=transform,
            rmse_mm=rmse_mm,
            max_error_mm=max_error_mm,
            num_samples=len(self.dataset),
            bands=self.bands,
        )