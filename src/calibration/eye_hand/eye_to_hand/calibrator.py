"""Fixed-camera eye-to-hand calibration workflow."""

from __future__ import annotations

from src.geometry import Frame, Transform

from src.calibration.extrinsics import Extrinsics
from src.calibration.solver import HandEyeAXXB

from ..common import BaseEyeHandCalibrator
from ..types import EyeHandCalibrationResult, MountingMode

__all__ = ["EyeToHandCalibrator"]


class EyeToHandCalibrator(BaseEyeHandCalibrator):
    """Solve ``T_cam_to_base`` for a fixed camera observing the robot tool."""

    def calibrate(self) -> EyeHandCalibrationResult:
        """Return a typed ``Transform(CAMERA -> BASE)`` result."""
        self._assert_ready()
        T_base_to_tool_list, T_cam_to_marker_list = self._sample_matrices()
        A_mats = HandEyeAXXB.relative_motions(T_base_to_tool_list)
        B_mats = HandEyeAXXB.relative_motions(T_cam_to_marker_list)
        T_cam_to_base, rmse_mm, max_error_mm = self._solve_axxb(A_mats, B_mats)
        transform = Transform.from_matrix(
            T_cam_to_base,
            from_frame=Frame.CAMERA,
            to_frame=Frame.BASE,
        )
        return EyeHandCalibrationResult.from_solver(
            mode=MountingMode.EYE_TO_HAND,
            transform=transform,
            rmse_mm=rmse_mm,
            max_error_mm=max_error_mm,
            num_samples=len(self.dataset),
            bands=self.bands,
        )

    def calibrate_extrinsics(self, *, rig_id: str) -> Extrinsics:
        """Solve and return schema-versioned camera-to-base extrinsics."""
        return self.calibrate().to_extrinsics(rig_id=rig_id)