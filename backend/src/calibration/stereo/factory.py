from __future__ import annotations

from pathlib import Path
from typing import Optional, TYPE_CHECKING

from .config import StereoRigConfig

from .repository import StereoRigRepository
from .runtime import StereoRigRunTime

from .sub_modules.calibration import StereoCalibration, CalibrationResult
from .sub_modules.rectifier import StereoRectifier
from .sub_modules.disparity import DisparityComputer
from .sub_modules.pointcloud import PointCloudReconstructor
from .sub_modules.extrinsics import ExtrinsicsTransformer

from backend.src.calibration.exceptions import StereoCalibrationError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from backend.config.schema.camera import StereoMatcherConfig

__all__ = ["StereoRigFactory"]

class StereoRigFactory:
    def __init__(self, chessboard_size, square_size_mm, rectify_alpha, stereo_matcher_cfg: StereoMatcherConfig, logger):
        self.logger = logger
        self.stereo_matcher_cfg = stereo_matcher_cfg
        self.calibration = StereoCalibration(chessboard_size, square_size_mm, rectify_alpha)
        self.repository = StereoRigRepository()

    def create(self, config: StereoRigConfig, frame_size) -> StereoRigRunTime:
        calib_result, extrinsics = self._load_or_calibrate(config, frame_size)

        extrinsics_transformer = ExtrinsicsTransformer()
        if extrinsics is not None:
            extrinsics_transformer.set_matrix(extrinsics)

        rectifier = StereoRectifier(calib_result)
        disparity = DisparityComputer(self.stereo_matcher_cfg)
        pointcloud = PointCloudReconstructor(calib_result.Q)

        return StereoRigRunTime(
            config=config,
            calib_result=calib_result,
            rectifier=rectifier,
            disparity=disparity,
            pointcloud=pointcloud,
            extrinsics=extrinsics_transformer,
        )

    def _load_or_calibrate(self, config: StereoRigConfig, frame_size) -> tuple[CalibrationResult, Optional[object]]:
        stereomap_file = Path(config.stereomap_file)

        if stereomap_file.exists():
            try:
                calib_result, T_cam_to_base = self.repository.load(stereomap_file)
            except (OSError, RuntimeError, ValueError, StereoCalibrationError) as exc:
                raise StereoCalibrationError(
                    f"failed to load stereo map {stereomap_file!s}"
                ) from exc
            self.logger.info("[Rig] Loaded stereomap: %s", stereomap_file)
            return calib_result, T_cam_to_base

        self.logger.info(
            "[Rig] Start stereo calibration using images: %s | %s",
            config.left_glob,
            config.right_glob,
        )

        calib_result = self.calibration.calibrate(
            frame_size,
            config.left_glob,
            config.right_glob
        )

        self.repository.save(stereomap_file, calib_result, None)
        self.logger.info("[Rig] Calibration saved -> %s", stereomap_file)
        return calib_result, None