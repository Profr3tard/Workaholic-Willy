from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from .config import CalibrationResult, StereoRigConfig

from .repository import StereoRigRepository
from .runtime import StereoRigRunTime

from .sub_modules.calibrator import StereoCalibrator
from .sub_modules.rectifier import StereoRectifier
from .sub_modules.disparity import DisparityComputer
from .sub_modules.pointcloud import PointCloudReconstructor
from .sub_modules.extrinsics import ExtrinsicsTransformer

from src.calibration.constants import CALIBRATION_LOG_DIR, STEREO_FACTORY_LOG_FILE
from src.calibration.exceptions import StereoCalibrationError
from src.utility.log_cfg import create_logger

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from backend.config.schema.camera import StereoMatcherConfig

__all__ = ["StereoRigFactory"]

# The injected `self.logger` belongs to the OWNING rig (StereoCam3D) and keeps its existing
# lines; this module logger is the calibration package's own trail, carrying the timings,
# sizes and artifact provenance you want when a rig comes up with the wrong geometry.
logger = create_logger("StereoRigFactory", STEREO_FACTORY_LOG_FILE, log_dir=CALIBRATION_LOG_DIR)

class StereoRigFactory:
    def __init__(
        self,
        squares_x,
        squares_y,
        square_length_mm,
        marker_length_mm,
        aruco_dict_name,
        rectify_alpha,
        stereo_matcher_cfg: StereoMatcherConfig,
        logger,
    ):
        self.logger = logger
        self.stereo_matcher_cfg = stereo_matcher_cfg
        self.calibrator = StereoCalibrator(
            squares_x, squares_y, square_length_mm, marker_length_mm, aruco_dict_name, rectify_alpha
        )
        self.repository = StereoRigRepository()

    def create(self, config: StereoRigConfig, frame_size) -> StereoRigRunTime:
        calib_result, extrinsics = self._load_or_calibrate(config, frame_size)

        extrinsics_transformer = ExtrinsicsTransformer()
        if extrinsics is not None:
            extrinsics_transformer.set_matrix(extrinsics)

        rectifier = StereoRectifier(calib_result)
        disparity = DisparityComputer(self.stereo_matcher_cfg)
        pointcloud = PointCloudReconstructor(calib_result.Q)

        logger.info(
            "Stereo rig runtime built: frame_size=%s, stereomap=%s, extrinsics=%s.",
            frame_size,
            config.stereomap_file,
            "loaded" if extrinsics is not None else "absent",
        )
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
            logger.info(
                "Stereomap loaded: %s (%d bytes, frame_size=%s, extrinsics=%s).",
                stereomap_file,
                stereomap_file.stat().st_size,
                calib_result.frame_size,
                "present" if T_cam_to_base is not None else "absent",
            )
            return calib_result, T_cam_to_base

        self.logger.info(
            "[Rig] Start stereo calibration using images: %s | %s",
            config.left_glob,
            config.right_glob,
        )

        started = time.perf_counter()
        calib_result = self.calibrator.calibrate(
            frame_size,
            config.left_glob,
            config.right_glob
        )
        logger.info(
            "Stereo calibration finished in %.1f s (frame_size=%s).",
            time.perf_counter() - started,
            calib_result.frame_size,
        )

        self.repository.save(stereomap_file, calib_result, None)
        self.logger.info("[Rig] Calibration saved -> %s", stereomap_file)
        logger.info(
            "Stereomap written: %s (%d bytes, no extrinsics -- hand-eye still owes a CAMERA->BASE).",
            stereomap_file,
            stereomap_file.stat().st_size,
        )
        return calib_result, None