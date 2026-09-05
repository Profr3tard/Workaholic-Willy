from dataclasses import dataclass
import numpy as np

from .config import StereoRigConfig, CalibrationResult
from .sub_modules.rectifier import StereoRectifier
from .sub_modules.disparity import DisparityComputer
from .sub_modules.pointcloud import PointCloudReconstructor
from .sub_modules.extrinsics import ExtrinsicsTransformer

__all__ = ["StereoRigRunTime"]


@dataclass(slots=True)
class StereoRigRunTime:
    """One calibrated rig: rectification, disparity, reconstruction, extrinsics.

    Assembled by `StereoRigFactory`. Reconstruction is computed in millimetres and
    the `unit` argument scales it from there; these methods default to `mm`, where
    `PointCloudReconstructor` underneath defaults to `cm`. `transform_cam_to_base`
    raises while the rig has no CAMERA -> BASE matrix.
    """

    config: StereoRigConfig
    calib_result: CalibrationResult
    rectifier: StereoRectifier
    disparity: DisparityComputer
    pointcloud: PointCloudReconstructor
    extrinsics: ExtrinsicsTransformer

    def rectify(self, left_bgr, right_bgr):
        return self.rectifier.rectify(left_bgr, right_bgr)

    def compute_disparity(self, rect_left_bgr, rect_right_bgr):
        return self.disparity.compute(rect_left_bgr, rect_right_bgr)

    def reproject_to_3d(self, disparity_px: np.ndarray):
        return self.pointcloud.reproject(disparity_px)

    def compute_depth_map(self, rect_left_bgr, rect_right_bgr, unit: str = "mm"):
        return self.pointcloud.depth_map(rect_left_bgr, rect_right_bgr, self.disparity, unit)

    def compute_3d_point(self, rect_left_bgr, rect_right_bgr, mask=None, reducer: str = "median", unit: str = "mm"):
        return self.pointcloud.representative_point_mask(
            rect_left_bgr, rect_right_bgr, self.disparity, mask, reducer, unit
        )

    def compute_3d_point_xy(self, rect_left_bgr, rect_right_bgr, x, y, unit: str = "mm"):
        """3D point at pixel `(x, y)`, which is rounded to the nearest whole pixel."""
        xi, yi = int(round(x)), int(round(y))
        return self.pointcloud.representative_point(
            rect_left_bgr, rect_right_bgr, self.disparity, xi, yi, unit
        )

    def set_extrinsics(self, T_cam_to_base_4x4):
        self.extrinsics.set_matrix(T_cam_to_base_4x4)

    def transform_cam_to_base(self, point_cam):
        return self.extrinsics.transform(point_cam)
