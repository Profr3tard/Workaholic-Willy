import cv2 as cv
import numpy as np

from src.calibration.exceptions import StereoCalibrationError
from src.calibration.helpers import validate_image_pair_shapes
from src.calibration.stereo.config import CalibrationResult

__all__ = ["StereoRectifier"]

class StereoRectifier:
    """
    Rectifies stereo pairs with the remap tables of a `CalibrationResult`.

    Rectification puts corresponding points on the same image row, the
    epipolar alignment that disparity computation and 3D reconstruction
    assume.
    """
    
    def __init__(self, calib: CalibrationResult):
        self.calib = calib
    
    def rectify(self, left_bgr, right_bgr):
        """
        Rectifies a left/right BGR pair and returns both images, still BGR.

        Both must have the calibration's `frame_size`; any other pair raises
        StereoCalibrationError.
        """
        left = np.asarray(left_bgr)
        right = np.asarray(right_bgr)
        try:
            validate_image_pair_shapes(left, right, expected_size=self.calib.frame_size)
        except Exception as exc:
            raise StereoCalibrationError("cannot rectify incompatible stereo image pair") from exc
        Lr = cv.remap(left, self.calib.stereoMapL_x, self.calib.stereoMapL_y, cv.INTER_LANCZOS4)
        Rr = cv.remap(right, self.calib.stereoMapR_x, self.calib.stereoMapR_y, cv.INTER_LANCZOS4)
        return Lr, Rr