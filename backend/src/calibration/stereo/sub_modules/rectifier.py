import cv2 as cv
import numpy as np

from backend.src.calibration.exceptions import StereoCalibrationError
from backend.src.calibration.helpers import validate_image_pair_shapes
from backend.src.calibration.stereo.config import CalibrationResult

__all__ = ["StereoRectifier"]

class StereoRectifier:
    """
    Performs stereo image rectification using precomputed calibration maps.

    Given a stereo calibration result (`CalibrationResult`), this class
    rectifies a pair of left and right images so that corresponding points
    lie on the same horizontal lines (epipolar alignment). This is required
    for accurate disparity computation and 3D reconstruction.
    """
    
    def __init__(self, calib: CalibrationResult):
        self.calib = calib
    
    def rectify(self, left_bgr, right_bgr):
        """
        Rectifies a pair of stereo images using the calibration maps.

        Args:
            left_bgr (np.ndarray): Left image in BGR format.
            right_bgr (np.ndarray): Right image in BGR format.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                Lr: Rectified left image (BGR).
                Rr: Rectified right image (BGR).
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