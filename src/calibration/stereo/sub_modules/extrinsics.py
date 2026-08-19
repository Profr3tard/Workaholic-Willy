"""ExtrinsicsTransformer: Transform 3D points from camera to robot base coordinates."""

import numpy as np

from src.calibration.exceptions import StereoCalibrationError
from src.calibration.extrinsics import Extrinsics as CalibrationExtrinsics
from src.geometry import Frame, Transform
from src.geometry.validation import validate_position_mm

__all__ = ["ExtrinsicsTransformer"]

class ExtrinsicsTransformer:
    """
    Handles 3D coordinate transformations from a camera frame to a robot base frame.

    This class stores a 4x4 homogeneous transformation matrix `T_cam_to_base` that 
    represents the extrinsic calibration between a camera and a robot base. It can 
    validate, set, and apply this transformation to 3D points.

    Usage:
        transformer = ExtrinsicsTransformer(T_cam_to_base)
        point_base = transformer.transform(point_cam)

    Attributes:
        T (np.ndarray | None): 4x4 homogeneous transformation matrix from camera
            to base coordinates. None if not set.

    Methods:
        set_matrix(T): Set or update the transformation matrix with validation.
        transform(point_cam): Transform a 3D point from camera coordinates
            to base coordinates.

    Example:
        >>> T = np.eye(4)
        >>> transformer = ExtrinsicsTransformer(T)
        >>> point_cam = np.array([1.0, 2.0, 3.0])
        >>> point_base = transformer.transform(point_cam)
    """
    
    def __init__(
            self,
            T_cam_to_base: np.ndarray | None = None
    ):
        self._transform: Transform | None = None
        if T_cam_to_base is not None:
            self.set_matrix(T_cam_to_base)

    @property
    def T(self) -> np.ndarray | None:
        return None if self._transform is None else self._transform.to_matrix()
    
    def _coerce_transform(self, value) -> Transform:
        if isinstance(value, CalibrationExtrinsics):
            return value.transform
        if isinstance(value, Transform):
            transform = value
        else:
            try:
                transform = Transform.from_matrix(
                    np.asarray(value, dtype=np.float64),
                    from_frame=Frame.CAMERA,
                    to_frame=Frame.BASE,
                )
            except Exception as exc:
                raise StereoCalibrationError("T_cam_to_base must be a valid 4x4 transform") from exc
        if transform.from_frame is not Frame.CAMERA or transform.to_frame is not Frame.BASE:
            raise StereoCalibrationError(
                "T_cam_to_base must be Transform(CAMERA -> BASE); "
                f"got {transform.from_frame.name} -> {transform.to_frame.name}"
            )
        return transform
    
    def set_matrix(self, T):
        self._transform = self._coerce_transform(T)

    def set_transform(self, transform: Transform | CalibrationExtrinsics) -> None:
        self._transform = self._coerce_transform(transform)
    
    def transform(self, point_cam):
        """
        Transforms a 3D point from camera coordinates to robot base coordinates.

        Args:
                1. Convert the 3D camera point to homogeneous coordinates:
                        P_c = [x_c, y_c, z_c, 1]

                2. Multiply with the transformation matrix:
                        P_r = T_cam_to_base @ P_c

                3. Convert back from homogeneous coordinates by taking the first
                three components:
                        [x_r, y_r, z_r]

        Returns:
                The returned point represents the position of the original camera
                point expressed in the robot base coordinate frame.
        """

        if self._transform is None:
            raise StereoCalibrationError("T_cam_to_base is not set")
        point = validate_position_mm(point_cam, name="point_cam")
        return self._transform.apply_point(point)