import numpy as np

from src.calibration.exceptions import StereoCalibrationError
from src.calibration.extrinsics import Extrinsics as CalibrationExtrinsics
from src.geometry import Frame, Transform
from src.geometry.validation import validate_position_mm

__all__ = ["ExtrinsicsTransformer"]

class ExtrinsicsTransformer:
    """
    Transforms 3D points from a camera frame into a robot base frame.

    Holds the extrinsic calibration as one Transform tagged CAMERA -> BASE;
    any other frame pair is refused. `set_matrix` takes a 4x4 matrix and
    `set_transform` a Transform or a calibration Extrinsics, each replacing
    what is held. Points are positions in millimetres. `T` is the same
    transform as a 4x4 homogeneous matrix, or None while none is set.
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

        `point_cam` is a position in millimetres; it is validated and then
        carried through T_cam_to_base. Raises StereoCalibrationError while no
        transform is set.
        """

        if self._transform is None:
            raise StereoCalibrationError("T_cam_to_base is not set")
        point = validate_position_mm(point_cam, name="point_cam")
        return self._transform.apply_point(point)