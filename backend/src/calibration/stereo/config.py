from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.src.calibration.exceptions import StereoCalibrationError

__all__ = ["CalibrationResult", "StereoRigConfig"]


@dataclass(frozen=True, slots=True)
class StereoRigConfig:
    stereomap_file: str | Path
    left_glob: str
    right_glob: str

    def __post_init__(self) -> None:
        if not str(self.stereomap_file).strip():
            raise StereoCalibrationError("stereomap_file must be a non-empty path")
        if not self.left_glob.strip():
            raise StereoCalibrationError("left_glob must be a non-empty glob pattern")
        if not self.right_glob.strip():
            raise StereoCalibrationError("right_glob must be a non-empty glob pattern")


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Stereo calibration outputs required by the runtime pipeline."""

    stereoMapL_x: np.ndarray
    stereoMapL_y: np.ndarray
    stereoMapR_x: np.ndarray
    stereoMapR_y: np.ndarray
    Q: np.ndarray
    projL: np.ndarray
    projR: np.ndarray
    K_rect: np.ndarray
    fx_rect: float
    fov_x_deg: float
    frame_size: tuple[int, int]

    def __post_init__(self) -> None:
        for name in (
            "stereoMapL_x",
            "stereoMapL_y",
            "stereoMapR_x",
            "stereoMapR_y",
            "Q",
            "projL",
            "projR",
            "K_rect",
        ):
            array = np.asarray(getattr(self, name))
            if array.size == 0:
                raise StereoCalibrationError(f"{name} must not be empty")
            if not np.all(np.isfinite(array)):
                raise StereoCalibrationError(f"{name} contains non-finite values")
            array.setflags(write=False)
            object.__setattr__(self, name, array)

        if self.Q.shape != (4, 4):
            raise StereoCalibrationError(f"Q must have shape (4, 4), got {self.Q.shape}")
        if self.projL.shape != (3, 4) or self.projR.shape != (3, 4):
            raise StereoCalibrationError("projL and projR must have shape (3, 4)")
        if self.K_rect.shape != (3, 3):
            raise StereoCalibrationError(f"K_rect must have shape (3, 3), got {self.K_rect.shape}")

        if len(self.frame_size) != 2:
            raise StereoCalibrationError("frame_size must be (width, height)")
        width, height = int(self.frame_size[0]), int(self.frame_size[1])
        if width <= 0 or height <= 0:
            raise StereoCalibrationError("frame_size values must be positive")
        object.__setattr__(self, "frame_size", (width, height))
        object.__setattr__(self, "fx_rect", float(self.fx_rect))
        object.__setattr__(self, "fov_x_deg", float(self.fov_x_deg))