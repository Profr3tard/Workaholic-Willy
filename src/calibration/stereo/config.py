from dataclasses import dataclass, field
from pathlib import Path

import cv2 as cv
import numpy as np

from src.calibration.exceptions import StereoCalibrationError
from src.calibration.helpers import proj_to_K

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
    """Stereo calibration outputs.

    The small per-camera parameters (intrinsics ``camL``/``camR``, distortion
    ``distL``/``distR``, rectification rotations ``rectL``/``rectR`` and
    projections ``projL``/``projR``, plus ``Q``) are the source of truth and
    the only fields persisted. The per-pixel rectification remap tables and the
    rectified intrinsics are derived on construction, so a megabyte of maps
    never has to be serialised.
    """

    camL: np.ndarray
    distL: np.ndarray
    rectL: np.ndarray
    projL: np.ndarray
    camR: np.ndarray
    distR: np.ndarray
    rectR: np.ndarray
    projR: np.ndarray
    Q: np.ndarray
    frame_size: tuple[int, int]

    stereoMapL_x: np.ndarray = field(init=False)
    stereoMapL_y: np.ndarray = field(init=False)
    stereoMapR_x: np.ndarray = field(init=False)
    stereoMapR_y: np.ndarray = field(init=False)
    K_rect: np.ndarray = field(init=False)
    fx_rect: float = field(init=False)
    fov_x_deg: float = field(init=False)

    def __post_init__(self) -> None:
        if len(self.frame_size) != 2:
            raise StereoCalibrationError("frame_size must be (width, height)")
        width, height = int(self.frame_size[0]), int(self.frame_size[1])
        if width <= 0 or height <= 0:
            raise StereoCalibrationError("frame_size values must be positive")
        object.__setattr__(self, "frame_size", (width, height))

        for name, shape in (
            ("camL", (3, 3)), ("camR", (3, 3)),
            ("rectL", (3, 3)), ("rectR", (3, 3)),
            ("projL", (3, 4)), ("projR", (3, 4)),
            ("Q", (4, 4)),
        ):
            array = np.ascontiguousarray(getattr(self, name), dtype=np.float64)
            if array.shape != shape:
                raise StereoCalibrationError(f"{name} must have shape {shape}, got {array.shape}")
            if not np.all(np.isfinite(array)):
                raise StereoCalibrationError(f"{name} contains non-finite values")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        for name in ("distL", "distR"):
            array = np.ascontiguousarray(getattr(self, name), dtype=np.float64).reshape(-1)
            if array.size == 0 or not np.all(np.isfinite(array)):
                raise StereoCalibrationError(f"{name} must be a finite, non-empty vector")
            array.setflags(write=False)
            object.__setattr__(self, name, array)

        # Derive the rectification remap tables + rectified intrinsics.
        size = (width, height)
        mapL_x, mapL_y = cv.initUndistortRectifyMap(self.camL, self.distL, self.rectL, self.projL, size, cv.CV_16SC2)
        mapR_x, mapR_y = cv.initUndistortRectifyMap(self.camR, self.distR, self.rectR, self.projR, size, cv.CV_16SC2)
        maps = {
            "stereoMapL_x": np.asarray(mapL_x),
            "stereoMapL_y": np.asarray(mapL_y),
            "stereoMapR_x": np.asarray(mapR_x),
            "stereoMapR_y": np.asarray(mapR_y),
        }
        for name, mp in maps.items():
            mp.setflags(write=False)
            object.__setattr__(self, name, mp)

        fx_rect = float(self.projR[0, 0])
        k_rect = np.ascontiguousarray(proj_to_K(self.projL), dtype=np.float64)
        k_rect.setflags(write=False)
        object.__setattr__(self, "K_rect", k_rect)
        object.__setattr__(self, "fx_rect", fx_rect)
        object.__setattr__(self, "fov_x_deg", float(np.degrees(2.0 * np.arctan(width / (2.0 * fx_rect)))))