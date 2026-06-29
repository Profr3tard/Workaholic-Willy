"""Camera schemas shared between rig types and calibration routines."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import AfterValidator, Field, model_validator

from .._base import StrictModel, validate_aruco_dict_name

# ArUco dictionary name with strict validation against the OpenCV catalogue.
ArucoDictName = Annotated[str, AfterValidator(validate_aruco_dict_name)]


# ---------------------------------------------------------------------------
# Calibration paths (auto-derived from ``base_dir``)
# ---------------------------------------------------------------------------

class StereoCalibPaths(StrictModel):
    """Filesystem layout for stereo rigs (``webcam_pair`` / ``single_device``).

    Only ``base_dir`` is required. Sub-paths are auto-derived from it but can
    be overridden individually by writing them out in YAML.
    """

    base_dir: str
    stereo_map_file: str = ""
    left_images_dir: str = ""
    right_images_dir: str = ""
    left_images_glob: str = ""
    right_images_glob: str = ""

    @model_validator(mode="before")
    @classmethod
    def _derive_from_base(cls, data: Any) -> Any:
        if isinstance(data, dict):
            base = data.get("base_dir", "")
            if base:
                data.setdefault("stereo_map_file", f"{base}/stereoMap.xml")
                data.setdefault("left_images_dir", f"{base}/left")
                data.setdefault("right_images_dir", f"{base}/right")
                data.setdefault("left_images_glob", f"{base}/left/*.png")
                data.setdefault("right_images_glob", f"{base}/right/*.png")
        return data


class RGBDCalibPaths(StrictModel):
    """Filesystem layout for RGB-D rigs.

    Only ``base_dir`` is required; sub-paths are auto-derived but can be
    overridden in YAML.
    """

    base_dir: str
    color_images_dir: str = ""
    depth_images_dir: str = ""
    intrinsics_file: str = ""

    @model_validator(mode="before")
    @classmethod
    def _derive_from_base(cls, data: Any) -> Any:
        if isinstance(data, dict):
            base = data.get("base_dir", "")
            if base:
                data.setdefault("color_images_dir", f"{base}/color")
                data.setdefault("depth_images_dir", f"{base}/depth")
                data.setdefault("intrinsics_file", f"{base}/intrinsics.json")
        return data


# ---------------------------------------------------------------------------
# Shared rig + calibration settings
# ---------------------------------------------------------------------------

class QualityConfig(StrictModel):
    """Camera quality / capture-mode tuning shared by all rig types."""

    prefer_uncompressed: bool = False
    manual_exposure: float | None = None
    manual_gain: float | None = None
    manual_wb: float | None = None
    disable_auto_features: bool = True
    warmup_frames: int = Field(default=30, ge=0)
    fps_tolerance: float = Field(default=3.0, ge=0.0)


class CalibrationConfig(StrictModel):
    """Stereo-calibration pattern parameters (chessboard + ArUco)."""

    chessboard_size: tuple[int, int]
    square_size_mm: float = Field(gt=0.0)
    frame_size: tuple[int, int]
    rectify_alpha: float = Field(ge=0.0, le=1.0)
    marker_length_mm: float = Field(gt=0.0)
    aruco_dict_name: ArucoDictName

    @model_validator(mode="after")
    def _check_positive_pairs(self) -> CalibrationConfig:
        for name, pair in (
            ("chessboard_size", self.chessboard_size),
            ("frame_size", self.frame_size),
        ):
            if len(pair) != 2 or pair[0] <= 0 or pair[1] <= 0:
                raise ValueError(f"{name} must contain two positive values")
        return self


class BaseRigConfig(StrictModel):
    """Fields common to every camera rig variant."""

    rig_id: str = Field(min_length=1)
    enabled: bool
    source: Literal["webcam_pair", "single_device", "rgbd"]

    fps: int = Field(gt=0)
    backend: int = Field(default=0, ge=0)

    quality: QualityConfig = Field(default_factory=QualityConfig)
