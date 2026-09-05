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
    """Filesystem layout for RGB-D rigs: where the intrinsics artefact lives.

    Only ``base_dir`` is required; ``intrinsics_file`` is derived from it and can be overridden.

    There are deliberately no ``color_images_dir`` / ``depth_images_dir`` fields, unlike the stereo
    rig above where those are real. The stereo capture pipeline reads through
    ``left/right_images_glob`` and this class has no glob, so nothing could read such folders
    even if something wrote them. The RGB-D artefact story does not use folders at all:
    ``export_intrinsics_on_open`` writes ``intrinsics_file`` and ``load_intrinsics`` reads it back,
    while ``StereoCapturePipeline`` explicitly skips every RGB-D rig. If in-repo RGB-D intrinsics
    capture is ever built, add the dir and the glob together: one without the other is dead.
    """

    base_dir: str
    intrinsics_file: str = ""

    @model_validator(mode="before")
    @classmethod
    def _derive_from_base(cls, data: Any) -> Any:
        if isinstance(data, dict):
            base = data.get("base_dir", "")
            if base:
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
    """Stereo-calibration ChArUco board parameters + this block's ArUco marker settings.

    These describe the stereo board only. The standalone hand-eye marker is a separate physical
    artefact configured under ``camera.hand_eye`` and may use a different dictionary; under
    ``WILLY_PROFILE=sim`` it demonstrably does (stereo ``DICT_5X5_100`` / 50 mm vs hand-eye
    ``DICT_4X4_50`` / 48 mm). There is deliberately no cross-block validator tying them: OpenCV allows
    different dictionaries, and the board and the single marker are different objects.
    """

    charuco_squares_x: int = Field(gt=1)
    charuco_squares_y: int = Field(gt=1)
    charuco_square_length_mm: float = Field(gt=0.0)
    charuco_marker_length_mm: float = Field(gt=0.0)
    frame_size: tuple[int, int]
    rectify_alpha: float = Field(ge=0.0, le=1.0)
    marker_length_mm: float = Field(gt=0.0)
    aruco_dict_name: ArucoDictName

    @model_validator(mode="after")
    def _check(self) -> CalibrationConfig:
        if len(self.frame_size) != 2 or self.frame_size[0] <= 0 or self.frame_size[1] <= 0:
            raise ValueError("frame_size must contain two positive values")
        if self.charuco_marker_length_mm >= self.charuco_square_length_mm:
            raise ValueError(
                "charuco_marker_length_mm must be < charuco_square_length_mm (the marker fits inside a square)"
            )
        return self


class BaseRigConfig(StrictModel):
    """Fields common to every camera rig variant."""

    rig_id: str = Field(min_length=1)
    enabled: bool
    source: Literal["webcam_pair", "single_device", "rgbd"]

    fps: int = Field(gt=0)
    backend: int = Field(default=0, ge=0)

    quality: QualityConfig = Field(default_factory=QualityConfig)
