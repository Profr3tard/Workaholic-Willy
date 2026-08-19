"""Typed eye-hand calibration models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from src.geometry import Frame, Transform

from src.calibration.exceptions import CalibrationDataError
from src.calibration.extrinsics import Extrinsics
from src.calibration.quality import (
    DEFAULT_BANDS_MM,
    QualityBandsMm,
    QualityLabel,
    classify_rmse,
)

__all__ = [
    "EyeHandCalibrationResult",
    "EyeHandCalibrationSettings",
    "MountingMode",
]


class MountingMode(StrEnum):
    """Supported camera mounting modes for hand-eye calibration."""

    EYE_TO_HAND = "eye_to_hand"
    EYE_IN_HAND = "eye_in_hand"


@dataclass(frozen=True, slots=True)
class EyeHandCalibrationSettings:
    """Runtime settings for eye-hand sample acceptance and solving."""

    min_samples: int = 4
    min_distance_mm: float = 10.0
    min_angle_deg: float = 5.0

    def __post_init__(self) -> None:
        if not isinstance(self.min_samples, int) or isinstance(self.min_samples, bool):
            raise CalibrationDataError("min_samples must be an integer")
        if self.min_samples < 4:
            raise CalibrationDataError("min_samples must be >= 4 for AX=XB solving")
        for name, value in (
            ("min_distance_mm", self.min_distance_mm),
            ("min_angle_deg", self.min_angle_deg),
        ):
            scalar = float(value)
            if not math.isfinite(scalar) or scalar < 0.0:
                raise CalibrationDataError(f"{name} must be finite and >= 0")

    @classmethod
    def from_config(cls, config: object) -> EyeHandCalibrationSettings:
        """Build settings from a Pydantic-style config object without coupling to it."""
        angle = getattr(config, "min_angle_deg", getattr(config, "min_angle", 5.0))
        return cls(
            min_samples=int(getattr(config, "min_samples")),
            min_distance_mm=float(getattr(config, "min_distance_mm")),
            min_angle_deg=float(angle),
        )


@dataclass(frozen=True, slots=True)
class EyeHandCalibrationResult:
    """Frame-checked result returned by either eye-hand workflow."""

    mode: MountingMode
    transform: Transform
    rmse_mm: float
    max_error_mm: float
    num_samples: int
    quality: QualityLabel
    captured_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.mode, MountingMode):
            object.__setattr__(self, "mode", MountingMode(self.mode))
        if not isinstance(self.transform, Transform):
            raise CalibrationDataError("transform must be a geometry.Transform")
        expected_to_frame = (
            Frame.BASE if self.mode is MountingMode.EYE_TO_HAND else Frame.TOOL
        )
        if (
            self.transform.from_frame is not Frame.CAMERA
            or self.transform.to_frame is not expected_to_frame
        ):
            raise CalibrationDataError(
                f"{self.mode.value} requires Transform(CAMERA -> {expected_to_frame.name}); "
                f"got {self.transform.from_frame.name} -> {self.transform.to_frame.name}"
            )
        for name, value in (("rmse_mm", self.rmse_mm), ("max_error_mm", self.max_error_mm)):
            scalar = float(value)
            if not math.isfinite(scalar) or scalar < 0.0:
                raise CalibrationDataError(f"{name} must be finite and >= 0")
            object.__setattr__(self, name, scalar)
        if self.max_error_mm + 1e-9 < self.rmse_mm:
            raise CalibrationDataError("max_error_mm must be >= rmse_mm")
        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise CalibrationDataError("num_samples must be a positive integer")
        if self.captured_at.tzinfo is None:
            raise CalibrationDataError("captured_at must be timezone-aware")

    @classmethod
    def from_solver(
        cls,
        *,
        mode: MountingMode,
        transform: Transform,
        rmse_mm: float,
        max_error_mm: float,
        num_samples: int,
        bands: QualityBandsMm = DEFAULT_BANDS_MM,
        captured_at: datetime | None = None,
    ) -> EyeHandCalibrationResult:
        timestamp = captured_at if captured_at is not None else datetime.now(UTC)
        return cls(
            mode=mode,
            transform=transform,
            rmse_mm=float(rmse_mm),
            max_error_mm=float(max_error_mm),
            num_samples=int(num_samples),
            quality=classify_rmse(rmse_mm, bands),
            captured_at=timestamp,
        )

    def to_extrinsics(self, *, rig_id: str) -> Extrinsics:
        """Convert an eye-to-hand result into persisted camera-to-base extrinsics."""
        if self.mode is not MountingMode.EYE_TO_HAND:
            raise CalibrationDataError("only eye_to_hand results can be saved as Extrinsics")
        return Extrinsics(
            transform=self.transform,
            rmse_mm=self.rmse_mm,
            max_error_mm=self.max_error_mm,
            num_samples=self.num_samples,
            captured_at=self.captured_at,
            rig_id=rig_id,
            quality=self.quality,
        )