"""Hand-eye calibration config schema for robot."""

from __future__ import annotations


from pydantic import Field, model_validator

from .._base import StrictModel


class RobotCalibrationQualityBandsMm(StrictModel):
    """RMSE thresholds (mm) that classify a hand-eye calibration result.

    Boundaries are *inclusive* upper bounds, so a value of exactly
    ``good`` falls into the "good" band, not "marginal".
    """

    excellent: float = Field(default=1.0, gt=0.0)
    good: float = Field(default=2.5, gt=0.0)
    marginal: float = Field(default=5.0, gt=0.0)

    @model_validator(mode="after")
    def _check_ordering(self) -> RobotCalibrationQualityBandsMm:
        if not (self.excellent < self.good < self.marginal):
            raise ValueError(
                f"quality bands must be strictly ordered: "
                f"{self.excellent} < {self.good} < {self.marginal}"
            )
        return self


class RobotCalibrationConfig(StrictModel):
    """Tunable parameters for the hand-eye calibration routine."""

    # RMSE above this is held for explicit operator review before extrinsics
    # are applied. Should match (or be tighter than) ``quality_bands_mm.good``.
    quality_threshold_mm: float = Field(default=2.5, gt=0.0)

    # Fraction of the operator's global speed setting forced while the
    # routine is running. Restored on exit.
    speed_scale: float = Field(default=0.25, gt=0.0, le=1.0)

    # Mechanical settle time after each pose move, before image capture.
    settle_time_s: float = Field(default=0.5, ge=0.0)

    # Random orientation deviation (deg) when generating calibration poses.
    orientation_spread_deg: float = Field(default=15.0, ge=0.0, le=180.0)

    # Pose-generator retry budget per pose before giving up.
    max_attempts_per_pose: int = Field(default=200, ge=1)

    quality_bands_mm: RobotCalibrationQualityBandsMm = Field(
        default_factory=RobotCalibrationQualityBandsMm
    )

    @model_validator(mode="after")
    def _check_threshold_within_bands(self) -> RobotCalibrationConfig:
        if self.quality_threshold_mm > self.quality_bands_mm.marginal:
            raise ValueError(
                f"quality_threshold_mm ({self.quality_threshold_mm}) must be "
                f"<= quality_bands_mm.marginal ({self.quality_bands_mm.marginal})"
            )
        return self
