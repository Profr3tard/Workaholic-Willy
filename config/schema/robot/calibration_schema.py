"""Hand-eye calibration config schema."""

from __future__ import annotations

from typing import ClassVar


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
    """Tunable parameters for the hand-eye calibration routine.

    ``quality_bands_mm`` below is the opposite case and stays: both calibration runners pass it straight
    into :func:`src.calibration.quality.classify_rmse`.
    """

    # Mechanical settle time after each pose move, before image capture.
    settle_time_s: float = Field(default=0.5, ge=0.0)
    # Random orientation deviation (deg) when generating calibration poses.
    orientation_spread_deg: float = Field(default=15.0, ge=0.0, le=180.0)
    # Pose-generator retry budget per pose before giving up.
    max_attempts_per_pose: int = Field(default=200, ge=1)
    pose_box_z_frac: dict[str, tuple[float, float]] | None = None
    MARKER_SOURCES: ClassVar[tuple[str, ...]] = ("ground_truth", "aruco")

    @model_validator(mode="after")
    def _check_pose_box_z_frac(self) -> "RobotCalibrationConfig":
        """Refuse the three ways this field was silently wrong."""
        if self.pose_box_z_frac is None:
            return self
        for key, band in self.pose_box_z_frac.items():
            if key not in self.MARKER_SOURCES:
                raise ValueError(
                    f"calibration.pose_box_z_frac has an unknown marker source {key!r}. A key that is "
                    f"not looked up is silently ignored -- the shipped default applies and your setting "
                    f"disappears. Valid keys: {list(self.MARKER_SOURCES)}."
                )
            lo, hi = float(band[0]), float(band[1])
            if not (0.0 <= lo < hi <= 1.0):
                raise ValueError(
                    f"calibration.pose_box_z_frac[{key!r}] = ({lo}, {hi}) is not a band. It is read as "
                    f"FRACTIONS of workspace_limits.z_max, so it needs 0 <= lo < hi <= 1; an inverted "
                    f"pair would produce z_min > z_max and the pose generator would sample nothing."
                )
        return self

    quality_bands_mm: RobotCalibrationQualityBandsMm = Field(
        default_factory=RobotCalibrationQualityBandsMm
    )
