"""Hand-eye calibration config schema."""

from __future__ import annotations

from typing import ClassVar


from pydantic import Field, model_validator

from .._base import StrictModel


class RobotCalibrationQualityBandsMm(StrictModel):
    """RMSE thresholds (mm) that classify a hand-eye calibration result.

    Boundaries are inclusive upper bounds, so a value of exactly
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

    Two keys this block does not offer:

    * ``quality_threshold_mm``, an auto-apply gate on the RMSE that would hold a result above it
      for operator review. No such gate exists: ``save_extrinsics`` writes whatever it is given,
      judging the RMSE is the operator's job, and :data:`quality_bands_mm` labels it.
    * ``speed_scale``, an arm-speed scale for the duration of the routine. There is no speed
      governor; ``robot.motion_limits`` is what the driver layer enforces, and the routine takes
      a ``motion_limits=`` argument.

    ``quality_bands_mm`` is the opposite case: both calibration runners (``run_eth_calibrate.py``,
    ``run_eih_calibrate.py``) pass it straight into :func:`src.calibration.quality.classify_rmse`.
    """

    # Mechanical settle time after each pose move, before image capture.
    settle_time_s: float = Field(default=0.5, ge=0.0)

    # Random orientation deviation (deg) when generating calibration poses.
    orientation_spread_deg: float = Field(default=15.0, ge=0.0, le=180.0)

    # Pose-generator retry budget per pose before giving up.
    max_attempts_per_pose: int = Field(default=200, ge=1)

    #: ETH pose-generator z band per marker source (``"ground_truth"`` / ``"aruco"``), as ``(lo, hi)``
    #: fractions of ``workspace_limits.z_max``. An absent key keeps that source's shipped default
    #: (ground truth: the full box; ArUco: 0.55-0.75, the band that holds the tool marker inside the
    #: narrow overhead FOV).
    #:
    #: Keyed by source because the two sources want opposite bands, so a single value fixes one and
    #: breaks the other. On a UR3e, ArUco collects 11/22 samples at 0.55-0.75 (rmse 6.44 mm) and 1/22
    #: at 0.5-0.9 (21 status=timeout), while ground truth collects 9/22 at 0.5-0.9 (rmse 0.0000 mm)
    #: and 4/22 on the full box (16 status=timeout), too few to solve. A UR5e needs neither band: it
    #: collects 17/22 on the full box and solves exactly, so the default is None and a cell that sets
    #: no key keeps the shipped behaviour.
    #:
    #: The pose box is the sampler's range, not a filter: moving one bound re-draws every pose, so a
    #: narrower band is not a safer band. ``run_eth_calibrate --z-frac LO,HI`` overrides it for a
    #: one-off experiment.
    pose_box_z_frac: dict[str, tuple[float, float]] | None = None

    #: The only marker sources a pose box can be keyed by: the same fixed set ``run_eth_calibrate``
    #: offers as ``--marker`` choices. Kept next to the field so the two cannot drift apart silently.
    MARKER_SOURCES: ClassVar[tuple[str, ...]] = ("ground_truth", "aruco")

    @model_validator(mode="after")
    def _check_pose_box_z_frac(self) -> "RobotCalibrationConfig":
        """Refuse the three ways this field can be silently wrong.

        Unchecked, a typo'd key, an inverted band and out-of-range fractions are all accepted. A
        typo'd key is the silent one: ``{"typo_source": [0.5, 0.9]}`` validates, the runner's
        ``.get(marker)`` then returns ``None``, the shipped default applies, and the operator's
        setting is gone without a word.
        """
        if self.pose_box_z_frac is None:
            return self
        for key, band in self.pose_box_z_frac.items():
            if key not in self.MARKER_SOURCES:
                raise ValueError(
                    f"calibration.pose_box_z_frac has an unknown marker source {key!r}. A key that is "
                    f"not looked up is silently ignored: the shipped default applies and your setting "
                    f"disappears. Valid keys: {list(self.MARKER_SOURCES)}."
                )
            lo, hi = float(band[0]), float(band[1])
            if not (0.0 <= lo < hi <= 1.0):
                raise ValueError(
                    f"calibration.pose_box_z_frac[{key!r}] = ({lo}, {hi}) is not a band. It is read as "
                    f"fractions of workspace_limits.z_max, so it needs 0 <= lo < hi <= 1; an inverted "
                    f"pair would produce z_min > z_max and the pose generator would sample nothing."
                )
        return self

    #: Structurally identical to :class:`src.calibration.quality.QualityBandsMm`; the config layer
    #: redeclares it rather than importing it, because config must not depend on the source tree.
    quality_bands_mm: RobotCalibrationQualityBandsMm = Field(
        default_factory=RobotCalibrationQualityBandsMm
    )
