"""Hand-eye calibration config schema (R7.1 split from robot_schema.py)."""

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

    Two keys this block deliberately does not offer, because a config key that claims to protect you
    and does not is worse than no key at all:

    * ``quality_threshold_mm``, which would be an auto-apply gate on the RMSE, holding a result
      above it for operator review. No such gate exists anywhere: ``save_extrinsics`` writes
      whatever it is given, judging the RMSE is the operator's job, and :data:`quality_bands_mm`
      labels it. Measured: the only reader of the key in the whole repository was this class's
      own cross-field validator, a rule validating a dead key against a live one.
    * ``speed_scale``, which would scale arm speed during the routine and restore it on exit.
      There is no speed governor; ``robot.motion_limits`` is what the driver layer actually
      enforces, and the routine takes a ``motion_limits=`` argument.

    ``quality_bands_mm`` below is the opposite case: both calibration runners pass it straight into
    :func:`src.calibration.quality.classify_rmse`.
    """

    # Mechanical settle time after each pose move, before image capture.
    settle_time_s: float = Field(default=0.5, ge=0.0)

    # Random orientation deviation (deg) when generating calibration poses.
    orientation_spread_deg: float = Field(default=15.0, ge=0.0, le=180.0)

    # Pose-generator retry budget per pose before giving up.
    max_attempts_per_pose: int = Field(default=200, ge=1)

    #: ETH pose-generator z band PER MARKER SOURCE (``"ground_truth"`` / ``"aruco"``), as ``(lo, hi)``
    #: fractions of ``workspace_limits.z_max``. An absent key keeps that source's shipped default
    #: (ground-truth: the full box; ArUco: 0.55-0.75, the band that holds the tool marker inside the
    #: narrow overhead FOV).
    #:
    #: It is keyed by source because MEASUREMENT says the two sources want opposite things, and a single
    #: value would fix one while breaking the other. On-box 2026-07-24, UR3e, each figure reproduced:
    #:
    #:   ground_truth  full box      4/22 samples, 16x status=timeout, too few to solve
    #:   ground_truth  0.5-0.9       9/22 samples, rmse 0.0000 mm, quality excellent, GATE PASS
    #:   aruco         0.55-0.75    11/22 samples, rmse 6.44 mm, quality poor
    #:   aruco         0.5-0.9       1/22 samples, 21x status=timeout   <- the SAME band that fixes GT
    #:
    #: (A UR5e needs neither: it collects 17/22 on the full box and solves exactly. So the default is
    #: None and every existing cell is untouched.) The pose box is the sampler's RANGE, not a
    #: filter: moving one bound re-draws every pose, so a narrower band is not a safer band.
    #: ``run_eth_calibrate --z-frac LO,HI`` overrides it for a one-off experiment.
    pose_box_z_frac: dict[str, tuple[float, float]] | None = None

    #: The only marker sources a pose box can be keyed by: the same fixed set ``run_eth_calibrate``
    #: offers as ``--marker`` choices. Kept next to the field so the two cannot drift apart silently.
    MARKER_SOURCES: ClassVar[tuple[str, ...]] = ("ground_truth", "aruco")

    @model_validator(mode="after")
    def _check_pose_box_z_frac(self) -> "RobotCalibrationConfig":
        """Refuse the three ways this field can be silently wrong.

        Unchecked, a typo'd key, an inverted band and out-of-range fractions are all accepted. The
        typo is the worst of the three: ``{"typo_source": [0.5, 0.9]}`` validates, the runner's
        ``.get(marker)`` then returns ``None``, the shipped default applies, and the operator's
        setting has vanished without a word.
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
                    f"FRACTIONS of workspace_limits.z_max, so it needs 0 <= lo < hi <= 1; an inverted "
                    f"pair would produce z_min > z_max and the pose generator would sample nothing."
                )
        return self

    #: Consumed for real: both calibration runners classify their result with it via
    #: ``classify_rmse(result.rmse_mm, cast(QualityBandsMm, cfg.calibration.quality_bands_mm))``
    #: (``run_eth_calibrate.py:191``, ``run_eih_calibrate.py:162``). Structurally identical to
    #: :class:`src.calibration.quality.QualityBandsMm`; the config layer redeclares it rather
    #: than importing it, because config must not depend on the source tree.
    quality_bands_mm: RobotCalibrationQualityBandsMm = Field(
        default_factory=RobotCalibrationQualityBandsMm
    )
