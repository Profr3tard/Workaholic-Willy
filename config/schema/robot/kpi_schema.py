"""Typed schema for the locked KPI thresholds."""

from __future__ import annotations

from pydantic import ConfigDict, Field

from config.schema._base import StrictModel


class EasyModeThresholds(StrictModel):
    model_config = ConfigDict(frozen=True)

    false_positive_grasp_rate_max: float = Field(ge=0.0, le=1.0)
    dead_loop_rate_max: float = Field(ge=0.0, le=1.0)
    median_cycle_time_increase_pct_max: float = Field(ge=0.0)


class AutoModeThresholds(StrictModel):
    model_config = ConfigDict(frozen=True)

    dead_loop_rate_max: float = Field(ge=0.0, le=1.0)


class DenseModeThresholds(StrictModel):
    model_config = ConfigDict(frozen=True)

    dead_loop_rate_max: float = Field(ge=0.0, le=1.0)
    false_positive_grasp_rate_max: float = Field(ge=0.0, le=1.0)


class SoakThresholds(StrictModel):
    model_config = ConfigDict(frozen=True)

    min_attempts: int = Field(gt=0)
    dead_loop_rate_max: float = Field(ge=0.0, le=1.0)


class KpiThresholdsConfig(StrictModel):
    """Top-level schema mounted on ``kpi_thresholds.yaml``."""

    model_config = ConfigDict(frozen=True)

    easy: EasyModeThresholds
    auto: AutoModeThresholds
    dense: DenseModeThresholds
    soak: SoakThresholds
