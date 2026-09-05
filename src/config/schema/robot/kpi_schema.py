"""Typed schema for the locked KPI thresholds.

The artefact lives at ``config/robot/kpi_thresholds.yaml``.
Operators may override any value per-deployment without code
changes.

All "rate" fields use the unit interval (``[0, 1]``). The
``median_cycle_time_increase_pct_max`` is expressed as a percentage
(``5.0`` == "+5%").
"""

from __future__ import annotations

from pydantic import ConfigDict, Field

from src.config.schema._base import StrictModel


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
