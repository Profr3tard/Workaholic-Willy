"""UR driver connection config schema."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from .._base import StrictModel
from ._ur_models import UR_MODEL_KEYS


class URConfig(StrictModel):
    """UR-specific connection settings."""

    model: str = Field(default="ur5e")
    ip: str = Field(default="192.168.1.100", min_length=1)
    rtde_frequency: float = Field(default=0.0, ge=0.0)
    motion_planner: Literal["ik", "curobo"] = "ik"

    @field_validator("model")
    @classmethod
    def _known_model(cls, v: str) -> str:
        """Reject a typo at config load (<1 s) rather than at the moment an arm is asked to move."""
        if v not in UR_MODEL_KEYS:
            raise ValueError(f"unknown UR model {v!r}; supported: {list(UR_MODEL_KEYS)}")
        return v
