"""UR driver connection config schema."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from .._base import StrictModel
from ._ur_models import UR_MODEL_KEYS


class URConfig(StrictModel):
    """UR-specific connection settings.

    Only consulted when ``robot.vendor == "ur"``. Replaces the older
    flat ``connection:`` block; vendor blocks are now siblings under
    ``robot:`` so each driver owns its own typed surface.
    """

    #: Which UR this cell drives. NOT cosmetic: the model keys the safety DH chain, the exact-mesh
    #: collision bundle, and the cuRobo robot config. Before this existed, a real UR3e was planned and
    #: collision-checked against UR5e link lengths (a2/a3 -425/-392.2 mm vs -243.55/-213.2) with nothing
    #: anywhere saying so -- on real hardware. Default "ur5e" keeps every existing cell unchanged.
    model: str = Field(default="ur5e")
    ip: str = Field(default="192.168.1.100", min_length=1)
    rtde_frequency: float = Field(default=0.0, ge=0.0)
    # Approach-phase planner: "ik" (default) = the controller's calibrated IK + a moveJ/moveL
    # straight line; "curobo" = a global collision-free trajectory from the process-isolated cuRobo
    # planner (safety.planning), executed waypoint-by-waypoint over ur_rtde. The "curobo" path is
    # FAIL-CLOSED on real hardware — it never falls back to blind IK. Default keeps today's behaviour.
    motion_planner: Literal["ik", "curobo"] = "ik"

    @field_validator("model")
    @classmethod
    def _known_model(cls, v: str) -> str:
        """Reject a typo at config load (<1 s) rather than at the moment an arm is asked to move."""
        if v not in UR_MODEL_KEYS:
            raise ValueError(f"unknown UR model {v!r}; supported: {list(UR_MODEL_KEYS)}")
        return v
