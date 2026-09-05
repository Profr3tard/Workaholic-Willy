"""UR driver connection config schema."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from .._base import StrictModel
from ._ur_models import UR_MODEL_KEYS


class URConfig(StrictModel):
    """UR-specific connection settings.

    Only consulted when ``robot.vendor == "ur"``. Vendor blocks are
    siblings under ``robot:`` so each driver owns its own typed surface.
    """

    #: Which UR this cell drives. NOT cosmetic: the model keys the safety DH chain, the exact-mesh
    #: collision bundle, and the cuRobo robot config. Before this existed, a real UR3e was planned and
    #: collision-checked against UR5e link lengths (a2/a3 -425/-392.2 mm vs -243.55/-213.2) with nothing
    #: anywhere saying so, on real hardware. Default "ur5e" keeps every existing cell unchanged.
    model: str = Field(default="ur5e")
    ip: str = Field(default="192.168.1.100", min_length=1)
    #: RTDE update rate in Hz. ``0.0`` means "not configured: let the controller pick its default"
    #: (500 Hz on the e-Series). That is OUR convention, not ur_rtde's: its sentinel for the same
    #: thing is -1.0 and it reads 0.0 as literally zero hertz, which fails synchronisation. The UR
    #: connection translates 0.0 -> -1.0 at the driver boundary; measured against URSim 5.26.0 on
    #: 2026-08-18, where 0.0 failed in 6.1 s and both -1.0 and 500.0 connected.
    rtde_frequency: float = Field(default=0.0, ge=0.0)
    # Approach-phase planner. "curobo" (the default) plans a global collision-free trajectory in the
    # process-isolated cuRobo planner (safety.planning) and executes it waypoint by waypoint over
    # ur_rtde; "ik" is the controller's calibrated IK and a moveJ/moveL straight line, which knows
    # nothing about the cell and will drive through anything in it.
    #
    # The "curobo" path is FAIL-CLOSED: with the planner unavailable the move returns
    # CONTROLLER_REJECTED and with no collision-free plan it returns TIMEOUT. It never degrades to
    # blind IK, so a cell with no cuRobo environment does not move at all rather than moving blindly.
    # That is the point of the default; `python -m src.robot.safety.planning --doctor` is how
    # you check before commissioning.
    motion_planner: Literal["ik", "curobo"] = "curobo"

    @field_validator("model")
    @classmethod
    def _known_model(cls, v: str) -> str:
        """Reject a typo at config load (<1 s) rather than at the moment an arm is asked to move."""
        if v not in UR_MODEL_KEYS:
            raise ValueError(f"unknown UR model {v!r}; supported: {list(UR_MODEL_KEYS)}")
        return v
