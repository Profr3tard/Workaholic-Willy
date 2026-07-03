"""UR driver connection config schema."""

from __future__ import annotations


from pydantic import Field

from .._base import StrictModel


class URConfig(StrictModel):
    """UR-specific connection settings.

    Only consulted when ``robot.vendor == "ur"``. Replaces the older
    flat ``connection:`` block; vendor blocks are now siblings under
    ``robot:`` so each driver owns its own typed surface.
    """

    ip: str = Field(default="192.168.1.100", min_length=1)
    rtde_frequency: float = Field(default=0.0, ge=0.0)