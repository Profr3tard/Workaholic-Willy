"""App-runtime tuning: what the console encodes its frames at."""

from __future__ import annotations

from pydantic import Field

from ._base import StrictModel


class ImageEncodingConfig(StrictModel):
    """JPEG quality for the frames the console streams (1-100)."""
    frame_quality: int = Field(default=60, ge=1, le=100)


class RuntimeConfig(StrictModel):
    """Aggregated app-runtime tuning. Section in ``app/runtime.yaml``."""

    image_encoding: ImageEncodingConfig = Field(default_factory=ImageEncodingConfig)
