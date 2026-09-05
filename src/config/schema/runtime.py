"""App-runtime tuning: what the console encodes its frames at, and nothing else.

The one block here is operator-tunable and safety-irrelevant: picture quality. Editing it cannot
move the robot.

Loaded from ``data/app/runtime.yaml`` (optional: schema defaults are applied when the file is
absent).
"""

from __future__ import annotations

from pydantic import Field

from ._base import StrictModel


class ImageEncodingConfig(StrictModel):
    """JPEG quality for the frames the console streams (1-100)."""

    #: Quality of each viewfinder frame, read by ``GET /v1/camera``. Lower trades legibility for
    #: bandwidth; measured at 1280x720 the whole encode costs 1.0 ms and 53 KB at the default 60,
    #: so this is a picture-quality knob rather than a performance one.
    frame_quality: int = Field(default=60, ge=1, le=100)


class RuntimeConfig(StrictModel):
    """Aggregated app-runtime tuning. Section in ``app/runtime.yaml``."""

    image_encoding: ImageEncodingConfig = Field(default_factory=ImageEncodingConfig)
