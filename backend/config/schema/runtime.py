"""App-runtime tuning schemas (event hub, decision cache, timeouts, …).

These values are operator-tunable but safety-irrelevant: they control
queue sizes, cache TTLs, encoding quality and prompt timeouts. Editing
them cannot move the robot.

Loaded from ``data/app/runtime.yaml`` (optional — schema defaults are
applied when the file is absent).
"""

from __future__ import annotations

from pydantic import Field, model_validator

from ._base import StrictModel


class EventHubConfig(StrictModel):
    """Tuning for the in-process event bus + WebSocket fan-out."""

    buffer_size: int = Field(default=512, ge=1)
    subscriber_queue_size: int = Field(default=256, ge=1)
    slow_subscriber_threshold: int = Field(default=3, ge=1)
    heartbeat_interval_s: float = Field(default=2.5, gt=0.0)


class DecisionImagesConfig(StrictModel):
    """In-memory cache for decision preview JPEGs."""

    capacity: int = Field(default=16, ge=1)
    ttl_s: float = Field(default=300.0, gt=0.0)


class InteractionConfig(StrictModel):
    """Operator-prompt and confirmation timeouts (seconds)."""

    prompt_timeout_s: float = Field(default=300.0, gt=0.0)
    confirm_timeout_s: float = Field(default=300.0, gt=0.0)
    confirm_watchdog_s: float = Field(default=30.0, gt=0.0)

    @model_validator(mode="after")
    def _watchdog_le_timeout(self) -> InteractionConfig:
        if self.confirm_watchdog_s > self.confirm_timeout_s:
            raise ValueError(
                f"confirm_watchdog_s ({self.confirm_watchdog_s}) must be "
                f"<= confirm_timeout_s ({self.confirm_timeout_s})"
            )
        return self


class RunRegistryConfig(StrictModel):
    """In-memory bookkeeping for completed pipeline runs."""

    recent_limit: int = Field(default=50, ge=1)


class ImageEncodingConfig(StrictModel):
    """JPEG quality used for streamed and persisted preview images (1-100)."""

    frame_quality: int = Field(default=60, ge=1, le=100)
    decision_quality: int = Field(default=75, ge=1, le=100)


class RuntimeConfig(StrictModel):
    """Aggregated app-runtime tuning. Section in ``app/runtime.yaml``."""

    event_hub: EventHubConfig = Field(default_factory=EventHubConfig)
    decision_images: DecisionImagesConfig = Field(default_factory=DecisionImagesConfig)
    interaction: InteractionConfig = Field(default_factory=InteractionConfig)
    run_registry: RunRegistryConfig = Field(default_factory=RunRegistryConfig)
    image_encoding: ImageEncodingConfig = Field(default_factory=ImageEncodingConfig)
