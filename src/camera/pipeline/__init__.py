"""Entry point for the pipeline that turns a camera configuration into runtime objects."""

from __future__ import annotations

from .stereo_capture import StereoCapturePipeline

__all__ = ["StereoCapturePipeline"]