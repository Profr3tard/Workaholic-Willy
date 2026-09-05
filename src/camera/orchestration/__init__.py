"""Entry point for the rig-keyed frame provider and the errors it raises."""

from __future__ import annotations

from .frame_provider import FrameProvider, FrameProviderStateError, UnknownCameraRigError

__all__ = ["FrameProvider", "FrameProviderStateError", "UnknownCameraRigError"]