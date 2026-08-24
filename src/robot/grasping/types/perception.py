"""Perception snapshot value objects shared across the grasping stack."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


class SegmentationLike(Protocol):
    """Minimal segmentation boundary: only a ``mask`` (HxW bool/uint8 array) is required."""

    mask: np.ndarray


@dataclass(frozen=True, slots=True)
class PerceptionFrame:
    """Single snapshot of the bin from the perception system."""

    depth_map: np.ndarray
    intrinsics: np.ndarray
    segmentations: tuple[SegmentationLike, ...]
    rgb: np.ndarray | None = None
    timestamp: float | None = None


@runtime_checkable
class PerceptionSource(Protocol):
    """Anything that can produce a :class:`PerceptionFrame` on demand."""

    def acquire(self) -> PerceptionFrame: ...


@dataclass(frozen=True, slots=True)
class CameraObservation:
    """One named camera's frame, from a rig of several."""

    camera_id: str
    frame: PerceptionFrame


@runtime_checkable
class MultiCameraPerceptionSource(Protocol):
    """Protocol for a synchronized multi-camera rig.

    Provides frames from multiple cameras captured as close together as the
    hardware allows, each with a timestamp, for consistent multi-view fusion.
    Unavailable cameras are omitted rather than replaced with empty frames;
    callers handle missing cameras according to their configured policy.
    """

    def acquire_all(self) -> tuple[CameraObservation, ...]: ...


@dataclass(frozen=True, slots=True)
class MappedCameraRig:
    """Adapter that combines named single-camera sources into a multi-camera source.

    Preserves insertion order for deterministic observation ordering. Cameras are
    acquired sequentially, so this adapter is suitable for stepped simulators or
    settled scenes, but not for moving parts requiring synchronized capture.
    Hardware-triggered rigs should implement ``MultiCameraPerceptionSource``
    directly.
    """

    sources: Mapping[str, PerceptionSource]

    def acquire_all(self) -> tuple[CameraObservation, ...]:
        return tuple(
            CameraObservation(camera_id=name, frame=source.acquire())
            for name, source in self.sources.items()
        )
