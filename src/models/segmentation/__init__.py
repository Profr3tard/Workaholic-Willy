"""Segmentation: the shared :class:`SegmentationResult` type and two backends.

The torch-heavy segmenter wrappers live under ``realtime`` (SAM2, box-prompted,
fast) and ``research`` (OneFormer, universal, higher-accuracy, GPU-heavy) and are
imported from there directly. Only the lightweight result type is re-exported
here, so importing this package stays torch-free.
"""

from __future__ import annotations

from src.models.segmentation.types import SegmentationResult

__all__ = ["SegmentationResult"]
