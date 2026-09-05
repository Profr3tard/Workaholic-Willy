"""Object detection: the shared :class:`Detection` type and the backend packages.

The torch-heavy wrappers live in ``zero_shot`` (GroundingDINO, open vocabulary,
prompt-driven) and ``closed_set`` (RT-DETR, fixed classes, no prompt), and are
imported from there directly. Only :class:`Detection` is re-exported here, which
keeps an import of this package free of torch.
"""

from __future__ import annotations

from src.models.detection.types import Detection

__all__ = ["Detection"]
