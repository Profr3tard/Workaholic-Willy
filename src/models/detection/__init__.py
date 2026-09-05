"""Object detection: shared :class:`Detection` type + zero-shot / closed-set backends.

The torch-heavy detector wrappers live under ``zero_shot`` (GroundingDINO,
open-vocabulary, prompt-driven) and ``closed_set`` (RT-DETR, fixed classes, no
prompt) and are imported from there directly; only the lightweight result type
is re-exported here so importing this package stays torch-free.
"""

from __future__ import annotations

from src.models.detection.types import Detection

__all__ = ["Detection"]
