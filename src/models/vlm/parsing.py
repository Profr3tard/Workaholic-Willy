"""Turn a vision-language model's free-form answer into validated :class:`Detection` boxes.

This is the load-bearing half of the VLM route and the only half that can be tested without a GPU. A
grounding VLM does not return a tensor; it returns **text that is supposed to contain JSON**, and every
part of that sentence is doing work: it is text (so it can carry prose, markdown fences, or an apology),
it is supposed to (so it can be malformed), and it contains JSON (so the JSON can be well-formed and
still describe a nonsensical box).

The contract this parses is Qwen's grounding format --

    [{"bbox_2d": [x0, y0, x1, y1], "label": "red cube"}, ...]

in whichever **coordinate space** the model was trained to emit. That space is an explicit,
named parameter (:class:`CoordinateSpace`) and never a guess, because guessing it wrong is silent.
"""

from __future__ import annotations

import json
import math
import re
from enum import StrEnum
from typing import Any

from src.models.constants import MODELS_LOG_DIR, VLM_PARSING_LOG_FILE
from src.models.detection.types import Detection
from src.utility.log_cfg import create_logger

#: The parser's own file, separate from the model wrapper's. ``qwen.py`` logs WHAT the model answered;
#: these lines log which boxes this parser refused and why and every refusal here is a grasp pose
#: that will not happen, so "the model said nothing" and "the model said something unusable" must not
#: look the same afterwards. Module scope because this module is functions, not a class.
_LOG = create_logger("VLMGroundingParser", log_file=VLM_PARSING_LOG_FILE, log_dir=MODELS_LOG_DIR)

__all__ = [
    "VLM_NOMINAL_SCORE",
    "CoordinateSpace",
    "GRID_SIZE",
    "parse_grounding_response",
    "extract_json_payload",
]

#: The normalised grid Qwen grounds on.
GRID_SIZE = 1000.0


class CoordinateSpace(StrEnum):
    """What a model's box numbers MEAN. Declared per backend, never inferred per box."""

    ABSOLUTE = "absolute"    #: pixels of the image as submitted
    GRID_1000 = "grid_1000"  #: Qwen's 0-1000 normalised grid (MEASURED for Qwen3-VL, see module docs)

#: The score stamped on every VLM detection.
VLM_NOMINAL_SCORE = 1.0

#: ```json ... ``` or bare ``` ... ``` fences, which instruct-tuned models add unprompted.
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

#: The outermost JSON array or object embedded in prose ("Here are the objects: [...]. Let me know...").
_ARRAY = re.compile(r"\[.*\]", re.DOTALL)
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

#: Keys seen in the wild for the box field. ``bbox_2d`` is Qwen's documented name; the others are
#: accepted because they cost one tuple entry and the alternative is dropping a perfectly good box.
_BOX_KEYS = ("bbox_2d", "bbox", "box_2d", "box")
_LABEL_KEYS = ("label", "text", "name", "category")


def extract_json_payload(text: str) -> Any | None:
    """Recover the JSON value from a model answer, or ``None`` if there is none."""
    if not isinstance(text, str) or not text.strip():
        return None

    candidates: list[str] = [text.strip()]
    fenced = _FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1))
    for pattern in (_ARRAY, _OBJECT):
        found = pattern.search(text)
        if found:
            candidates.append(found.group(0))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (ValueError, TypeError):
            continue
    return None


def _as_items(payload: Any) -> list[dict[str, Any]]:
    """Normalise the payload to a list of dicts. A lone object is a one-element list."""
    if isinstance(payload, dict):
        # Some answers wrap the list: {"objects": [...]} / {"detections": [...]}
        for key in ("objects", "detections", "results", "boxes", "items"):
            nested = payload.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _box_from(item: dict[str, Any]) -> list[float] | None:
    """The four corner values, or ``None`` if this item has no usable box."""
    for key in _BOX_KEYS:
        raw = item.get(key)
        if raw is None:
            continue
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            continue
        try:
            values = [float(v) for v in raw]
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in values):
            continue
        return values
    return None


def _label_from(item: dict[str, Any], fallback: str) -> str:
    for key in _LABEL_KEYS:
        raw = item.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return fallback


def parse_grounding_response(
    text: str,
    *,
    image_width: int,
    image_height: int,
    fallback_label: str,
    space: CoordinateSpace = CoordinateSpace.GRID_1000,
) -> list[Detection]:
    """Validated detections from a grounding answer. Never raises; returns ``[]`` when nothing survives."""
    payload = extract_json_payload(text)
    if payload is None:
        # Degraded, not empty: the model DID answer, and none of it was JSON. Downstream this is
        # indistinguishable from "no such object", so the raw answer is the only way to tell an
        # apology apart from a hallucinated prose location.
        _LOG.warning("no JSON payload in grounding answer for %r (raw: %.200s)", fallback_label, text)
        return []

    detections: list[Detection] = []
    # Drops are counted per reason and reported ONCE below, not per item: a wrong coordinate space
    # drops every box in the answer, and twenty identical lines say no more than one line with a
    # count, while making the interesting case (a few boxes lost out of many) harder to spot.
    dropped_no_box = 0
    dropped_degenerate = 0
    for item in _as_items(payload):
        values = _box_from(item)
        if values is None:
            dropped_no_box += 1
            continue
        if space is CoordinateSpace.GRID_1000:
            # Scale BEFORE clamping: clamping first would pin grid values against the image bounds and
            # silently flatten every box in the right half of a wide frame.
            values = [
                values[0] / GRID_SIZE * image_width,
                values[1] / GRID_SIZE * image_height,
                values[2] / GRID_SIZE * image_width,
                values[3] / GRID_SIZE * image_height,
            ]
        x0, y0, x1, y1 = values
        # Inverted corners have exactly one possible intent repair. Everything else is a guess.
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        # Clamp into the frame, then re-check: a box entirely outside collapses here and is dropped
        # below, which is the correct outcome for a hallucinated location.
        x0 = min(max(x0, 0.0), float(image_width))
        x1 = min(max(x1, 0.0), float(image_width))
        y0 = min(max(y0, 0.0), float(image_height))
        y1 = min(max(y1, 0.0), float(image_height))
        if x1 - x0 < 1.0 or y1 - y0 < 1.0:
            # Sub-pixel after clamping: an off-frame box, or normalised coordinates we deliberately do
            # not rescale. Either way there is no box here that a gripper should be sent to.
            dropped_degenerate += 1
            continue
        detections.append(Detection(
            box=[x0, y0, x1, y1],
            x_center=(x0 + x1) / 2.0,
            y_center=(y0 + y1) / 2.0,
            label=_label_from(item, fallback_label),
            score=VLM_NOMINAL_SCORE,
        ))

    if dropped_no_box or dropped_degenerate:
        # WARNING because each of these was a box the model asserted and this parser refused.
        _LOG.warning(
            "kept %d box(es), dropped %d (no usable box) + %d (degenerate after clamp) "
            "for %r in %dx%d, space=%s",
            len(detections), dropped_no_box, dropped_degenerate,
            fallback_label, image_width, image_height, space.value,
        )
    else:
        # DEBUG, not INFO: on the clean path the caller (`Qwen3VLGrounder.detect_all`) already
        # reports the outcome; this line only adds the detail behind it.
        _LOG.debug(
            "parsed %d box(es) for %r in %dx%d, space=%s",
            len(detections), fallback_label, image_width, image_height, space.value,
        )
    return detections
