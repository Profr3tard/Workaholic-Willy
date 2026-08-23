"""Qwen3-VL as a *detector* — ``detect_all(bgr, prompt) -> [Detection]``, same shape as GroundingDINO."""

from __future__ import annotations

import time
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

import numpy as np

from src.models.constants import MODELS_LOG_DIR, VLM_GROUNDER_LOG_FILE
from src.models.detection.types import Detection
from src.utility.log_cfg import create_logger

from .parsing import CoordinateSpace, parse_grounding_response

__all__ = ["Qwen3VLGrounder", "GROUNDING_INSTRUCTION"]

#: A real file rather than the bare ``getLogger`` this module used to hold: nothing in this repo
#: configures the root logger, so the three lines below were formatted and then discarded. The logger
#: NAME stays ``__name__`` to match its siblings in this package -- ``availability.py`` has a test that
#: asserts on that name, and one naming rule across the route is worth more than matching the
#: class-named loggers elsewhere in ``models``. ``create_logger`` is stdlib-only, so the
#: no-torch-at-import promise in the docstring above still holds.
_LOG = create_logger(__name__, log_file=VLM_GROUNDER_LOG_FILE, log_dir=MODELS_LOG_DIR)

#: The instruction wrapped around every operator prompt.
#:
#: It asks for Qwen's documented ``bbox_2d`` grounding format and, deliberately, for an EMPTY LIST when
#: the object is absent. Instruct-tuned models are agreeable by default: without that sentence they
#: invent a plausible box rather than return nothing, and an invented box is a grasp at the wrong
#: place. "Return [] if absent" is the single most safety-relevant line in this file.
GROUNDING_INSTRUCTION = (
    "Locate every object matching this description and return ONLY a JSON array, no prose:\n"
    '[{{"bbox_2d": [x0, y0, x1, y1], "label": "<short name>"}}]\n'
    "Coordinates are absolute pixels in this image. "
    "If no object matches, return an empty array []. Do not guess.\n\n"
    "Description: {prompt}"
)


class Qwen3VLGrounder:
    """Ground a free-form prompt to boxes with a Qwen3-VL-Instruct checkpoint.

    Duck-types the detector contract (``detect_all``), so it drops straight into ``TwoStageBackend``.
    """

    def __init__(
        self,
        *,
        model_id: str,
        model_path: str | None = None,
        local: bool = False,
        device: str = "cuda",
        max_new_tokens: int = 512,
        preload: bool = False,
        coordinate_space: CoordinateSpace = CoordinateSpace.GRID_1000,
    ) -> None:
        #: What this checkpoint's box numbers mean.
        self.coordinate_space = coordinate_space
        self.model_id = model_id
        self._source = model_path or model_id
        self._local = local
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._model: Any = None
        self._processor: Any = None
        if preload:
            # Predictable latency and VRAM held from the start, versus nothing paid until a complex
            # prompt arrives. Config chooses; both are legitimate for different cells.
            self._ensure_loaded()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _ensure_loaded(self) -> None:
        """Load weights on first use. Raises the underlying error, the guard decides what it means."""
        if self._model is not None:
            return

        started = time.perf_counter()
        _LOG.info("loading VLM %s (device=%s)", self._source, self._device)
        self._processor = AutoProcessor.from_pretrained(self._source, local_files_only=self._local)
        model: Any = AutoModelForImageTextToText.from_pretrained(
            self._source,
            local_files_only=self._local,
            dtype="auto",
        )
        self._model = model.to(self._device)
        self._model.eval()
        self._torch = torch
        # The load cost is exactly what `preload` trades against first-prompt latency, and the
        # docstring above says that number is unmeasured. This is where it gets measured.
        _LOG.info("VLM %s ready in %.1f s", self.model_id, time.perf_counter() - started)

    def _generate(self, image_rgb: Any, prompt: str) -> str:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_rgb},
                {"type": "text", "text": GROUNDING_INSTRUCTION.format(prompt=prompt)},
            ],
        }]
        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self._model.device)

        with self._torch.inference_mode():
            generated = self._model.generate(
                **inputs, max_new_tokens=self._max_new_tokens, do_sample=False,
            )
        # Strip the prompt tokens: decoding the whole sequence would feed the instruction (which
        # CONTAINS a bbox_2d example) back into the parser and yield a fabricated detection.
        trimmed = generated[:, inputs["input_ids"].shape[1]:]
        return str(self._processor.batch_decode(trimmed, skip_special_tokens=True)[0])

    def detect_all(self, image_bgr: Any, prompt: str) -> list[Detection]:
        """Ground ``prompt``; ``[]`` when nothing matches. Same contract as the phrase detectors."""
        self._ensure_loaded()
        array = np.asarray(image_bgr)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"expected an HxWx3 BGR image, got shape {array.shape}")
        height, width = int(array.shape[0]), int(array.shape[1])
        # The processor expects RGB; every caller here speaks BGR. `ascontiguousarray` is load-bearing,
        # not tidiness: a bare `[..., ::-1]` is a NEGATIVE-STRIDE view, and torch refuses those with
        # "At least one stride in the given numpy array is negative" from deep inside the image
        # processor, several frames away from anything that mentions this file.
        image_rgb = np.ascontiguousarray(array[..., ::-1])

        # Timed around generate+parse together: that span is what a caller waits for, and it is the
        # number the "is the VLM route too slow for this cell" question is actually about.
        started = time.perf_counter()
        answer = self._generate(image_rgb, prompt)
        detections = parse_grounding_response(
            answer, image_width=width, image_height=height, fallback_label=prompt,
            space=self.coordinate_space,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if not detections:
            # Worth a line: "the model answered and nothing survived parsing" and "the model said the
            # object is absent" look identical downstream, and only the log distinguishes them.
            _LOG.info(
                "VLM grounded nothing for %r in %.0f ms (raw answer: %.200s)",
                prompt, elapsed_ms, answer,
            )
        else:
            # The other half of the same event, so exactly one line per inference either way.
            _LOG.info(
                "VLM grounded %d box(es) for %r in %.0f ms", len(detections), prompt, elapsed_ms,
            )
        return detections
