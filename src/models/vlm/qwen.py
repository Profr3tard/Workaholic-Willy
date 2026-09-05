"""Qwen3-VL as a detector: ``detect_all(bgr, prompt) -> [Detection]``, the shape GroundingDINO has.

This is what keeps the VLM route small. A grounding VLM produces boxes, and boxes are what
:class:`~src.models.perception_backend.TwoStageBackend` already knows how to hand to SAM2. So the
VLM is not a new kind of pipeline but a drop-in replacement for the detector stage, and the existing
two-stage backend composes it with the existing segmenter unchanged.

No torch, transformers or weights are imported at module import. Everything heavy is loaded inside
:meth:`Qwen3VLGrounder._ensure_loaded`, so this module imports on CI, on macOS, and on a box with no
GPU, and a cell that never sends a complex prompt never pays for the model.

Honesty bucket (3): nothing below has run against real weights. The prompt template, the parsing
contract and the failure paths are unit-tested against recorded and synthetic model output; the
grounding quality, the VRAM cost and the 4B-versus-8B choice are unmeasured until the on-box run,
and the config default must follow that measurement rather than the other way round.
"""

from __future__ import annotations

import time
from typing import Any

from src.models.constants import MODELS_LOG_DIR, VLM_GROUNDER_LOG_FILE
from src.models.detection.types import Detection
from src.utility.log_cfg import create_logger

from .parsing import CoordinateSpace, parse_grounding_response

__all__ = ["Qwen3VLGrounder", "GROUNDING_INSTRUCTION"]

#: A file sink rather than a bare ``getLogger``: nothing in this repo configures the root logger, so
#: the lines below would be formatted and then discarded. The name stays ``__name__`` to match its
#: siblings in this package rather than the class-named loggers elsewhere in ``models``.
#: ``create_logger`` is stdlib-only, so the no-torch-at-import promise above still holds.
_LOG = create_logger(__name__, log_file=VLM_GROUNDER_LOG_FILE, log_dir=MODELS_LOG_DIR)

#: The instruction wrapped around every operator prompt.
#:
#: It asks for Qwen's documented ``bbox_2d`` grounding format and, deliberately, for an empty list
#: when the object is absent. Instruct-tuned models are agreeable by default: without that sentence
#: they invent a plausible box rather than return nothing, and an invented box is a grasp at the
#: wrong place. "Return [] if absent" is the most safety-relevant line in this file.
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
        #: What this checkpoint's box numbers mean, measured for Qwen3-VL-4B-Instruct. Exposed as a
        #: parameter because a future checkpoint may differ and the failure is silent: grid values
        #: look like plausible pixels on a large frame. See :class:`CoordinateSpace`.
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
        """Load weights on first use. Raises the underlying error; the guard decides what it means."""
        if self._model is not None:
            return
        # Imported here, not at module scope: this module must import with no torch and no GPU.
        import torch  # noqa: PLC0415
        from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: PLC0415

        started = time.perf_counter()
        _LOG.info("loading VLM %s (device=%s)", self._source, self._device)
        self._processor = AutoProcessor.from_pretrained(self._source, local_files_only=self._local)
        # Not `device_map=`: that routes through `accelerate`, which is absent from the validated
        # Isaac environment and would fail at load with an error naming accelerate rather than the
        # real situation. `.to(device)` needs no extra dependency and is the correct call for a
        # single GPU anyway; device_map earns its keep only for multi-GPU sharding or CPU offload,
        # neither of which this cell does.
        # Typed as Any deliberately: transformers annotates `.to()` in a way that mypy reads as
        # taking a PreTrainedModel rather than a device string, and this module already treats the
        # model as an opaque handle.
        model: Any = AutoModelForImageTextToText.from_pretrained(
            self._source,
            local_files_only=self._local,
            dtype="auto",
        )
        self._model = model.to(self._device)
        self._model.eval()
        self._torch = torch
        # The load cost is what `preload` trades against first-prompt latency, so it is logged.
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
        # Strip the prompt tokens: decoding the whole sequence would feed the instruction, which
        # contains a bbox_2d example, back into the parser and yield a fabricated detection.
        trimmed = generated[:, inputs["input_ids"].shape[1]:]
        return str(self._processor.batch_decode(trimmed, skip_special_tokens=True)[0])

    def detect_all(self, image_bgr: Any, prompt: str) -> list[Detection]:
        """Ground ``prompt``; ``[]`` when nothing matches. Same contract as the phrase detectors."""
        self._ensure_loaded()
        import numpy as np  # noqa: PLC0415

        array = np.asarray(image_bgr)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"expected an HxWx3 BGR image, got shape {array.shape}")
        height, width = int(array.shape[0]), int(array.shape[1])
        # The processor expects RGB; every caller here speaks BGR. `ascontiguousarray` is load-bearing,
        # not tidiness: a bare `[..., ::-1]` is a negative-stride view, and torch refuses those with
        # "At least one stride in the given numpy array is negative" from deep inside the image
        # processor, several frames away from anything that mentions this file.
        image_rgb = np.ascontiguousarray(array[..., ::-1])

        # Timed around generate and parse together, because that span is what a caller waits for.
        started = time.perf_counter()
        answer = self._generate(image_rgb, prompt)
        detections = parse_grounding_response(
            answer, image_width=width, image_height=height, fallback_label=prompt,
            space=self.coordinate_space,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if not detections:
            # "The model answered and nothing survived parsing" and "the model said the object is
            # absent" look identical downstream, and only this line distinguishes them.
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
