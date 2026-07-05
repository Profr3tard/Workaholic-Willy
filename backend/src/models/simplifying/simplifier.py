import time

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from backend.config.schema.models.models_schema import SimplifierConfig
from backend.src.models._inference import (
    autocast_ctx,
    build_load_kwargs,
    finalize_model,
)
from backend.src.models.constants import MODELS_LOG_DIR, SIMPLIFIER_LOG_FILE
from backend.src.utility.device import get_device, move_inputs_to_device
from backend.src.utility.log_cfg import create_logger


class TextSimplifier:

    def __init__(self, config: SimplifierConfig):
        self.device = get_device()
        self.max_length = config.max_length
        self.num_beams = config.num_beams
        self.no_repeat_ngram_size = config.no_repeat_ngram_size
        self.early_stopping = config.early_stopping
        self.optim = config.optim

        self.logger = create_logger("Simplifier", log_file=SIMPLIFIER_LOG_FILE, log_dir=MODELS_LOG_DIR)

        source = config.model_path if config.model_id is None else config.model_id
        model_kwargs = build_load_kwargs(self.optim, self.device)
        self._model_dtype = model_kwargs.get("torch_dtype")
        self.logger.info(
            "Initializing TextSimplifier: source='%s', device=%s, dtype=%s",
            source, self.device, self._model_dtype,
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(source)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                source, **model_kwargs
            ).to(self.device)
        except RuntimeError:
            self.logger.error("Failed to load simplifier model '%s'", source)
            raise

        self.model = finalize_model(self.model, self.device, self.optim)

        if self.device.type == "cpu":
            self.logger.warning("TextSimplifier loaded on CPU. Inference may be slow.")
        else:
            self.logger.info("TextSimplifier loaded on %s.", self.device.type.upper())

    @torch.inference_mode()
    def simplify(self, text: str, clean_up:bool=False) -> str:
        """
        Simplifies the given text using the seq2seq model.

        Args:
            text     : Input English text to simplify
            clean_up : Optional, removes extra prompts in output if True

        Returns:
            simplified string
        """
        start_time = time.time()

        # -------------------------------------------------
        # Prepare input prompt for the model
        # -------------------------------------------------
        # In this version, the prompt is just the raw text
        prompt = f"{text}"

        # Encode prompt to model input tensors
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = move_inputs_to_device(dict(inputs), self.device)

        # -------------------------------------------------
        # Generate simplified text
        # -------------------------------------------------
        with autocast_ctx(self.device, dtype=self._model_dtype):
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=self.num_beams,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                early_stopping=self.early_stopping,
            )

        # Decode tokens to string
        # decode() of a single 1-D sequence always returns str at runtime (list[str] is the batch
        # overload only); the str() makes that explicit for mypy and is a no-op on a real str (R9.0).
        simplified = str(self.tokenizer.decode(outputs[0], skip_special_tokens=True))

        # -------------------------------------------------
        # Optional cleanup
        # -------------------------------------------------
        if clean_up:
            # Remove everything before colon if present
            if ":" in simplified:
                simplified = simplified.split(":", 1)[1].strip()

        # Ensure text ends with a period
        if not simplified.endswith("."):
            simplified += "."

        # --- Log elapsed time and output ---
        elapsed = time.time() - start_time
        self.logger.info(f"Simplification completed in {elapsed:.3f}s")
        self.logger.info(f"Simplified text: {simplified}")

        return simplified
