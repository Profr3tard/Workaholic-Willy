from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from backend.config.schema.models.models_schema import ObjectDetectorConfig
from backend.src.models._inference import (
    autocast_ctx,
    build_load_kwargs,
    finalize_model,
)
from backend.src.models.constants import DETECTOR_LOG_FILE, MODELS_LOG_DIR
from backend.src.models._helpers import Detection
from backend.src.utility import (
    bgr_to_rgb,
    create_logger,
    debug_dir,
    get_device,
    move_inputs_to_device,
)


class GroundingDinoObjectDetector:
    """
    Object detector using GroundingDINO for text-guided detection.
    Parameters:
        - model_id : Hugging Face model ID (e.g., "ShilongLi/groundingdino-v1.3")
        - model_path : Local path to model files (if local=True)
        - threshold : Confidence threshold for detections
        - local : Whether to load model from local path
        - debug_images : Whether to save debug images with detections drawn

    Output:
        - Bounding box (pixel coordinates)
        - Box center (pixel coordinates)
        - Detected label
    """

    def __init__(self, config: ObjectDetectorConfig, debug_images: bool = False):
        self.device = get_device()
        self.model_id = config.model_id
        self.model_path = config.model_path
        self.threshold = config.threshold
        self.local = config.local
        self.optim = config.optim
        self.debug_images = debug_images

        self.logger = create_logger("GroundingDINOObjectDetector", log_file=DETECTOR_LOG_FILE, log_dir=MODELS_LOG_DIR)

        source = self.model_path if self.local else self.model_id
        if not source:
            raise ValueError("model_path (local=True) or model_id (local=False) must be set.")
        base_kwargs = {"local_files_only": True} if self.local else {}
        proc_kwargs = dict(base_kwargs)
        model_kwargs = build_load_kwargs(self.optim, self.device, base_kwargs)
        self._model_dtype = model_kwargs.get("torch_dtype")

        self.logger.info(
            f"Initializing GroundingDINOObjectDetector: source='{source}', "
            f"threshold={self.threshold}, device={self.device}, "
            f"dtype={self._model_dtype}, attn={model_kwargs.get('attn_implementation')}"
        )

        try:
            self.processor = AutoProcessor.from_pretrained(source, **proc_kwargs)
            self.model = (
                AutoModelForZeroShotObjectDetection
                .from_pretrained(source, **model_kwargs)
                .to(self.device)
            )
        except RuntimeError:
            self.logger.error(f"Failed to load model '{source}'")
            raise

        self.model = finalize_model(
            self.model, self.device, self.optim, vision=True,
        )

        device_type = self.device.type
        if device_type == "cpu":
            self.logger.warning(f"Dino Model '{source}' loaded on CPU. Inference may be slow.")
        else:
            self.logger.info(f"Dino Model '{source}' loaded on {device_type.upper()}.")

    @torch.inference_mode()
    def detect(self, image: np.ndarray, prompt: str) -> Detection:
        """
        Runs text-guided object detection on a BGR image.

        Parameters:
            image  : Input image as BGR numpy array (from OpenCV).
            prompt : Natural language description of the target object.

        Returns:
            Detection with box [x0, y0, x1, y1] in pixels, center, label, score.
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None.")
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        start_time = time.time()

        pil_image = Image.fromarray(bgr_to_rgb(image))
        w, h = pil_image.size

        inputs = self.processor(
            images=pil_image,
            text=prompt,
            return_tensors="pt",
        )
        inputs = move_inputs_to_device(dict(inputs), self.device)

        if self.optim and self.optim.channels_last and self.device.type == "cuda":
            pv = inputs.get("pixel_values")
            if torch.is_tensor(pv) and pv.dim() == 4:
                inputs["pixel_values"] = pv.to(memory_format=torch.channels_last)

        with autocast_ctx(self.device, dtype=self._model_dtype):
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.threshold
        )

        if not results or len(results[0]["scores"]) == 0:
            self.logger.info(f"No detections above threshold {self.threshold:.2f} for prompt: '{prompt}'")
            raise ValueError(f"No object detected with description: {prompt}")

        result = results[0]
        boxes = result["boxes"]
        labels = result.get("text_labels", result.get("labels"))
        scores = result["scores"]

        best_idx = scores.argmax()
        box = boxes[best_idx]
        label = labels[best_idx]
        score = scores[best_idx]

        # post_process_grounded_object_detection was called WITHOUT
        # ``target_sizes`` above, so it returns boxes NORMALISED to [0, 1].
        # We scale by the image (width, height) to recover pixel coordinates.
        # Verified against real-hardware output — the scaling is correct; the
        # `_px` values below ARE pixels (do not remove the *w/*h).
        x0, y0, x1, y1 = [float(c) for c in box]

        x0_px, y0_px, x1_px, y1_px = x0 * w, y0 * h, x1 * w, y1 * h
        box_px = [x0_px, y0_px, x1_px, y1_px]

        x_center = (x0_px + x1_px) / 2
        y_center = (y0_px + y1_px) / 2
        score_val = float(score.item()) if hasattr(score, "item") else float(score)

        detection = Detection(
            box=box_px,
            x_center=x_center,
            y_center=y_center,
            label=label,
            score=score_val,
        )

        self._log_detection(detection, prompt)

        if self.debug_images:
            self._draw_detected_obj(pil_image, detection)

        inference_time = time.time() - start_time
        self.logger.info(f"Inference completed in {inference_time:.3f}s")

        return detection

    @torch.inference_mode()
    def detect_all(self, image: np.ndarray, prompt: str) -> list[Detection]:
        """Like :meth:`detect` but returns ALL detections above threshold (not just the argmax).

        For multi-object / clutter selection: each :class:`Detection` carries its own box (pixels),
        center, label and score. Returns an empty list (not a raise) when nothing clears the threshold,
        so callers can handle 'no objects' without exception flow. ``detect`` is left unchanged.
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None.")
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        pil_image = Image.fromarray(bgr_to_rgb(image))
        w, h = pil_image.size
        inputs = self.processor(images=pil_image, text=prompt, return_tensors="pt")
        inputs = move_inputs_to_device(dict(inputs), self.device)
        if self.optim and self.optim.channels_last and self.device.type == "cuda":
            pv = inputs.get("pixel_values")
            if torch.is_tensor(pv) and pv.dim() == 4:
                inputs["pixel_values"] = pv.to(memory_format=torch.channels_last)
        with autocast_ctx(self.device, dtype=self._model_dtype):
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"], threshold=self.threshold
        )
        if not results or len(results[0]["scores"]) == 0:
            return []
        result = results[0]
        boxes = result["boxes"]
        labels = result.get("text_labels", result.get("labels"))
        scores = result["scores"]
        dets: list[Detection] = []
        for i in range(len(scores)):
            x0, y0, x1, y1 = [float(c) for c in boxes[i]]
            box_px = [x0 * w, y0 * h, x1 * w, y1 * h]
            s = scores[i]
            # GDINO can ground a box to an empty/None phrase -> coerce to a non-empty label string.
            lbl = str(labels[i]).strip() if labels[i] is not None else ""
            dets.append(Detection(
                box=box_px, x_center=(box_px[0] + box_px[2]) / 2, y_center=(box_px[1] + box_px[3]) / 2,
                label=lbl or "object", score=float(s.item()) if hasattr(s, "item") else float(s),
            ))
        return dets

    def _log_detection(self, det: Detection, prompt: str):
        self.logger.info(
            f"\n{'='*40}\n"
            f"Prompt      : {prompt}\n"
            f"Label       : {det.label}\n"
            f"Score       : {det.score:.3f}\n"
            f"Box Pixel   : x0={det.box[0]:.1f}, y0={det.box[1]:.1f}, x1={det.box[2]:.1f}, y1={det.box[3]:.1f}\n"
            f"Center      : x={det.x_center:.1f}, y={det.y_center:.1f}\n"
            f"{'='*40}"
        )

    def _draw_detected_obj(self, image: Image.Image, det: Detection):
        draw = ImageDraw.Draw(image)

        font: ImageFont.FreeTypeFont | ImageFont.ImageFont
        try:
            font = ImageFont.truetype("arialbd.ttf", 20)
        except OSError:
            font = ImageFont.load_default()

        x0, y0, x1, y1 = det.box

        draw.rectangle([x0, y0, x1, y1], outline="red", width=4)

        text_to_draw = f"{det.label} ({det.score:.1%})"

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                draw.text(
                    (x0 + dx, y0 + dy),
                    text_to_draw,
                    font=font,
                    fill="black"
                )

        draw.text(
            (x0, y0),
            text_to_draw,
            font=font,
            fill="white"
        )

        # Save image with timestamp into the rotated logs/debug/detection bucket.
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"detection_debug_{now}.png"
        output_path = debug_dir("detection", max_files=200) / file_name

        image.save(output_path)
