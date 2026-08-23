"""Live-camera perception: an RGB-D stream -> GroundingDINO -> SAM2 -> one :class:`PerceptionFrame`."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from src.robot.grasping.types.perception import PerceptionFrame
from src.robot.perception.mask_completion import (
    DEFAULT_MASK_COMPLETION,
    MaskCompletion,
    complete_mask,
)

__all__ = ["RealSenseVisionPerceptionSource"]


class RealSenseVisionPerceptionSource:
    """Grab one RGB-D frame, ground + segment the prompted object(s), emit a :class:`PerceptionFrame`.

    Parameters
    ----------
    streamer
        Anything with ``grab() -> RGBDFrame`` (``.color`` BGR uint8, ``.depth`` uint16 millimetres) and
        ``get_intrinsics() -> 3x3 K | None``. In production this is
        ``camera.setup.image_taking.rgbd.RealSenseRGBDStreamer`` (already complete); a fake satisfies it
        in tests.
    detector, segmenter
        ``detector.detect_all(bgr, prompt) -> [Detection]`` and
        ``segmenter.segment_detection(bgr, det) -> SegmentationResult`` (``.mask`` HxW, ``.label`` str).
    prompt
        The GroundingDINO phrase(s). A multi-phrase prompt grounds every object (the neighbour clutter
        the dense sampler needs), not only the target.
    object_labels
        Optional canonical labels (the known object names). When given, each detection's free-form GDINO
        label is mapped to the nearest canonical one, so an EXACT ``seg.label == target`` match works
        downstream.
    intrinsics
        Optional 3x3 K override. The **D1 seam**: leave ``None`` to use the streamer's factory K (the
        D435 ships calibrated), or pass a bench-calibrated K to override it. The source records which
        was used on the frame's provenance via ``intrinsics_source``.
    grasp_top_penetration_mm
        Bias the grasp depth this far BELOW each object's nearest (top) surface. A single top-down view
        sees only the top; gripping the very edge slips, so the grasp is referenced to the top + this.
    warmup_grabs
        Throwaway grabs before the real one, so a physical camera's auto-exposure / auto-white-balance
        has settled. A real RealSense needs a handful; a fake ignores it.
    """

    def __init__(
        self,
        *,
        streamer: Any,
        detector: Any = None,
        segmenter: Any = None,
        backend: Any = None,
        prompt: str,
        object_labels: tuple[str, ...] = (),
        intrinsics: np.ndarray | None = None,
        grasp_top_penetration_mm: float = 3.0,
        warmup_grabs: int = 5,
        mask_completion: MaskCompletion = DEFAULT_MASK_COMPLETION,
    ) -> None:
        # Either a ready-made perception backend, or the detector+segmenter pair this source has always
        # taken -- which it now composes into the same backend rather than running the two-stage chain
        # itself. The pair is kept because callers that hand-assemble models are a supported path, and
        # because it keeps every existing construction site working unchanged.
        if backend is None:
            if detector is None or segmenter is None:
                raise ValueError(
                    "RealSenseVisionPerceptionSource needs either backend=..., or both detector=... "
                    "and segmenter=..."
                )
            from src.models.perception_backend import TwoStageBackend

            backend = TwoStageBackend(detector=detector, segmenter=segmenter)
        self._streamer = streamer
        self._backend = backend
        self._prompt = prompt
        self._object_labels = tuple(object_labels)
        self._intrinsics_override = None if intrinsics is None else np.asarray(intrinsics, dtype=np.float64)
        self._grasp_top_penetration_mm = float(grasp_top_penetration_mm)
        self._warmup_grabs = max(0, int(warmup_grabs))
        #: What to do with a mask that underfills its detection box. The default is the shipped
        #: behaviour; `mask_completion.py` carries the measurement that says why it is a lever.
        self._mask_completion = MaskCompletion(mask_completion)
        #: "override" if a calibrated K was supplied, else "factory" (set on acquire). D1 provenance.
        self.intrinsics_source = "override" if intrinsics is not None else "factory"

    # ------------------------------------------------------------------ label canonicalisation
    def _canonical_label(self, gdino_label: str) -> str:
        """Map a free-form GDINO phrase to the nearest known object label (identity if none given)."""
        if not self._object_labels:
            return gdino_label
        gl = gdino_label.lower().strip()
        for c in self._object_labels:
            cl = c.lower()
            if cl and (cl in gl or gl in cl):
                return c
        gw = set(gl.split())
        best, best_n = gdino_label, 0
        for c in self._object_labels:
            n = len(gw & set(c.lower().split()))
            if n > best_n:
                best, best_n = c, n
        return best

    # ------------------------------------------------------------------ mask robustness
    def _maybe_fill_to_detection_box(self, mask: np.ndarray, det: Any) -> np.ndarray:
        """Apply the configured mask-completion policy."""
        return complete_mask(mask, det, policy=self._mask_completion)

    # ------------------------------------------------------------------ the frame
    #: These pixels came off a physical device, so the console may say "camera". See
    #: `perception/viewfinder.py` for why this is declared rather than inferred.
    colour_source_kind = "camera"

    def peek_color(self) -> np.ndarray | None:
        """One colour frame. BGR uint8."""
        grabbed = self._streamer.grab()
        colour = getattr(grabbed, "color", None)
        if colour is None:
            return None
        return np.ascontiguousarray(np.asarray(colour))

    def close(self) -> None:
        """Release the camera. Idempotent, and it never raises."""
        release = getattr(self._streamer, "release", None)
        if callable(release):
            try:
                release()
            except Exception:  # noqa: BLE001 - teardown reports nothing it can fix
                pass

    def acquire(self) -> PerceptionFrame:
        for _ in range(self._warmup_grabs):
            self._streamer.grab()  # discard: let auto-exposure / white-balance settle on real hardware

        rgbd = self._streamer.grab()
        bgr = np.ascontiguousarray(np.asarray(rgbd.color))          # detector/segmenter take OpenCV BGR
        depth_mm = np.asarray(rgbd.depth, dtype=np.float64)         # uint16 mm -> float; 0 == hole
        rendered_depth_mm = depth_mm.copy()                         # true surface, read for top-referencing

        if self._intrinsics_override is not None:
            intrinsics = self._intrinsics_override.copy()
        else:
            k = self._streamer.get_intrinsics()
            if k is None:
                raise RuntimeError(
                    "streamer.get_intrinsics() returned None open() the streamer before acquire(), or "
                    "pass a calibrated `intrinsics=` (the D1 override)."
                )
            intrinsics = np.asarray(k, dtype=np.float64)

        segmentations: list[Any] = []
        # The detector/segmenter chain, and its two failure rules, now live in the backend: a detector
        # error yields no objects (an honest no_valid_grasp downstream), a single segmentation error
        # skips that object and keeps the rest.
        perceived = self._backend.perceive(bgr, self._prompt)

        for obj in perceived:
            det, seg = obj.detection, obj.segmentation
            label = self._canonical_label(getattr(seg, "label", "") or "")
            mask = self._maybe_fill_to_detection_box(np.asarray(seg.mask).astype(bool), det)
            seg = replace(seg, label=label, mask=mask.astype(np.uint8))
            # Top-referenced depth: read the nearest REAL surface over the mask, skipping D435 holes (0),
            # and overlay grasp_depth = that top + penetration.
            if mask.any():
                vals = rendered_depth_mm[mask]
                vals = vals[vals > 0.0]
                if vals.size:
                    grasp_depth_mm = float(np.min(vals)) + self._grasp_top_penetration_mm
                    depth_mm = np.where(mask, grasp_depth_mm, depth_mm)
            segmentations.append(seg)

        rgb = bgr[..., ::-1]  # BGR -> RGB for any debugging consumer (no reader in robot/ today)
        return PerceptionFrame(
            depth_map=depth_mm, intrinsics=intrinsics, segmentations=tuple(segmentations), rgb=rgb
        )
