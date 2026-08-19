"""JSON persistence for stereo calibration parameters.

Stores the small per-camera intrinsics / rectification parameters (never the
megabyte remap tables, those are recomputed on load inside
:class:`CalibrationResult`), schema-versioned like the extrinsics serializer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from src.calibration.exceptions import StereoCalibrationError
from src.calibration.stereo.config import CalibrationResult
from src.geometry.validation import validate_homogeneous_matrix

logger = logging.getLogger(__name__)

STEREO_CALIB_SCHEMA = "willy.calibration.stereo/1"

_PARAM_KEYS = ("camL", "distL", "rectL", "projL", "camR", "distR", "rectR", "projR", "Q")

__all__ = ["STEREO_CALIB_SCHEMA", "StereoCalibrationStore"]


class StereoCalibrationStore:
    """Read/write stereo calibration params (+ optional extrinsics) as JSON."""

    def save(
        self,
        filepath: str,
        result: CalibrationResult,
        T_cam_to_base: Optional[np.ndarray] = None,
    ) -> None:
        target = Path(filepath)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {
            "schema": STEREO_CALIB_SCHEMA,
            "frame_size": [int(result.frame_size[0]), int(result.frame_size[1])],
        }
        for key in _PARAM_KEYS:
            payload[key] = np.asarray(getattr(result, key), dtype=np.float64).tolist()
        if T_cam_to_base is not None:
            T_valid = validate_homogeneous_matrix(T_cam_to_base, name="T_cam_to_base")
            payload["T_cam_to_base"] = T_valid.tolist()
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, filepath: str) -> Tuple[CalibrationResult, Optional[np.ndarray]]:
        try:
            payload = json.loads(Path(filepath).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise StereoCalibrationError(f"cannot read stereo calibration '{filepath}': {exc}") from exc
        if payload.get("schema") != STEREO_CALIB_SCHEMA:
            raise StereoCalibrationError(
                f"stereo calibration schema mismatch: {payload.get('schema')!r} != {STEREO_CALIB_SCHEMA!r}"
            )
        try:
            p = {key: np.asarray(payload[key], dtype=np.float64) for key in _PARAM_KEYS}
            width, height = (int(v) for v in payload["frame_size"])
        except (KeyError, TypeError, ValueError) as exc:
            raise StereoCalibrationError(f"stereo calibration file has missing/invalid fields: {exc}") from exc

        result = CalibrationResult(
            camL=p["camL"], distL=p["distL"], rectL=p["rectL"], projL=p["projL"],
            camR=p["camR"], distR=p["distR"], rectR=p["rectR"], projR=p["projR"],
            Q=p["Q"], frame_size=(width, height),
        )

        raw_t = payload.get("T_cam_to_base")
        t_cam_to_base = (
            validate_homogeneous_matrix(np.asarray(raw_t, dtype=np.float64), name="T_cam_to_base")
            if raw_t is not None
            else None
        )
        return result, t_cam_to_base
