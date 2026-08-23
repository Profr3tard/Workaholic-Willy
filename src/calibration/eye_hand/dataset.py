"""Eye-hand sample dataset and deterministic JSON persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from src.geometry.validation import validate_homogeneous_matrix

from src.calibration.constants import CALIBRATION_LOG_DIR, EYE_HAND_DATASET_LOG_FILE
from src.calibration.exceptions import CalibrationDataError
from src.utility.log_cfg import create_logger

logger = create_logger("EyeHandDataset", EYE_HAND_DATASET_LOG_FILE, log_dir=CALIBRATION_LOG_DIR)

__all__ = [
    "EYE_HAND_DATASET_SCHEMA",
    "EyeHandDataset",
    "EyeHandSample",
]

EYE_HAND_DATASET_SCHEMA = "willy.calibration.eye_hand.dataset/1"


def _as_valid_transform(value: object, *, name: str) -> np.ndarray:
    try:
        matrix = validate_homogeneous_matrix(value, name=name).astype(np.float64, copy=True)
    except Exception as exc:
        raise CalibrationDataError(f"{name} is not a valid homogeneous transform") from exc
    matrix.setflags(write=False)
    return matrix


@dataclass(frozen=True, slots=True)
class EyeHandSample:
    """One paired robot pose and marker pose measurement."""

    T_base_to_tool: np.ndarray
    T_cam_to_marker: np.ndarray
    marker_id: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "T_base_to_tool",
            _as_valid_transform(self.T_base_to_tool, name="T_base_to_tool"),
        )
        object.__setattr__(
            self,
            "T_cam_to_marker",
            _as_valid_transform(self.T_cam_to_marker, name="T_cam_to_marker"),
        )
        if not isinstance(self.marker_id, int) or isinstance(self.marker_id, bool):
            raise CalibrationDataError("marker_id must be an integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "marker_id": int(self.marker_id),
            "T_base_to_tool": self.T_base_to_tool.reshape(16).tolist(),
            "T_cam_to_marker": self.T_cam_to_marker.reshape(16).tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EyeHandSample:
        try:
            return cls(
                marker_id=int(data.get("marker_id", 0)),
                T_base_to_tool=np.asarray(data["T_base_to_tool"], dtype=np.float64).reshape(4, 4),
                T_cam_to_marker=np.asarray(data["T_cam_to_marker"], dtype=np.float64).reshape(4, 4),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationDataError("invalid eye-hand sample payload") from exc


class EyeHandDataset:
    """Container for ordered eye-hand calibration samples."""

    def __init__(self, samples: list[EyeHandSample] | None = None) -> None:
        self._samples: list[EyeHandSample] = list(samples or [])

    def add_sample(self, sample: EyeHandSample) -> None:
        self._samples.append(sample)

    def add(
        self,
        T_base_to_tool: np.ndarray,
        T_cam_to_marker: np.ndarray,
        marker_id: int = 0,
    ) -> None:
        self.add_sample(
            EyeHandSample(
                T_base_to_tool=T_base_to_tool,
                T_cam_to_marker=T_cam_to_marker,
                marker_id=marker_id,
            )
        )

    def iter_samples(self) -> Iterator[EyeHandSample]:
        yield from self._samples

    def base_to_tool_matrices(self) -> list[np.ndarray]:
        return [sample.T_base_to_tool for sample in self._samples]

    def cam_to_marker_matrices(self) -> list[np.ndarray]:
        return [sample.T_cam_to_marker for sample in self._samples]

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": EYE_HAND_DATASET_SCHEMA,
            "samples": [sample.to_dict() for sample in self._samples],
        }
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(target)
        logger.info(
            "Dataset written: %s (%d samples, %d bytes).",
            target,
            len(self._samples),
            target.stat().st_size,
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> EyeHandDataset:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CalibrationDataError(f"{path!s} is not valid JSON") from exc
        if not isinstance(data, dict):
            raise CalibrationDataError("eye-hand dataset JSON must be an object")
        schema = data.get("schema")
        if schema != EYE_HAND_DATASET_SCHEMA:
            raise CalibrationDataError(
                f"expected dataset schema {EYE_HAND_DATASET_SCHEMA!r}, got {schema!r}"
            )
        samples = data.get("samples")
        if not isinstance(samples, list):
            raise CalibrationDataError("eye-hand dataset must contain a samples list")
        dataset = cls([EyeHandSample.from_dict(sample) for sample in samples])
        logger.info("Dataset loaded: %s (%d samples, schema %s).", path, len(dataset), schema)
        return dataset

    def __len__(self) -> int:
        return len(self._samples)