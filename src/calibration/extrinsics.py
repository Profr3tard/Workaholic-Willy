"""
:class:`Extrinsics` — the typed, vendor-neutral output of an eye-to-hand
calibration solve.

Wraps a :class:`~src.geometry.transform.Transform` whose direction
is fixed at ``Frame.CAMERA -> Frame.BASE`` (translation in millimetres),
together with the validation metadata callers need to decide whether the
calibration is trustworthy.

The dataclass is frozen + slotted, frame-checked, and self-classifying.
Once constructed it is safe to share across threads or persist via
:mod:`src.calibration.serialization`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from src.geometry import (
    Frame,
    Pose,
    Transform,
)

from .exceptions import ExtrinsicsError
from .quality import (
    DEFAULT_BANDS_MM,
    QUALITY_LABELS,
    QualityBandsMm,
    QualityLabel,
    classify_rmse,
)

__all__ = ["EXTRINSICS_SCHEMA", "Extrinsics"]


EXTRINSICS_SCHEMA: str = "willy.calibration.extrinsics/1"


@dataclass(frozen=True, slots=True)
class Extrinsics:
    """Typed result of an eye-to-hand calibration.

    Attributes
    ----------
    transform :
        Rigid transform with ``from_frame == Frame.CAMERA`` and
        ``to_frame == Frame.BASE``. Translation is in millimetres.
    rmse_mm :
        Validation RMSE in millimetres (must be >= 0, finite).
    max_error_mm :
        Worst-case residual in millimetres on the validation set.
    num_samples :
        Number of (move, marker) samples accepted by the solver. Must
        be > 0.
    captured_at :
        Timezone-aware UTC timestamp of the solve.
    rig_id :
        Identifier of the camera rig this calibration belongs to.
    quality :
        One of :data:`QUALITY_LABELS`. Use :meth:`from_solver` to
        compute it automatically from ``rmse_mm``.
    schema_version :
        Wire-format version. Pinned to :data:`EXTRINSICS_SCHEMA`.
    """

    transform: Transform
    rmse_mm: float
    max_error_mm: float
    num_samples: int
    captured_at: datetime
    rig_id: str
    quality: QualityLabel = "unknown"
    schema_version: str = EXTRINSICS_SCHEMA

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if not isinstance(self.transform, Transform):
            raise ExtrinsicsError(
                f"transform must be a Transform, got {type(self.transform).__name__}"
            )
        if self.transform.from_frame is not Frame.CAMERA or self.transform.to_frame is not Frame.BASE:
            raise ExtrinsicsError(
                "Extrinsics requires Transform(from_frame=CAMERA, to_frame=BASE); "
                f"got {self.transform.from_frame.value} -> {self.transform.to_frame.value}"
            )

        rmse = float(self.rmse_mm)
        if not math.isfinite(rmse) or rmse < 0:
            raise ExtrinsicsError(f"rmse_mm must be finite and >= 0, got {self.rmse_mm!r}")
        max_err = float(self.max_error_mm)
        if not math.isfinite(max_err) or max_err < 0:
            raise ExtrinsicsError(
                f"max_error_mm must be finite and >= 0, got {self.max_error_mm!r}"
            )
        if max_err + 1e-9 < rmse:
            raise ExtrinsicsError(
                f"max_error_mm ({max_err}) must be >= rmse_mm ({rmse})"
            )

        if not isinstance(self.num_samples, int) or isinstance(self.num_samples, bool):
            raise ExtrinsicsError(
                f"num_samples must be int, got {type(self.num_samples).__name__}"
            )
        if self.num_samples <= 0:
            raise ExtrinsicsError(f"num_samples must be > 0, got {self.num_samples}")

        if not isinstance(self.captured_at, datetime):
            raise ExtrinsicsError(
                f"captured_at must be a datetime, got {type(self.captured_at).__name__}"
            )
        if self.captured_at.tzinfo is None:
            raise ExtrinsicsError("captured_at must be timezone-aware (UTC preferred)")

        if not isinstance(self.rig_id, str) or not self.rig_id.strip():
            raise ExtrinsicsError("rig_id must be a non-empty string")

        if self.quality not in QUALITY_LABELS:
            raise ExtrinsicsError(
                f"quality must be one of {QUALITY_LABELS}, got {self.quality!r}"
            )

        if self.schema_version != EXTRINSICS_SCHEMA:
            raise ExtrinsicsError(
                f"schema_version must be {EXTRINSICS_SCHEMA!r}, got {self.schema_version!r}"
            )

        # Normalise floats so round-tripping through JSON stays bit-stable.
        object.__setattr__(self, "rmse_mm", rmse)
        object.__setattr__(self, "max_error_mm", max_err)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_solver(
        cls,
        *,
        transform: Transform,
        rmse_mm: float,
        max_error_mm: float,
        num_samples: int,
        rig_id: str,
        captured_at: datetime | None = None,
        bands: QualityBandsMm = DEFAULT_BANDS_MM,
    ) -> Extrinsics:
        """Build an :class:`Extrinsics` from raw solver outputs.

        ``quality`` is derived from ``rmse_mm`` via the supplied
        :class:`QualityBandsMm`. ``captured_at`` defaults to ``now(UTC)``.
        """
        ts = captured_at if captured_at is not None else datetime.now(UTC)
        quality = classify_rmse(rmse_mm, bands)
        return cls(
            transform=transform,
            rmse_mm=float(rmse_mm),
            max_error_mm=float(max_error_mm),
            num_samples=int(num_samples),
            captured_at=ts,
            rig_id=rig_id,
            quality=quality,
        )

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def apply_pose(self, pose: Pose) -> Pose:
        """Map a camera-frame :class:`Pose` into the base frame.

        Raises :class:`FrameMismatchError` (via :meth:`Transform.apply_pose`)
        if ``pose.frame`` is not ``Frame.CAMERA``.
        """
        return self.transform.apply_pose(pose)

    def apply_point_mm(self, point_camera_mm) -> np.ndarray:
        """Map a 3-D camera-frame point (mm) into the base frame.

        Returns a fresh ``(3,) float64`` array.
        """
        return self.transform.apply_point(np.asarray(point_camera_mm, dtype=np.float64))
