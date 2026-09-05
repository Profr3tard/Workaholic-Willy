"""Pure helpers shared by calibration modules.

Unit scaling, intrinsics extraction and stereo image-shape checks. Nothing
here touches a device or the filesystem.
"""

from __future__ import annotations

import numpy as np

from src.utility.unit_scaling import unit_scaling as _shared_unit_scaling

from .exceptions import CalibrationDataError

__all__ = ["proj_to_K", "unit_scaling", "validate_image_pair_shapes"]


def unit_scaling(unit: str) -> float:
    """Return the multiplier that converts millimetres into ``unit``.

    Wraps :mod:`src.utility.unit_scaling` and re-raises an unknown unit as
    :class:`CalibrationDataError`, so callers only catch calibration errors.
    """
    try:
        return _shared_unit_scaling(unit)
    except ValueError as exc:
        raise CalibrationDataError(str(exc)) from exc


def proj_to_K(projection_matrix: np.ndarray) -> np.ndarray:
    """Extract rectified camera intrinsics from a 3x4 projection matrix.

    Only fx, fy, cx and cy reach the returned 3x3 K; the fourth column, which
    holds the translation term, is dropped.
    """
    projection = np.asarray(projection_matrix, dtype=np.float64)
    if projection.shape != (3, 4):
        raise CalibrationDataError(
            f"projection_matrix must have shape (3, 4), got {projection.shape}"
        )
    if not np.all(np.isfinite(projection)):
        raise CalibrationDataError("projection_matrix must contain only finite values")

    intrinsics = np.eye(3, dtype=np.float64)
    intrinsics[0, 0] = projection[0, 0]
    intrinsics[1, 1] = projection[1, 1]
    intrinsics[0, 2] = projection[0, 2]
    intrinsics[1, 2] = projection[1, 2]
    return intrinsics


def validate_image_pair_shapes(
    left_image: np.ndarray,
    right_image: np.ndarray,
    *,
    expected_size: tuple[int, int] | None = None,
) -> None:
    """Check that a stereo image pair shares one height and width.

    Each image must be grayscale or colour and the two must agree on HxW; with
    ``expected_size`` given as (width, height) they must also match the frame
    size the calibration was solved at. Any mismatch raises
    :class:`CalibrationDataError`.
    """
    if left_image is None or right_image is None:
        raise CalibrationDataError("left_image and right_image are required")
    if left_image.ndim not in (2, 3):
        raise CalibrationDataError(
            f"left_image must be grayscale or color, got ndim={left_image.ndim}"
        )
    if right_image.ndim not in (2, 3):
        raise CalibrationDataError(
            f"right_image must be grayscale or color, got ndim={right_image.ndim}"
        )
    if left_image.shape[:2] != right_image.shape[:2]:
        raise CalibrationDataError(
            "left_image and right_image must have the same height/width; "
            f"got {left_image.shape[:2]} vs {right_image.shape[:2]}"
        )
    if expected_size is not None:
        width, height = int(expected_size[0]), int(expected_size[1])
        if left_image.shape[:2] != (height, width):
            raise CalibrationDataError(
                f"image pair shape must match calibration frame size {width}x{height}; "
                f"got {left_image.shape[1]}x{left_image.shape[0]}"
            )
