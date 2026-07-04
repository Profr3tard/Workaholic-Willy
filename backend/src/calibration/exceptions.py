"""Exception hierarchy for the Willy calibration subsystem."""

from __future__ import annotations


class CalibrationError(Exception):
    """Base class for all calibration-layer errors."""


class CalibrationDataError(CalibrationError):
    """Raised when calibration inputs are missing, malformed, or ambiguous."""


class CalibrationSolveError(CalibrationError):
    """Raised when a calibration solve cannot produce a valid result."""


class ExtrinsicsError(CalibrationError):
    """Raised when an :class:`Extrinsics` value is invalid or used incorrectly."""


class StereoCalibrationError(CalibrationError):
    """Raised by stereo calibration, persistence, or reconstruction boundaries."""


__all__ = [
    "CalibrationDataError",
    "CalibrationError",
    "CalibrationSolveError",
    "ExtrinsicsError",
    "StereoCalibrationError",
]