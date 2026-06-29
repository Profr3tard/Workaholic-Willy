from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Reusable validators
# ---------------------------------------------------------------------------

# Known OpenCV ArUco dictionary names. We hard-code the list rather than
# poking at ``cv2.aruco`` so that schema validation does not require OpenCV
# to be importable during config introspection (e.g. doc generation).
_ARUCO_DICT_NAMES: frozenset[str] = frozenset(
    {
        # Standard
        "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
        "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
        "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
        "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
        # AprilTag-compatible
        "DICT_ARUCO_ORIGINAL",
        "DICT_APRILTAG_16h5", "DICT_APRILTAG_25h9",
        "DICT_APRILTAG_36h10", "DICT_APRILTAG_36h11",
        # MIP
        "DICT_ARUCO_MIP_36h12",
    }
)


def validate_aruco_dict_name(value: Any) -> str:
    """Validate that ``value`` names a known OpenCV ArUco dictionary.

    Used as an ``AfterValidator`` on string fields holding ArUco dict names.
    Accepts only the exact upper-case names recognised by
    ``cv2.aruco.getPredefinedDictionary`` in OpenCV ≥ 4.7.
    """
    if not isinstance(value, str):
        raise TypeError(f"aruco_dict_name must be a string, got {type(value).__name__}")
    if value not in _ARUCO_DICT_NAMES:
        raise ValueError(
            f"unknown ArUco dictionary {value!r}; "
            f"must be one of: {sorted(_ARUCO_DICT_NAMES)}"
        )
    return value
