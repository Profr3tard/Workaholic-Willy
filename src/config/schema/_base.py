"""Shared base classes and validators for every config schema.

Every configuration model inherits :class:`StrictModel`, which carries two
invariants across the whole tree:

* ``extra="forbid"``: a typo in a YAML file is refused at load time rather
  than ignored without comment.
* ``frozen=True``: a config object is immutable once constructed, so no
  component can mutate what another one reads. Configs are built once at
  startup and treated as values, not as state.

Use :class:`StrictModel` for every new config class.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base class for every configuration schema in this package.

    Carries the ``extra="forbid"`` and ``frozen=True`` invariants the module
    docstring describes. A subclass may opt out of either by overriding
    ``model_config``, and must state in a comment why.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )


# ---------------------------------------------------------------------------
# Reusable validators
# ---------------------------------------------------------------------------

# Known OpenCV ArUco dictionary names, held here rather than read from
# ``cv2.aruco``, so that validating a schema does not need OpenCV to be
# importable. Config introspection such as doc generation runs without it.
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

    Used as an ``AfterValidator`` on the string fields that hold a dictionary
    name. Only the exact upper-case names recognised by
    ``cv2.aruco.getPredefinedDictionary`` in OpenCV >= 4.7 are accepted.
    """
    if not isinstance(value, str):
        raise TypeError(f"aruco_dict_name must be a string, got {type(value).__name__}")
    if value not in _ARUCO_DICT_NAMES:
        raise ValueError(
            f"unknown ArUco dictionary {value!r}; "
            f"must be one of: {sorted(_ARUCO_DICT_NAMES)}"
        )
    return value
