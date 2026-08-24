"""Typed operator-facing contract for grasp sampling modes.

Maps the public ``GraspSamplingMode`` enum to the calculator's internal
tri-state ``dense_sampling`` setting while preserving legacy boolean
callers. Supports a closed set of YAML/CLI string values and rejects
unknown values with ``ValueError``.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "GraspSamplingMode",
    "GraspSamplingModeInput",
    "mode_to_dense_sampling",
    "resolve_grasp_sampling_mode",
]


class GraspSamplingMode(StrEnum):
    """Operator-facing grasp sampling mode."""

    AUTO = "auto"
    SINGLE_OBJECT = "single_object"
    DENSE_CLUTTER = "dense_clutter"


# Public input alias used in signatures that accept the typed enum,
# the legacy bool, a stable string alias, or ``None`` (== ``AUTO``).
GraspSamplingModeInput = "GraspSamplingMode | bool | str | None"


_STR_ALIASES: dict[str, GraspSamplingMode] = {
    "auto": GraspSamplingMode.AUTO,
    "single": GraspSamplingMode.SINGLE_OBJECT,
    "single_object": GraspSamplingMode.SINGLE_OBJECT,
    "dense": GraspSamplingMode.DENSE_CLUTTER,
    "dense_clutter": GraspSamplingMode.DENSE_CLUTTER,
}


def resolve_grasp_sampling_mode(
    value: GraspSamplingMode | bool | str | None,
) -> GraspSamplingMode:
    """Coerce any accepted input form into a :class:`GraspSamplingMode`.

    Accepted inputs:

    * an existing :class:`GraspSamplingMode` (returned as-is),
    * ``None`` -> :attr:`GraspSamplingMode.AUTO`,
    * ``True`` -> :attr:`GraspSamplingMode.DENSE_CLUTTER` (legacy bool),
    * ``False`` -> :attr:`GraspSamplingMode.SINGLE_OBJECT` (legacy bool),
    * a string from ``{"auto", "single", "single_object", "dense",
      "dense_clutter"}`` (case-insensitive).

    Any other value raises :class:`ValueError` listing the allowed forms
    so callers see exactly what they passed and what is accepted.
    """
    if isinstance(value, GraspSamplingMode):
        return value
    if value is None:
        return GraspSamplingMode.AUTO
    # ``bool`` is a subclass of ``int`` so it must be tested before any
    # generic int handling; we deliberately do not accept arbitrary
    # ints so a stray ``0`` / ``1`` cannot silently mean a mode.
    if isinstance(value, bool):
        return (
            GraspSamplingMode.DENSE_CLUTTER if value else GraspSamplingMode.SINGLE_OBJECT
        )
    if isinstance(value, str):
        key = value.strip().lower()
        mapped = _STR_ALIASES.get(key)
        if mapped is not None:
            return mapped
    allowed = (
        "GraspSamplingMode enum, None, True/False, or one of "
        + ", ".join(sorted(_STR_ALIASES))
    )
    raise ValueError(
        f"Invalid grasp_sampling_mode value {value!r}; expected {allowed}."
    )


def mode_to_dense_sampling(mode: GraspSamplingMode) -> bool | None:
    """Bridge enum -> calculator's internal ``dense_sampling`` tri-state.

    * ``AUTO`` -> ``None`` (let the calculator's auto heuristic decide),
    * ``SINGLE_OBJECT`` -> ``False`` (force silhouette path),
    * ``DENSE_CLUTTER`` -> ``True`` (force dense surface sampler).
    """
    if mode is GraspSamplingMode.AUTO:
        return None
    if mode is GraspSamplingMode.SINGLE_OBJECT:
        return False
    if mode is GraspSamplingMode.DENSE_CLUTTER:
        return True
    raise ValueError(f"Unhandled GraspSamplingMode: {mode!r}")
