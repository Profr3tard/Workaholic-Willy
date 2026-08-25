"""Operator-safe mode preset overlays.

The presets are static YAML files under
``config/data/grasping_presets/``. Each overlay is a partial
``robot:`` block that can be deep-merged onto an existing
:class:`RobotConfig` payload via :func:`apply_preset`.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml


_PRESETS_DIR: Path = (
    Path(__file__).resolve().parents[4]
    / "config"
    / "data"
    / "grasping_presets"
)


def _presets_dir() -> Path:
    if not _PRESETS_DIR.is_dir():  # pragma: no cover - defensive
        raise FileNotFoundError(
            f"grasping_presets directory not found at {_PRESETS_DIR}"
        )
    return _PRESETS_DIR


def list_presets() -> list[str]:
    """Return the available preset names, sorted."""

    return sorted(
        p.stem for p in _presets_dir().glob("*.yaml") if p.is_file()
    )


def load_preset(name: str) -> dict[str, Any]:
    """Return the raw overlay ``dict`` for ``name``."""

    path = _presets_dir() / f"{name}.yaml"
    if not path.is_file():
        raise KeyError(
            f"unknown preset {name!r}; available: {list_presets()!r}"
        )
    with path.open() as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, Mapping):
        raise ValueError(
            f"preset {name!r} must be a YAML mapping; got {type(data).__name__}"
        )
    return dict(data)


def _deep_merge(base: Any, overlay: Any) -> Any:
    if isinstance(base, Mapping) and isinstance(overlay, Mapping):
        result: dict[str, Any] = dict(base)
        for key, value in overlay.items():
            if key in result:
                result[key] = _deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result
    # Overlay values replace base values for non-mapping types
    # (lists, scalars). Mirrors :func:`config.loader._deep_merge`.
    return copy.deepcopy(overlay)


def apply_preset(base: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Return a new dict with the preset deep-merged onto ``base``."""

    overlay = load_preset(name)
    return _deep_merge(copy.deepcopy(dict(base)), overlay)


def validate_preset(name: str, *, base: Mapping[str, Any] | None = None) -> None:
    """
    Validate that preset ``name`` deep-merges onto ``base``
    (default: ``load_config().robot``) into a schema-valid :class:`RobotConfig`;
    raises :class:`KeyError` for an unknown preset or
    ``pydantic.ValidationError`` for a bad key.
    """

    from config.loader import load_config
    from config.schema.robot import RobotConfig

    if base is not None:
        base_dict = dict(base)
    else:
        robot = load_config().robot
        if robot is None:
            raise ValueError(
                "validate_preset: the loaded config has no `robot` block to "
                "validate the preset against."
            )
        base_dict = robot.model_dump(mode="python")
    # The deep-merge runs outside Pydantic, so a typo'd key (e.g.
    # ``defualt_mode``) would slip through silently.
    RobotConfig.model_validate(apply_preset(base_dict, name))


def validate_all_presets(*, base: Mapping[str, Any] | None = None) -> list[str]:
    """Validate every shipped preset against the schema and return the validated names."""

    names = list_presets()
    for name in names:
        validate_preset(name, base=base)
    return names
