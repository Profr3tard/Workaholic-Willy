"""Workaholic-Willy configuration package.

Schema classes are imported eagerly. YAML-backed loader helpers are loaded
lazily so schema-only imports do not require the optional YAML dependency.
"""

from __future__ import annotations

from .schema import (
    AppConfig,
    CameraConfig,
    ModelsConfig,
    RobotConfig,
    RuntimeConfig,
)

_LOADER_EXPORTS = {"ConfigError", "load_config", "load_robot_config", "reload_config"}

#: The noun. Lazy for the same reason the loader helpers are: it reaches YAML on first use, and a
#: schema-only import must not require the optional dependency.
_TREE_EXPORTS = {"ConfigTree", "LoadedTree", "default_data_dir"}


def __getattr__(name: str):
    if name in _LOADER_EXPORTS:
        from . import loader

        value = getattr(loader, name)
        globals()[name] = value
        return value
    if name in _TREE_EXPORTS:
        from . import tree

        value = getattr(tree, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AppConfig",
    "CameraConfig",
    "ConfigError",
    "ConfigTree",
    "LoadedTree",
    "ModelsConfig",
    "RobotConfig",
    "RuntimeConfig",
    "default_data_dir",
    "load_config",
    "load_robot_config",
    "reload_config",
]
