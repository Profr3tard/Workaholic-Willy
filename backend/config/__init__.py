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

_LOADER_EXPORTS = {"ConfigError", "load_config", "reload_config"}


def __getattr__(name: str):
    if name in _LOADER_EXPORTS:
        from . import loader

        value = getattr(loader, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AppConfig",
    "CameraConfig",
    "ConfigError",
    "ModelsConfig",
    "RobotConfig",
    "RuntimeConfig",
    "load_config",
    "reload_config",
]