"""Configuration loader: reads YAML files, merges them, validates the result.

Public API
----------
:func:`load_config`
    Read the configured ``data/`` tree, merge the YAMLs and return a
    fully-validated :class:`AppConfig`. The result is cached by
    ``data_dir`` and active profile (use :func:`reload_config` to
    invalidate after file edits).

:func:`reload_config`
    Drop the cache. Useful for tests and for hot-reloading scenarios.

Layout
------
The loader expects (paths are relative to ``data_dir``)::

    app/runtime.yaml          (optional — schema defaults are used otherwise)
    camera/cam.yaml
    camera/stereomatcher.yaml
    camera/eye_to_hand.yaml
    camera/hand_eye.yaml       (optional — schema defaults are used otherwise)
    models/*.yaml             (auto-discovered; every top-level key is merged)
    robot/robot.yaml          (optional — For Real Usage, please adapt that to you robot)

Environment-variable substitution
---------------------------------
Strings in any YAML may reference environment variables using the
``${VAR}`` or ``${VAR:-default}`` syntax. Substitution happens *before*
parsing, so the result must be valid YAML.

Examples::

    ip: ${ROBOT_IP:-192.168.1.100}
    model_path: ${MODEL_DIR}/whisper

If a referenced variable is unset and no default is supplied, the loader
raises :class:`ConfigError`.

Errors
------
Every load-time failure is raised as :class:`ConfigError`. The message
always names the offending file and, where possible, the field path.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from ._merge import _deep_merge
from .schema.app import AppConfig

# Constant imports for readability
from ._utils.constants import (
    ConfigError,
    _DEFAULT_DATA_DIR,
    _PROFILE_ENV_VAR,
    _ADAPTATION_OVERLAY_ENV_VAR,
    _CAMERA_FILES,
    _CAMERA_KEYS,
    _ENV_VAR_RE,
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(data_dir: str | Path | None = None) -> AppConfig:
    """Load + validate the YAML config tree.

    Args:
        data_dir: Override the default ``backend/config/data`` location.
            Useful for tests and for shipping per-deployment overlays.

    Returns:
        A fully-validated, immutable :class:`AppConfig` instance. Repeated
        calls with the same ``data_dir`` and active profile return the
        same cached object.

    Raises:
        ConfigError: If a required file is missing, YAML parsing fails,
            an environment variable is unresolved, or schema validation
            rejects the merged result.
    """
    root = Path(data_dir).resolve() if data_dir else _DEFAULT_DATA_DIR
    profile = _active_profile(root)
    return _load_cached(str(root), profile)


def reload_config() -> None:
    """Invalidate the :func:`load_config` cache.

    Call after modifying YAML files at runtime, or between tests that
    install different fixture configs.
    """
    _load_cached.cache_clear()


def active_profile() -> str | None:
    """Return the currently selected config profile, or ``None`` if unset."""
    raw = os.environ.get(_PROFILE_ENV_VAR, "").strip()
    return raw or None


def set_active_profile(profile: str | None) -> None:
    """Set/clear :data:`WORKAHOLIC-WILLY_PROFILE` for subsequent config loads."""
    val = (profile or "").strip()
    if val:
        os.environ[_PROFILE_ENV_VAR] = val
    else:
        os.environ.pop(_PROFILE_ENV_VAR, None)

def available_profiles(data_dir: str | Path | None = None) -> list[str]:
    """Return sorted profile names discovered from ``*.{profile}.yaml`` files."""
    root = Path(data_dir).resolve() if data_dir else _DEFAULT_DATA_DIR
    out: set[str] = set()
    for path in root.rglob("*.yaml"):
        parts = path.stem.split(".")
        if len(parts) < 2:
            continue
        profile = parts[-1].strip()
        if profile:
            out.add(profile)
    return sorted(out)


# ---------------------------------------------------------------------------
# Internal: cached loader
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _load_cached(root_str: str, profile: str) -> AppConfig:
    root = Path(root_str)

    raw: dict[str, Any] = {
        "camera": _load_camera_section(root, profile),
        "models": _load_models_section(root, profile),
    }

    robot = _load_optional_section(root / "robot" / "robot.yaml", "robot", profile)
    if robot is not None:
        raw["robot"] = robot

    # transparent guarded-adaptation overlay merge.
    # Default-off. Env-var-gated. Path allow-list enforced.
    overlay_path_str = os.environ.get(_ADAPTATION_OVERLAY_ENV_VAR, "").strip()
    if overlay_path_str:
        _apply_adaptation_overlay(raw, Path(overlay_path_str))

    runtime = _load_optional_section(root / "app" / "runtime.yaml", "runtime", profile)
    if runtime is not None:
        raw["runtime"] = runtime

    try:
        return AppConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(
            f"configuration failed schema validation under {root}:\n{exc}"
        ) from exc
    

# ---------------------------------------------------------------------------
# Internal: section loaders
# ---------------------------------------------------------------------------

def _load_camera_section(root: Path, profile: str) -> dict[str, Any]:
    section: dict[str, Any] = {}
    for filename, key in zip(_CAMERA_FILES, _CAMERA_KEYS):
        path = root / "camera" / filename
        data = _load_yaml_with_profile(path, profile, required=True)
        section[key] = _require_top_key(data, path, key)
    hand_eye_path = root / "camera" / "hand_eye.yaml"
    hand_eye = _load_optional_section(hand_eye_path, "hand_eye", profile)
    if hand_eye is not None:
        section["hand_eye"] = hand_eye
    return section


def _load_models_section(root: Path, profile: str) -> dict[str, Any]:
    """Auto-discover every ``models/*.yaml`` and merge their top-level keys.

    Each model file is expected to contribute one or more named model
    configs at its top level (e.g. ``handdetect:`` and ``gesturedetect:``
    in ``hand.yaml``). Duplicate keys across files are rejected to keep
    the merge unambiguous.
    """
    models_dir = root / "models"
    if not models_dir.is_dir():
        raise ConfigError(f"models directory not found: {models_dir}")

    merged: dict[str, Any] = {}
    seen_in: dict[str, Path] = {}
    base_paths = sorted(
        p for p in models_dir.glob("*.yaml") if not _is_profile_overlay_file(p)
    )

    for path in base_paths:
        data = _load_yaml_with_profile(path, profile, required=True)
        if not isinstance(data, dict):
            raise ConfigError(f"{path}: expected mapping at top level")
        for key, value in data.items():
            if key in merged:
                raise ConfigError(
                    f"duplicate model key {key!r} found in {path} "
                    f"(already defined in {seen_in[key]})"
                )
            merged[key] = value
            seen_in[key] = path

    # Allow profile-only model files (e.g. models/edge.prod.yaml) that have
    # no non-profile base counterpart.
    if profile:
        base_stems = {p.stem for p in base_paths}
        for overlay_path in sorted(models_dir.glob(f"*.{profile}.yaml")):
            stem = _base_stem_for_overlay(overlay_path, profile)
            if stem in base_stems:
                continue
            data = _load_yaml(overlay_path)
            if not isinstance(data, dict):
                raise ConfigError(f"{overlay_path}: expected mapping at top level")
            for key, value in data.items():
                if key in merged:
                    raise ConfigError(
                        f"duplicate model key {key!r} found in {overlay_path} "
                        f"(already defined in {seen_in[key]})"
                    )
                merged[key] = value
                seen_in[key] = overlay_path

    if not merged:
        raise ConfigError(f"no model configurations found under {models_dir}")
    return merged


def _load_optional_section(path: Path, key: str, profile: str) -> Any | None:
    """Load ``path`` and return ``data[key]``, or None if the file is absent."""
    data = _load_yaml_with_profile(path, profile, required=False)
    if data is None:
        return None
    return _require_top_key(data, path, key)


def _require_top_key(data: Any, path: Path, key: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a YAML mapping with top-level '{key}:'")
    if key not in data:
        raise ConfigError(f"{path} must contain a top-level '{key}:' key")
    return data[key]


def _load_yaml_with_profile(path: Path, profile: str, *, required: bool) -> Any | None:
    """Load base YAML and optional ``.<profile>.yaml`` overlay and deep-merge.

    Overlay semantics:
    - dict values merge recursively
    - scalars/lists replace base values
    - ``None`` in overlay keeps the base value
    """
    if required:
        base = _load_yaml(path)
    else:
        base = _load_yaml_optional(path)
    overlay = _load_profile_overlay(path, profile)
    if base is None and overlay is None:
        return None
    if base is None:
        return overlay
    if overlay is None:
        return base
    return _deep_merge(base, overlay)


def _load_profile_overlay(path: Path, profile: str) -> Any | None:
    if not profile:
        return None
    overlay_path = path.with_name(f"{path.stem}.{profile}{path.suffix}")
    if not overlay_path.exists():
        return None
    return _load_yaml(overlay_path)


def _active_profile(root: Path) -> str:
    profile = os.environ.get(_PROFILE_ENV_VAR, "").strip()
    if not profile:
        return ""
    if not any(root.rglob(f"*.{profile}.yaml")):
        raise ConfigError(
            f"{_PROFILE_ENV_VAR}={profile!r} is set, but no '*.{profile}.yaml' "
            f"overlay file exists under {root}."
        )
    return profile


def _is_profile_overlay_file(path: Path) -> bool:
    return "." in path.stem


def _base_stem_for_overlay(path: Path, profile: str) -> str:
    suffix = f".{profile}"
    stem = path.stem
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


# ---------------------------------------------------------------------------
# Internal: YAML I/O with env-var substitution and file-scoped errors
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> Any:
    if not path.exists():
        raise ConfigError(f"required config file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"could not read {path}: {exc}") from exc

    try:
        text = _substitute_env_vars(text, path)
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(f"env-var substitution failed in {path}: {exc}") from exc

    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML parse error in {path}: {exc}") from exc


def _load_yaml_optional(path: Path) -> Any | None:
    if not path.exists():
        return None
    return _load_yaml(path)


def _substitute_env_vars(text: str, path: Path) -> str:
    """Replace ``${VAR}`` / ``${VAR:-default}`` with the corresponding env values."""

    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2)
        value = os.environ.get(name)
        if value is not None:
            return value
        if default is not None:
            return default
        raise ConfigError(
            f"{path}: environment variable ${{{name}}} is not set and no "
            f"default was provided (use ${{{name}:-fallback}} to supply one)"
        )

    return _ENV_VAR_RE.sub(repl, text)


# ---------------------------------------------------------------------------
# Phase U11 — adaptation overlay merge
# ---------------------------------------------------------------------------

def _apply_adaptation_overlay(raw: dict[str, Any], overlay_path: Path) -> None:
    """Merge a U11 adaptation overlay YAML into ``raw`` in-place.

    The overlay is required to be a mapping rooted at ``robot:``; every
    leaf key path must be in the ``runtime_mutable`` allow-list
    discovered from the schema (see ``adaptation.discover_runtime_mutable_fields``).
    Any unknown / forbidden key raises :class:`ConfigError`.
    """

    if not overlay_path.exists():
        raise ConfigError(
            f"adaptation overlay not found: {overlay_path} "
            f"(env {_ADAPTATION_OVERLAY_ENV_VAR})"
        )

    # Local imports to avoid a top-level dependency on the grasping
    # replay package when the env var is unset.
    from backend.config.schema.robot.robot_schema import RobotConfig
    from backend.src.robot.grasping.replay.adaptation import (
        OverlayPathError,
        discover_runtime_mutable_fields,
        validate_overlay_against_allowlist,
    )

    data = _load_yaml(overlay_path)
    if not isinstance(data, dict):
        raise ConfigError(
            f"{overlay_path}: adaptation overlay must be a YAML mapping"
        )
    if set(data.keys()) - {"robot"}:
        raise ConfigError(
            f"{overlay_path}: adaptation overlay may only contain a "
            f"top-level 'robot:' section (saw {sorted(data.keys())!r})"
        )
    if "robot" not in data:
        return  # empty / no-op overlay

    overlay_robot = data["robot"]
    if not isinstance(overlay_robot, dict):
        raise ConfigError(
            f"{overlay_path}: 'robot' section must be a mapping"
        )

    allowed = tuple(
        # Allow-list paths are stored with a ``robot.`` prefix; the
        # overlay's ``robot:`` key is implicit, so strip it for matching.
        spec.dotted_key.removeprefix("robot.")
        for spec in discover_runtime_mutable_fields(RobotConfig())
    )
    forbidden = validate_overlay_against_allowlist(overlay_robot, allowed)
    if forbidden:
        raise ConfigError(
            f"{overlay_path}: adaptation overlay touches forbidden key(s): "
            f"{', '.join(sorted(forbidden))}"
        )

    base_robot = raw.get("robot", {})
    if not isinstance(base_robot, dict):
        raise ConfigError(
            f"{overlay_path}: cannot merge overlay — base 'robot' section "
            f"is not a mapping"
        )
    raw["robot"] = _deep_merge(base_robot, overlay_robot)
    # Suppress unused-symbol warning when the import is otherwise lint-only.
    _ = OverlayPathError