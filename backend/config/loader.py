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
    camera/hand_eye.yaml       (optional — schema defaults are used otherwise)
    models/*.yaml             (auto-discovered; every top-level key is merged)
    robot/robot.yaml          (optional)

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
Every load-time failure is raised as :class:`ConfigError`. A schema failure names, per offending key,
the FILE and LINE it was written on and which profile layer that file belongs to, plus a did-you-mean
for a near-miss key name. That matters because a profile chain merges several files into one section,
and the message used to name only the data DIRECTORY -- leaving the reader to guess which of them
carried the typo. (This paragraph previously claimed the file was named; it was not. It is now.)
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

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class ConfigError(RuntimeError):
    """Raised on any configuration load / validation failure."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
_PROFILE_ENV_VAR = "WILLY_PROFILE"

#: ``WILLY_PROFILE`` accepts a COMMA-SEPARATED CHAIN of profile layers, applied left-to-right
#: (``"sim,ur3e"`` merges every ``*.sim.yaml`` first, then every ``*.ur3e.yaml`` on top). A single
#: name is just a one-element chain, so the historical behaviour is unchanged.
#:
#: Composition exists because the layers are INDEPENDENT dimensions. The sim cell's overlays span
#: four files (``robot``, ``camera/hand_eye``, ``models/object``, ``models/segmenting``) and carry
#: measured values -- e.g. the detector ``torch_dtype`` reset that small-object recall depends on.
#: Expressing "the sim cell, but driving a UR3e" as a *separate* profile would have to FORK all four,
#: duplicating those measured values per robot; they would then drift. Chaining keeps one ``sim``
#: layer and adds a small per-robot layer that only states what the robot actually changes.
_PROFILE_SEPARATOR = ","
#: Phase U11 — opt-in environment variable. When set to a readable
#: YAML path, the loader deep-merges that overlay into the ``robot``
#: section of the raw config **before** Pydantic validation. The
#: overlay is rejected unless every leaf key path is in the
#: ``runtime_mutable=True`` allow-list discovered from ``RobotConfig``.
#: Default-off (env unset) ⇒ byte-identical legacy path.
_ADAPTATION_OVERLAY_ENV_VAR = "WILLY_ADAPTATION_OVERLAY"

#: Reset sentinel for PROFILE overlays. A ``None`` overlay leaf keeps the base value (so a partial
#: overlay never accidentally wipes a field). When a profile *does* need to unset a base value, it
#: writes this string; the loader resolves it to ``None`` after the merge. This is the only way an
#: overlay can force a field back to ``None`` (e.g. ``models/object.sim.yaml`` drops the production
#: ``optim.torch_dtype: auto`` so the sim's detector runs its validated unset/fp32-weights recall path).
_OVERLAY_RESET = "__null__"

# Required camera files. These are not auto-discovered because the
# loader maps each one to a specific section in the AppConfig tree.
_CAMERA_FILES = ("cam.yaml", "stereomatcher.yaml")
_CAMERA_KEYS = ("cameras", "stereomatcher")

# ${VAR} or ${VAR:-default}. Single-line, no nesting.
_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?:\:-([^}]*))?\}")


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
    """Return the currently selected config profile (the raw chain), or ``None`` if unset."""
    raw = os.environ.get(_PROFILE_ENV_VAR, "").strip()
    return raw or None


def profile_layers(profile: str | None) -> tuple[str, ...]:
    """Split a profile value into its ordered overlay layers (``"sim,ur3e"`` -> ``("sim", "ur3e")``).

    Later layers win: each layer's ``*.<layer>.yaml`` overlay is deep-merged on top of the previous
    one. Blank segments are dropped, so ``"sim,"`` / ``" sim , ur3e "`` are tolerated.
    """
    if not profile:
        return ()
    return tuple(part for part in (seg.strip() for seg in profile.split(_PROFILE_SEPARATOR)) if part)


def join_profiles(*layers: str) -> str:
    """Build a chain value for :func:`set_active_profile` (``join_profiles("sim", "ur3e")``)."""
    return _PROFILE_SEPARATOR.join(layer for layer in layers if layer)


def set_active_profile(profile: str | None) -> None:
    """Set/clear :data:`WILLY_PROFILE` for subsequent config loads."""
    val = (profile or "").strip()
    if val:
        os.environ[_PROFILE_ENV_VAR] = val
    else:
        os.environ.pop(_PROFILE_ENV_VAR, None)


def available_profiles(data_dir: str | Path | None = None) -> list[str]:
    """Return sorted profile LAYER names discovered from ``*.{layer}.yaml`` files.

    Layers are composable: any subset can be combined into a chain (``"sim,ur3e"``), so this is the
    menu of building blocks, not the list of valid whole configurations.
    """
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

    # Phase U11 — transparent guarded-adaptation overlay merge.
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
        raise ConfigError(_describe_validation_error(exc, root, profile)) from exc


def _describe_validation_error(exc: ValidationError, root: Path, profile: str) -> str:
    """Render a validation failure so the reader can go straight to the line that caused it.

    The old message named the data DIRECTORY — which, with a profile chain merging four files into one
    section, left the reader guessing which of them carried the typo. (The module docstring above
    promised it named the file; it did not.) The fix is not to relax anything: same exception, same
    fail-closed behaviour, strictly more information.

    Explaining is best-effort and must never become a second failure: if the side-car index cannot be
    built, the original pydantic report is still printed in full.
    """
    layers = profile_layers(profile)
    header = f"configuration failed schema validation under {root}"
    if layers:
        header += f" (profile chain: {' -> '.join(layers)})"
    lines = [header + ":"]
    try:
        from ._provenance import index_origins, nearest_keys

        origins = index_origins(root, layers)
    except Exception:  # noqa: BLE001 - never let the explainer mask the real error
        return f"{header}:\n{exc}"

    for err in exc.errors():
        dotted = ".".join(str(part) for part in err["loc"])
        lines.append(f"\n  {dotted}")
        origin = origins.get(dotted)
        if origin is not None:
            lines.append(f"      at {origin.location(root)}")
        else:
            # No origin means the key is NOT in any YAML: it is a schema default that failed a
            # cross-field rule, or a required field nobody wrote. Saying so is the useful half.
            lines.append("      (not written in any YAML — a default or a cross-field rule)")
        lines.append(f"      {err['msg']}")
        if err["type"] == "extra_forbidden" and err["loc"]:
            siblings = [
                key.rsplit(".", 1)[-1]
                for key in origins
                if key.rsplit(".", 1)[0] == dotted.rsplit(".", 1)[0] and key != dotted
            ]
            for suggestion in nearest_keys(str(err["loc"][-1]), siblings):
                lines.append(f"      did you mean: {suggestion}?")
    lines.append(
        "\nEvery key above is rejected on purpose (unknown keys are never ignored). "
        "`python -m backend.config` re-runs this check without booting anything."
    )
    return "\n".join(lines)


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

    # Allow profile-only model files (e.g. models/edge.prod.yaml) that have no non-profile base
    # counterpart. Gathered per profile LAYER in chain order and merged BY STEM first, so a later
    # layer refines the same profile-only file (models/foo.sim.yaml -> models/foo.ur3e.yaml) instead
    # of colliding with it; a genuine duplicate model key across DIFFERENT files still raises.
    base_stems = {p.stem for p in base_paths}
    extra: dict[str, tuple[Path, dict[str, Any]]] = {}
    for layer in profile_layers(profile):
        for overlay_path in sorted(models_dir.glob(f"*.{layer}.yaml")):
            stem = _base_stem_for_overlay(overlay_path, layer)
            if stem in base_stems:
                continue
            data = _load_yaml(overlay_path)
            if not isinstance(data, dict):
                raise ConfigError(f"{overlay_path}: expected mapping at top level")
            previous = extra.get(stem)
            extra[stem] = (
                overlay_path, data if previous is None else _deep_merge(previous[1], data),
            )
    for overlay_path, data in extra.values():
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
    """Load base YAML and the optional ``.<layer>.yaml`` overlay of every profile layer, deep-merged.

    Overlay semantics:
    - a multi-layer profile (``"sim,ur3e"``) merges its overlays left-to-right, then onto the base
    - dict values merge recursively
    - scalars/lists replace base values
    - a ``None`` (``null``) overlay leaf KEEPS the base value (a partial overlay can omit-by-null)
    - the string :data:`_OVERLAY_RESET` (``"__null__"``) RESETS a field to ``None`` — the one way a
      profile overlay can *unset* a base value (e.g. ``models/object.sim.yaml`` drops the production
      ``optim.torch_dtype: auto`` back to the unset/HF default that the sim's validated recall path needs).
    """
    if required:
        base = _load_yaml(path)
    else:
        base = _load_yaml_optional(path)
    overlay = _load_profile_overlay(path, profile)
    if base is None and overlay is None:
        return None
    if base is None:
        return _resolve_overlay_resets(overlay)
    if overlay is None:
        return base
    return _resolve_overlay_resets(_deep_merge(base, overlay))


def _load_profile_overlay(path: Path, profile: str) -> Any | None:
    """The merged overlay for ``path`` across every layer of ``profile`` (left-to-right, later wins).

    Returns ``None`` when no layer contributes an overlay file for ``path`` (so the caller keeps the
    base verbatim). A single-layer profile reduces to the historical "load one ``*.<profile>.yaml``".
    """
    merged: Any | None = None
    for layer in profile_layers(profile):
        overlay_path = path.with_name(f"{path.stem}.{layer}{path.suffix}")
        if not overlay_path.exists():
            continue
        data = _load_yaml(overlay_path)
        merged = data if merged is None else _deep_merge(merged, data)
    return merged


def _resolve_overlay_resets(value: Any) -> Any:
    """Replace every :data:`_OVERLAY_RESET` sentinel with ``None`` (post-merge, profile overlays only).

    Runs on the MERGED result so a sentinel in the overlay — which merges in as a normal scalar-replace —
    becomes ``None`` in the final config. Production (no profile) never carries the sentinel, so this is a
    no-op there.
    """
    if isinstance(value, dict):
        return {key: _resolve_overlay_resets(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_overlay_resets(item) for item in value]
    return None if value == _OVERLAY_RESET else value


def _active_profile(root: Path) -> str:
    """Validate + normalise ``WILLY_PROFILE``. EVERY layer of the chain must contribute a file.

    Fail-closed per layer: a typo'd layer (``sim,ur3``) would otherwise merge silently as a no-op and
    the cell would come up with the OTHER robot's geometry, which is the failure this whole check
    exists to prevent.
    """
    profile = os.environ.get(_PROFILE_ENV_VAR, "").strip()
    layers = profile_layers(profile)
    if not layers:
        return ""
    for layer in layers:
        if not any(root.rglob(f"*.{layer}.yaml")):
            raise ConfigError(
                f"{_PROFILE_ENV_VAR}={profile!r} selects profile layer {layer!r}, but no "
                f"'*.{layer}.yaml' overlay file exists under {root}."
            )
    return _PROFILE_SEPARATOR.join(layers)


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
