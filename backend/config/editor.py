"""Helpers for reading, validating, and writing config profile overlays.

The loader is authoritative for parsing + validation. This module adds a
thin editing layer intended for a future/hypothetical settings UI (D8: there is
NO frontend or web server in this repo today — these helpers are a pure-Python
config-editing API, used by the test suite + callable from a CLI):
    - Later this editor should be able to get all the profiles, from the GUI,
    either from Webservice (Education) or dedicated App (Comercial)

* enumerate editable base YAML files
* read a profile's overlay bundle as text
* validate/save a bundle against the real AppConfig schema
* build/update a small "basic settings" overlay patch
"""

from __future__ import annotations

import re
import shutil
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

from ._merge import _deep_merge
from .loader import (
    _DEFAULT_DATA_DIR,
    ConfigError,
    active_profile,
    available_profiles,
    load_config,
    reload_config,
    set_active_profile,
)

_PROFILE_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def data_root(data_dir: str | Path | None = None) -> Path:
    return Path(data_dir).resolve() if data_dir else _DEFAULT_DATA_DIR


def editable_base_files(data_dir: str | Path | None = None) -> list[Path]:
    """Return all base YAML files that can be overlaid by a profile."""
    root = data_root(data_dir)
    files = [
        p for p in root.rglob("*.yaml")
        if "." not in p.stem
    ]
    return sorted(files, key=lambda p: relpath(p, root))


def relpath(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def overlay_path(base_file: Path, profile: str) -> Path:
    return base_file.with_name(f"{base_file.stem}.{profile}{base_file.suffix}")


def read_overlay_bundle(profile: str, data_dir: str | Path | None = None) -> list[dict[str, Any]]:
    validate_profile_name(profile)
    root = data_root(data_dir)
    docs: list[dict[str, Any]] = []
    for base in editable_base_files(root):
        target = overlay_path(base, profile)
        exists = target.exists()
        docs.append({
            "path": relpath(base, root),
            "exists": exists,
            "content": target.read_text(encoding="utf-8") if exists else "",
        })
    return docs


def load_effective_config_dict(
    *,
    profile: str | None = None,
    data_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = data_root(data_dir)
    with _profile_context(profile):
        cfg = load_config(root)
        return cfg.model_dump(mode="python")


def profile_status(
    *,
    pipeline_running: bool = False,
    data_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Return active/available profile metadata for settings endpoints."""
    return {
        "active_profile": active_profile(),
        "available_profiles": available_profiles(data_dir),
        "pipeline_running": bool(pipeline_running),
    }


def switch_profile(
    profile: str | None,
    *,
    pipeline_running: bool = False,
    data_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Activate a profile and return its effective config.

    Passing ``None`` or an empty string clears the active profile. The
    previous profile is restored if validation fails.
    """
    _reject_if_pipeline_running(pipeline_running)
    next_profile = (profile or "").strip() or None
    if next_profile is not None:
        validate_profile_name(next_profile)
    root = data_root(data_dir)
    previous = active_profile()
    try:
        set_active_profile(next_profile)
        reload_config()
        return load_config(root).model_dump(mode="python")
    except Exception as original:
        # Roll back to the previous profile, but never let a rollback failure
        # mask the original error: chain it so operators see both and the
        # original ConfigError is not silently replaced.
        try:
            set_active_profile(previous)
            reload_config()
        except Exception as rollback_err:
            raise rollback_err from original
        raise


def validate_overlay_bundle(
    profile: str,
    documents: Iterable[dict[str, Any]],
    *,
    data_dir: str | Path | None = None,
) -> dict[str, Any]:
    validate_profile_name(profile)
    root = data_root(data_dir)
    with tempfile.TemporaryDirectory(prefix="aurora-config-") as tmp:
        tmp_root = Path(tmp) / "data"
        shutil.copytree(root, tmp_root)
        _write_bundle_to_tree(tmp_root, profile, documents)
        return load_effective_config_dict(profile=profile, data_dir=tmp_root)


def save_overlay_bundle(
    profile: str,
    documents: Iterable[dict[str, Any]],
    *,
    data_dir: str | Path | None = None,
    pipeline_running: bool = False,
) -> dict[str, Any]:
    _reject_if_pipeline_running(pipeline_running)
    validate_profile_name(profile)
    root = data_root(data_dir)
    effective = validate_overlay_bundle(profile, documents, data_dir=root)
    _write_bundle_to_tree(root, profile, documents)
    reload_config()
    return effective


def apply_basic_settings(
    profile: str,
    basic: dict[str, Any],
    *,
    data_dir: str | Path | None = None,
    pipeline_running: bool = False,
) -> dict[str, Any]:
    """Merge a small operator-safe settings patch into the profile overlay."""
    _reject_if_pipeline_running(pipeline_running)
    validate_profile_name(profile)
    root = data_root(data_dir)
    bundle = {
        doc["path"]: doc["content"]
        for doc in read_overlay_bundle(profile, root)
    }

    cam_patch = {
        "cameras": {
            "active_mode": basic["camera_active_mode"],
            "active_rig_id": basic.get("camera_active_rig_id"),
        }
    }
    runtime_patch = {
        "runtime": {
            "image_encoding": {
                "frame_quality": basic["frame_quality"],
                "decision_quality": basic["decision_quality"],
            },
            "interaction": {
                "prompt_timeout_s": basic["prompt_timeout_s"],
                "confirm_timeout_s": basic["confirm_timeout_s"],
                "confirm_watchdog_s": basic["confirm_watchdog_s"],
            },
            "run_registry": {
                "recent_limit": basic["recent_limit"],
            },
        }
    }

    robot_ip = (basic.get("robot_ip") or "").strip()
    robot_vendor = (basic.get("robot_vendor") or "").strip()
    robot_root: dict[str, Any] = {}
    if robot_vendor:
        robot_root["vendor"] = robot_vendor
    if robot_ip:
        # Vendor-block schema (Phase Q-C): the flat ``connection:`` key
        # is gone. Route the IP into the matching vendor block. If no
        # vendor is set in the patch, default to UR so existing
        # operator workflows that only edit the IP keep working.
        target_vendor = robot_vendor or "ur"
        if target_vendor == "ur":
            robot_root["ur"] = {"ip": robot_ip}
        elif target_vendor == "kuka":
            robot_root["kuka"] = {"controller_ip": robot_ip}
        # Other vendors (sim, dummy) have no operator-editable IP.
    robot_patch = {"robot": robot_root} if robot_root else None

    bundle["camera/cam.yaml"] = _merge_yaml_text(bundle.get("camera/cam.yaml", ""), cam_patch)
    bundle["app/runtime.yaml"] = _merge_yaml_text(bundle.get("app/runtime.yaml", ""), runtime_patch)
    if robot_patch is not None:
        bundle["robot/robot.yaml"] = _merge_yaml_text(bundle.get("robot/robot.yaml", ""), robot_patch)

    docs = [{"path": path, "content": content} for path, content in bundle.items()]
    return save_overlay_bundle(profile, docs, data_dir=root)


def apply_robot_settings(
    profile: str,
    robot_patch: dict[str, Any],
    *,
    data_dir: str | Path | None = None,
    pipeline_running: bool = False,
) -> dict[str, Any]:
    """Merge a typed robot-only patch into the profile's ``robot/robot.yaml``.

    The caller passes the *contents* of the ``robot:`` block (without the
    ``robot:`` key itself). Only top-level keys present in ``robot_patch``
    are touched; nested mappings are deep-merged into the existing
    overlay so partial edits do not erase unrelated fields.

    The result goes through the standard ``save_overlay_bundle`` round-trip
    so any schema violation surfaces as ``ConfigError`` before files change.
    """
    _reject_if_pipeline_running(pipeline_running)
    validate_profile_name(profile)
    root = data_root(data_dir)
    bundle = {
        doc["path"]: doc["content"]
        for doc in read_overlay_bundle(profile, root)
    }
    if robot_patch:
        bundle["robot/robot.yaml"] = _merge_yaml_text(
            bundle.get("robot/robot.yaml", ""),
            {"robot": robot_patch},
        )
    docs = [{"path": path, "content": content} for path, content in bundle.items()]
    return save_overlay_bundle(profile, docs, data_dir=root)


def validate_profile_name(profile: str) -> None:
    if not profile or not profile.strip():
        raise ConfigError("profile name must not be empty")
    if not _PROFILE_RE.match(profile):
        raise ConfigError(
            "profile name may contain only letters, digits, '_' and '-'"
        )


def _reject_if_pipeline_running(pipeline_running: bool) -> None:
    if pipeline_running:
        raise ConfigError(
            "PIPELINE_ALREADY_RUNNING: configuration changes are disabled while "
            "a pipeline is running"
        )


@contextmanager
def _profile_context(profile: str | None) -> Iterator[None]:
    prev = active_profile()
    try:
        set_active_profile(profile)
        reload_config()
        yield
    finally:
        set_active_profile(prev)
        reload_config()


def _write_bundle_to_tree(root: Path, profile: str, documents: Iterable[dict[str, Any]]) -> None:
    valid_paths = {relpath(p, root): p for p in editable_base_files(root)}
    for doc in documents:
        path = str(doc["path"])
        if path not in valid_paths:
            raise ConfigError(f"unknown config document path: {path}")
        target = overlay_path(valid_paths[path], profile)
        content = str(doc.get("content", ""))
        if not content.strip():
            if target.exists():
                target.unlink()
            continue
        target.write_text(_normalise_yaml_text(content), encoding="utf-8")


def _merge_yaml_text(existing_text: str, patch: dict[str, Any]) -> str:
    base = yaml.safe_load(existing_text) if existing_text.strip() else {}
    if base is None:
        base = {}
    if not isinstance(base, dict):
        raise ConfigError("overlay document must contain a YAML mapping at top level")
    merged = _deep_merge(base, patch)
    return _dump_yaml(merged)


def _dump_yaml(data: dict[str, Any]) -> str:
    return yaml.safe_dump(
        data,
        sort_keys=False,
        allow_unicode=False,
        default_flow_style=False,
    )


def _normalise_yaml_text(text: str) -> str:
    doc = yaml.safe_load(text) if text.strip() else {}
    if doc is None:
        return ""
    if not isinstance(doc, dict):
        raise ConfigError("overlay document must contain a YAML mapping at top level")
    return _dump_yaml(doc)


__all__ = [
    "apply_basic_settings",
    "available_profiles",
    "editable_base_files",
    "load_effective_config_dict",
    "profile_status",
    "read_overlay_bundle",
    "save_overlay_bundle",
    "switch_profile",
    "validate_overlay_bundle",
] 