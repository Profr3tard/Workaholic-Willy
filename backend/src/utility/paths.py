"""Cross-platform path helpers for Workaholic-Willy.

Centralises every filesystem-layout decision in one place so the codebase
works identically on Windows and Linux. All paths are ``pathlib.Path`` —
never raw strings with ``\\`` separators.

Overridable via environment variables:

* ``WILLY_PROJECT_ROOT`` — project root (defaults to the folder that
  contains ``backend/``).
* ``WILLY_LOG_DIR``      — logs root (defaults to ``<root>/logs``).
* ``WILLY_DEBUG_DIR``    — debug image root (defaults to ``<logs>/debug``).
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

__all__ = [
    "debug_dir",
    "ensure_dir",
    "logs_dir",
    "project_root",
    "rotate_files",
]


def _from_env(var: str) -> Path | None:
    raw = os.environ.get(var)
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def project_root() -> Path:
    """Return the Workaholic-Willy project root as a ``Path``.

    Resolution order:

    1. ``WILLY_PROJECT_ROOT`` environment variable, if set.
    2. Walk up from this file until a directory containing ``backend``
       and either ``requirements.txt`` or ``pyproject.toml`` is found.
    3. Three levels up from this file (``backend/src/utility/paths.py``
       → project root).
    """
    env = _from_env("WILLY_PROJECT_ROOT")
    if env is not None:
        return env

    here = Path(__file__).resolve()
    for ancestor in here.parents:
        if (ancestor / "backend").is_dir() and (
            (ancestor / "requirements.txt").is_file()
            or (ancestor / "pyproject.toml").is_file()
        ):
            return ancestor

    # utility/paths.py  ->  utility -> src -> backend -> <root>
    return here.parents[3]


def logs_dir() -> Path:
    """Return the directory where application logs live (created lazily)."""
    env = _from_env("WILLY_LOG_DIR")
    base = env if env is not None else project_root() / "logs"
    base.mkdir(parents=True, exist_ok=True)
    return base


def ensure_dir(path: os.PathLike[str] | str) -> Path:
    """Create ``path`` (and parents) if needed; return it as a ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def rotate_files(
    directory: os.PathLike[str] | str,
    *,
    max_files: int,
    patterns: Iterable[str] = ("*",),
) -> int:
    """Delete oldest files in ``directory`` so at most ``max_files`` remain.

    Only files matching any of ``patterns`` (glob) are considered. Returns
    the number of files removed. Silently ignores missing directories so
    it's safe to call eagerly on a fresh install.
    """
    d = Path(directory)
    if not d.is_dir() or max_files <= 0:
        return 0

    found: list[Path] = []
    seen: set[Path] = set()
    for pat in patterns:
        for f in d.glob(pat):
            if f.is_file() and f not in seen:
                seen.add(f)
                found.append(f)

    if len(found) <= max_files:
        return 0

    found.sort(key=lambda p: p.stat().st_mtime)
    to_remove = found[: len(found) - max_files]
    removed = 0
    for f in to_remove:
        try:
            f.unlink()
            removed += 1
        except OSError:
            # Best-effort; another process may hold the file.
            continue
    return removed


def debug_dir(
    subdir: str,
    *,
    max_files: int = 200,
    rotate_patterns: Iterable[str] = ("*.png", "*.jpg", "*.jpeg"),
) -> Path:
    """Return a per-module debug directory under ``logs/debug/<subdir>``.

    Creates the directory and rotates older files so the bucket never
    grows unbounded. Use this instead of dropping debug artefacts into
    the current working directory.
    """
    if not subdir or any(ch in subdir for ch in ("/", "\\", "..")):
        raise ValueError(f"debug_dir: invalid subdir {subdir!r}")

    env = _from_env("WILLY_DEBUG_DIR")
    base = env if env is not None else logs_dir() / "debug"
    target = base / subdir
    target.mkdir(parents=True, exist_ok=True)
    rotate_files(target, max_files=max_files, patterns=rotate_patterns)
    return target
