"""Cross-platform path helpers.

Every filesystem-layout decision lives here, so the code behaves identically on
Windows and Linux. All paths are ``pathlib.Path``, never raw strings with
``\\`` separators.

Three environment variables override the defaults:

* ``WILLY_PROJECT_ROOT`` the project root, otherwise the folder that holds
  ``src/``.
* ``WILLY_LOG_DIR``      the logs root, otherwise ``<root>/logs``.
* ``WILLY_DEBUG_DIR``    the debug image root, otherwise ``<logs>/debug``.

Two events reach ``logs/utility/paths.log``: a project root that had to be
guessed, and the bounded-bucket rotation in :func:`rotate_files`, the only place
in this package that deletes a file.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from src.utility.constants import PATHS_LOG_FILE, utility_logger

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from logging import Logger

__all__ = [
    "debug_dir",
    "ensure_dir",
    "logs_dir",
    "project_root",
    "rotate_files",
]


def _log() -> Logger:
    """Logger for this module, built on first use. See :func:`utility_logger`.

    Call it only where a line is actually going to be emitted. ``create_logger``
    opens the rotating file eagerly, and ``project_root`` runs on every path
    lookup here, so touching this accessor on a hot happy path leaves a
    permanently empty ``paths.log`` behind even at debug level.
    """
    return utility_logger("UtilityPaths", PATHS_LOG_FILE)


def _from_env(var: str) -> Path | None:
    raw = os.environ.get(var)
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def project_root() -> Path:
    """Return the project root as a ``Path``.

    Resolution order:

    1. The ``WILLY_PROJECT_ROOT`` environment variable, if set.
    2. The nearest ancestor directory that holds ``src`` together with either
       ``requirements.txt`` or ``pyproject.toml``.
    3. Two levels up from this file, which sits at ``src/utility/paths.py``.
    """
    env = _from_env("WILLY_PROJECT_ROOT")
    if env is not None:
        return env

    here = Path(__file__).resolve()
    for ancestor in here.parents:
        if (ancestor / "src").is_dir() and (
            (ancestor / "requirements.txt").is_file()
            or (ancestor / "pyproject.toml").is_file()
        ):
            return ancestor

    # utility/paths.py -> utility -> src -> <root>
    #
    # Nothing matched, so this is a guess from the depth of this file alone. It
    # holds in a normal checkout and breaks once the package is vendored or
    # installed elsewhere, where every caller building a path off the root
    # inherits the mistake silently. The warning below names the fix.
    fallback = here.parents[2]
    _log().warning(
        "project_root: no ancestor of %s holds src/ plus requirements.txt or "
        "pyproject.toml; guessing %s from path depth. Set WILLY_PROJECT_ROOT to be sure.",
        here,
        fallback,
    )
    return fallback


def logs_dir() -> Path:
    """Return the directory where application logs live, created on demand."""
    env = _from_env("WILLY_LOG_DIR")
    base = env if env is not None else project_root() / "logs"
    base.mkdir(parents=True, exist_ok=True)
    return base


def ensure_dir(path: os.PathLike[str] | str) -> Path:
    """Create ``path`` and its parents if needed, and return it as a ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def rotate_files(
    directory: os.PathLike[str] | str,
    *,
    max_files: int,
    patterns: Iterable[str] = ("*",),
) -> int:
    """Delete the oldest files in ``directory`` so that at most ``max_files`` remain.

    Only files matching one of the ``patterns`` globs count towards the cap.
    Returns the number of files removed. A missing directory is ignored, so this
    can be called before anything has written to the directory.
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
    failed = 0
    for f in to_remove:
        try:
            f.unlink()
            removed += 1
        except OSError:
            # Best effort: another process may hold the file.
            failed += 1
            continue

    # One line per call rather than per file, at info because this is the only
    # helper in the package that deletes data. It runs on every debug_dir()
    # call, and the answerable question afterwards is how many files left which
    # bucket, not which artefact was oldest.
    if removed:
        _log().info(
            "rotate_files: removed %d of %d matching file(s) from %s (cap %d)",
            removed,
            len(found),
            d,
            max_files,
        )
    if failed:
        # A locked file is the one way the cap quietly stops holding: the bucket
        # keeps growing and nothing else reports it.
        _log().warning(
            "rotate_files: %d file(s) in %s could not be deleted (held by another "
            "process?); the directory stays above its %d-file cap",
            failed,
            d,
            max_files,
        )
    return removed


def debug_dir(
    subdir: str,
    *,
    max_files: int = 200,
    rotate_patterns: Iterable[str] = ("*.png", "*.jpg", "*.jpeg"),
) -> Path:
    """Return a per-module debug directory under ``logs/debug/<subdir>``.

    ``subdir`` is a single path segment: a separator or ``..`` raises
    ``ValueError``. The directory is created and then rotated through
    :func:`rotate_files`, so the bucket never grows unbounded. Write debug
    artefacts here rather than into the current working directory.
    """
    if not subdir or any(ch in subdir for ch in ("/", "\\", "..")):
        raise ValueError(f"debug_dir: invalid subdir {subdir!r}")

    env = _from_env("WILLY_DEBUG_DIR")
    base = env if env is not None else logs_dir() / "debug"
    target = base / subdir
    target.mkdir(parents=True, exist_ok=True)
    # No log line here: this runs once per saved debug image, and rotate_files
    # names the bucket whenever it deletes anything.
    rotate_files(target, max_files=max_files, patterns=rotate_patterns)
    return target
