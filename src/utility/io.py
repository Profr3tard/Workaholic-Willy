"""Small, cross-platform JSON I/O helpers.

All helpers force UTF-8 encoding so files written on Windows are byte-
identical to files written on Linux, and we never accidentally pick up
``cp1252`` from the locale.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

__all__ = ["atomic_write_text", "dump_json", "load_json"]


def load_json(path: os.PathLike[str] | str) -> Any:
    """Load JSON from ``path`` using UTF-8."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dump_json(
    data: Any,
    path: os.PathLike[str] | str,
    *,
    indent: int = 2,
    sort_keys: bool = False,
    atomic: bool = True,
) -> None:
    """Write ``data`` as JSON to ``path`` using UTF-8."""
    text = json.dumps(data, indent=indent, sort_keys=sort_keys, ensure_ascii=False)
    if atomic:
        atomic_write_text(path, text)
    else:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)


def atomic_write_text(
    path: os.PathLike[str] | str,
    text: str,
    *,
    encoding: str = "utf-8",
) -> None:
    """Write ``text`` to ``path`` atomically (tmp-file + rename)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=target.name + ".",
        suffix=".tmp",
        dir=str(target.parent),
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as f:
            f.write(text)
        os.replace(tmp, target)
    except Exception:
        # Clean up the temp file on failure; don't mask the original error.
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
