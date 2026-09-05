"""Cross-platform JSON and file I/O helpers.

Every helper forces UTF-8, so a file written on Windows is byte-identical to
one written on Linux and the locale never contributes ``cp1252``.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, BinaryIO, Callable

__all__ = ["atomic_write_bytes", "atomic_write_text", "dump_json", "load_json"]


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
    """Write ``data`` as JSON to ``path`` using UTF-8.

    With ``atomic=True``, the default, the file is written to a temporary
    sibling and moved into place with ``os.replace``, so a reader never sees a
    half-written document.
    """
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
    """Write ``text`` to ``path`` atomically, through a temp file and a rename.

    This behaves the same on Windows and Linux because ``os.replace`` is atomic
    on both when the source and destination sit on one filesystem, which they do
    here: the temp file is created in the parent directory of the destination.
    """
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
        # Clean up the temp file on failure without masking the original error.
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def atomic_write_bytes(
    path: os.PathLike[str] | str,
    write: "Callable[[BinaryIO], None]",
    *,
    fsync: bool = True,
) -> None:
    """Write binary content to ``path`` atomically, by handing ``write`` an open file.

    ``atomic_write_text`` takes a finished string; this takes a writer, since a
    torch checkpoint is produced by one, ``torch.save(payload, handle)``, and
    can be tens of megabytes.

    ``fsync`` before the rename is what makes the guarantee survive a power loss
    rather than only a crashed process: ``os.replace`` publishes the directory
    entry, and without the flush that entry can be durable while the contents
    are not.

    On Windows ``os.replace`` raises ``PermissionError``, winerror 5, if any
    process holds the destination open, and this helper does not swallow it: for
    an artifact a silent non-write is the worse outcome. A caller for whom a
    failed write must not end the work, a training checkpoint being one, has to
    catch ``OSError`` itself.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            write(handle)
            handle.flush()
            if fsync:
                os.fsync(handle.fileno())
        os.replace(tmp, target)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
