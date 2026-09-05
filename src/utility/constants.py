"""Shared constants for the utility package.

The log paths live here rather than in the helpers that use them, the same way
:mod:`src.models.constants` holds them for the model classes.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from logging import Logger

# Logging

#: The single directory under which the utility helpers drop their rotating log
#: files. A bare relative string, resolved to an absolute path by
#: :func:`src.utility.log_cfg.create_logger`.
UTILITY_LOG_DIR: Final[str] = "logs/utility"

#: Per-module log-file names. One file per helper rather than one aggregate:
#: these helpers share nothing but a package, so a combined ``utility.log``
#: would have no chronological narrative to preserve.
DEVICE_LOG_FILE: Final[str] = "device.log"
PATHS_LOG_FILE: Final[str] = "paths.log"


@lru_cache(maxsize=None)
def utility_logger(name: str, log_file: str, level: int = logging.INFO) -> Logger:
    """Return the cached logger for one utility module.

    A lazy accessor rather than a module-scope ``logger = create_logger(...)``:
    this package's ``__init__`` pulls ``utility.paths`` into nearly every
    process, including ones that only resolve a path and never log. Building on
    first use keeps the package free of import-time global state, so no log
    directory is created and no empty ``paths.log`` appears until something has
    something to say. ``lru_cache`` keeps the call sites cheap enough to write
    inline.
    """
    from src.utility.log_cfg import create_logger

    return create_logger(name, log_file, level=level, log_dir=UTILITY_LOG_DIR)
