"""Shared constants for the utility package.

Centralising the log paths here keeps the magic strings out of the helpers
themselves, the same way :mod:`src.models.constants` does for the model classes.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from logging import Logger

# Logging

#: The single directory under which the utility helpers drop their rotating log
#: files. Resolved to an absolute path by :func:`src.utility.log_cfg.create_logger`.
UTILITY_LOG_DIR: Final[str] = "logs/utility"

#: Per-module log-file names. Separate files rather than one aggregate, because
#: these helpers share nothing but a package: a device question and a debug
#: image question are never read in the same sitting, so an aggregate
#: ``utility.log`` would have no chronological narrative to preserve.
DEVICE_LOG_FILE: Final[str] = "device.log"
PATHS_LOG_FILE: Final[str] = "paths.log"


@lru_cache(maxsize=None)
def utility_logger(name: str, log_file: str, level: int = logging.INFO) -> Logger:
    """Return the cached logger for one utility module.

    A lazy accessor rather than a module-scope ``logger = create_logger(...)``,
    because ``utility.paths`` reaches nearly every process here through this
    package's ``__init__``, including ones that only resolve a path and never
    log. Building on first use keeps the package free of import-time global
    state: no log directory is created and no empty ``paths.log`` appears until
    something has something to say. ``lru_cache`` keeps the call sites cheap
    enough to write inline.
    """
    from src.utility.log_cfg import create_logger

    return create_logger(name, log_file, level=level, log_dir=UTILITY_LOG_DIR)
