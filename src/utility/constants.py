"""Shared constants for the utility package.

Centralising the log paths here keeps the magic strings out of the helpers
themselves, the same way :mod:`backend.src.models.constants` does for the
model classes.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:  # pragma: no cover - typing only
    from logging import Logger

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
#: Single directory under which the utility helpers drop their rotating
#: log files. Resolved relative to the process working directory by
#: :func:`backend.src.utility.log_cfg.create_logger`.
UTILITY_LOG_DIR: Final[str] = "logs/backend/utility"

#: Per-module log-file names. Separate files (the models pattern) rather
#: than one aggregate (the robot pattern): these helpers share nothing but
#: a package. A "why is this running on CPU" question and a "where did my
#: debug PNG go" question are never read in the same sitting, so there is
#: no chronological narrative for an aggregate ``utility.log`` to preserve.
DEVICE_LOG_FILE: Final[str] = "device.log"
PATHS_LOG_FILE: Final[str] = "paths.log"


@lru_cache(maxsize=None)
def utility_logger(name: str, log_file: str, level: int = logging.INFO) -> Logger:
    """Return the (cached) logger for one utility module.

    Deliberately a lazy accessor instead of the usual module-scope
    ``logger = create_logger(...)``. ``utility.paths`` is imported by this
    package's ``__init__``, and therefore by nearly every process in the
    repo -- including ones that only resolve a path and never reach a line
    worth logging. Building on first *use* keeps the package's documented
    "no import-time global state" boundary: no log directory is created,
    and no permanently empty ``paths.log`` appears, until something
    actually has something to say. ``lru_cache`` keeps the call sites cheap
    enough to write inline.
    """
    from src.utility.log_cfg import create_logger

    return create_logger(name, log_file, level=level, log_dir=UTILITY_LOG_DIR)
