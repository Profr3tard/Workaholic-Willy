"""Logger construction: one rotating file per sink, shared across the process.

Every package builds its loggers through :func:`create_logger`, which resolves
the directory, honours ``WILLY_LOG_DIR``, and hands back a logger whose handlers
are shared by path so rotation stays atomic no matter how many components write
to the same file.
"""

import logging
import os
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Guards the process-wide handler registries below against concurrent
# create_logger() calls. Reentrant, so create_logger can hold it while calling
# the _get_*_handler helpers, which acquire it too.
_LOCK = threading.RLock()

# Process-wide registry of file handlers, keyed by absolute log file path.
# Loggers that target the same file, for instance every class in the robot
# package writing to robot.log, share one handler. One handler per logger would
# hit the Windows multiple-writer problem, where each RotatingFileHandler opens
# its own file descriptor and rotation can race.
_FILE_HANDLERS: dict[str, RotatingFileHandler] = {}

# One shared console handler, so every logger writes to stdout exactly once
# regardless of how many create_logger calls were made.
_CONSOLE_HANDLER: logging.StreamHandler | None = None


def _get_file_handler(
    log_path: str,
    max_bytes: int,
    backup_count: int,
    formatter: logging.Formatter,
) -> RotatingFileHandler:
    abs_path = os.path.abspath(log_path)
    with _LOCK:
        handler = _FILE_HANDLERS.get(abs_path)
        if handler is not None:
            return handler

        handler = RotatingFileHandler(
            abs_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
            # Open on first write rather than on construction. create_logger
            # runs at module import, so without this every process that imports
            # a logging module holds the file open whether or not it ever
            # writes a line, and on Windows the parent then cannot rotate it:
            # renaming a file another process holds open is refused with
            # PermissionError, winerror 32. Three pool workers holding three
            # handles each is already enough to block rotation. That rotation
            # failure is silent, because logging swallows handler errors, so
            # the symptom is a rotating log that grows without bound. With
            # delay, a process that never logs never opens the file, and a log
            # file appears when something is first written to it rather than
            # at import.
            delay=True,
        )
        handler.setFormatter(formatter)
        _FILE_HANDLERS[abs_path] = handler
        return handler


def _get_console_handler(formatter: logging.Formatter) -> logging.StreamHandler:
    global _CONSOLE_HANDLER
    with _LOCK:
        if _CONSOLE_HANDLER is None:
            _CONSOLE_HANDLER = logging.StreamHandler()
            _CONSOLE_HANDLER.setFormatter(formatter)
        return _CONSOLE_HANDLER


def _make_log_dir(path: str) -> None:
    """Create a log directory, or refuse with a sentence an operator can act on.

    Raises rather than swallowing: a cell whose logs cannot be written is a cell
    whose evidence is gone. The message names the directory that could not be
    made and the variable that chooses it, because this runs during an import,
    where a bare ``os.makedirs`` failure names neither.
    """
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        raise OSError(
            f"cannot create the log directory {path!r} ({type(exc).__name__}: {exc}). Logging is "
            f"set up at import time, so this stops the process before anything runs. Point "
            f"WILLY_LOG_DIR at a writable directory, or clear whatever occupies that path."
        ) from exc


def resolve_log_dir(log_dir: str) -> str:
    """Turn a log directory into an absolute one, honouring ``WILLY_LOG_DIR``.

    Every package constant is a bare relative string such as ``"logs/robot"``.
    Handed straight to ``os.makedirs`` those follow the working directory, so a
    cell started from two directories writes two divergent log trees and neither
    sits next to the checkout. This is the one seam every logger goes through,
    so resolving here through ``logs_dir()`` is what makes ``WILLY_LOG_DIR``
    reach the logs at all.

    The leading ``logs`` segment is stripped, which keeps the resolved path
    unchanged for a run from the repository root. Every constant already begins
    with it and ``logs_dir()`` is the logs directory, so joining them blindly
    would give ``<root>/logs/logs/robot``.

    An absolute directory is returned unchanged.
    """
    if os.path.isabs(log_dir):
        return log_dir
    # Imported here rather than at module level: a top level import would be a
    # cycle, paths to utility.constants to utility_logger to here. constants.py
    # imports create_logger lazily for the same reason. It costs nothing, since
    # the package __init__ imports both paths and this module, so paths is
    # already in sys.modules by the time any caller reaches this line.
    from src.utility.paths import logs_dir  # noqa: PLC0415

    # logs_dir() creates the base itself, so a WILLY_LOG_DIR pointing at a file
    # raises here and never reaches _make_log_dir below.
    try:
        base = Path(logs_dir())
    except OSError as exc:
        raise OSError(
            f"cannot use the log directory ({type(exc).__name__}: {exc}). Logging is set up at "
            f"import time, so this stops the process before anything runs. Point WILLY_LOG_DIR at a "
            f"writable directory, or unset it to use the checkout's own logs/."
        ) from exc
    parts = Path(log_dir).parts
    if parts and parts[0] == "logs":
        parts = parts[1:]
    return str(base.joinpath(*parts))


def create_logger(
    name: str,
    log_file: str,
    level: int = logging.INFO,
    log_dir: str = "logs",
    max_bytes: int = 5_000_000,
    backup_count: int = 3,
    *,
    aggregate_file: str | None = None,
    aggregate_dir: str | None = None,
) -> logging.Logger:
    """Create and configure a logger with file rotation and console output.

    Loggers are keyed by ``name``, so calling this twice with the same name
    returns the same logger, already configured.

    Args:
        name:           Logger name, typically the class or module.
        log_file:       Log file name, relative to ``log_dir``.
        level:          Logging level.
        log_dir:        Directory for log files, created if missing.
        max_bytes:      Maximum size per log file before rotation.
        backup_count:   Number of rotated files to keep.
        aggregate_file: Optional second file sink shared across many loggers,
                        for instance a package-wide ``robot.log``. When set, the
                        logger writes to both its per-module ``log_file`` and
                        this aggregate. ``None`` keeps the single-file behaviour.
        aggregate_dir:  Directory for ``aggregate_file``, defaulting to ``log_dir``.

    Returns:
        A configured ``logging.Logger``.
    """
    logger = logging.getLogger(name)

    with _LOCK:
        if logger.handlers:
            return logger  # already configured for this name

        logger.setLevel(level)

        resolved = resolve_log_dir(log_dir)
        _make_log_dir(resolved)
        log_path = os.path.join(resolved, log_file)

        formatter = logging.Formatter(FORMAT)

        logger.addHandler(
            _get_file_handler(log_path, max_bytes, backup_count, formatter)
        )
        if aggregate_file is not None:
            agg_dir = resolve_log_dir(aggregate_dir) if aggregate_dir else resolved
            _make_log_dir(agg_dir)
            agg_path = os.path.join(agg_dir, aggregate_file)
            logger.addHandler(
                _get_file_handler(agg_path, max_bytes, backup_count, formatter)
            )
        logger.addHandler(_get_console_handler(formatter))

        logger.propagate = False

    return logger
