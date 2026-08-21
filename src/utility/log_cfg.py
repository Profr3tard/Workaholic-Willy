"""Logging configuration helpers for Workaholic-Willy."""

import logging
import os
import threading
from logging.handlers import RotatingFileHandler

FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Lock to protect the global registries of file and console handlers.
_LOCK = threading.RLock()

# Process-wide registries of file handlers so that multiple loggers writing to
# the same file don't each create their own handler
_FILE_HANDLERS: dict[str, RotatingFileHandler] = {}

# Single shared console handler so every logger writes to stdout exactly
# once, regardless of how many ``create_logger`` calls were made.
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

    Args:
        name:           Logger name (typically the class or module).
        log_file:       Log file name (relative to ``log_dir``).
        level:          Logging level (default: ``logging.INFO``).
        log_dir:        Directory for log files; created if missing.
        max_bytes:      Max size per log file before rotation.
        backup_count:   Number of rotated files to keep.
        aggregate_file: Optional SECOND file sink shared across many loggers (e.g. a package-wide
                        ``robot.log``).
        aggregate_dir:  Directory for ``aggregate_file`` (defaults to ``log_dir``).

    Returns:
        A configured ``logging.Logger`` ready for use.
    """
    logger = logging.getLogger(name)

    with _LOCK:
        if logger.handlers:
            return logger  # already configured for this name

        logger.setLevel(level)

        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_file)

        formatter = logging.Formatter(FORMAT)

        logger.addHandler(
            _get_file_handler(log_path, max_bytes, backup_count, formatter)
        )
        if aggregate_file is not None:
            agg_dir = aggregate_dir or log_dir
            os.makedirs(agg_dir, exist_ok=True)
            agg_path = os.path.join(agg_dir, aggregate_file)
            logger.addHandler(
                _get_file_handler(agg_path, max_bytes, backup_count, formatter)
            )
        logger.addHandler(_get_console_handler(formatter))

        logger.propagate = False

    return logger
