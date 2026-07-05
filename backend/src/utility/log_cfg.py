import logging
import os
import threading
from logging.handlers import RotatingFileHandler

FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Guards the process-wide handler registries below against concurrent
# create_logger() calls (the Windows multi-writer race this module exists to
# prevent). Reentrant so create_logger can hold it while calling the
# _get_*_handler helpers, which acquire it too.
_LOCK = threading.RLock()

# Process-wide registry of file handlers, keyed by *absolute* log-file path.
# Multiple loggers that target the same file (e.g. all classes in the
# ``robot`` package writing to ``robot.log``) share a single handler. This
# avoids the multiple-writer problem on Windows where each
# ``RotatingFileHandler`` opens its own file descriptor and rotation can
# race.
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
) -> logging.Logger:
    """Create and configure a logger with file rotation and console output.

    Loggers are keyed by ``name``, so calling this function twice with the
    same name returns the same logger. File handlers are *shared by path*:
    every logger pointing at the same physical file uses one
    ``RotatingFileHandler`` instance, which keeps file rotation atomic
    even when many components write to the same log.

    Args:
        name:         Logger name (typically the class or module).
        log_file:     Log file name (relative to ``log_dir``).
        level:        Logging level (default: ``logging.INFO``).
        log_dir:      Directory for log files; created if missing.
        max_bytes:    Max size per log file before rotation.
        backup_count: Number of rotated files to keep.

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
        logger.addHandler(_get_console_handler(formatter))

        logger.propagate = False

    return logger
