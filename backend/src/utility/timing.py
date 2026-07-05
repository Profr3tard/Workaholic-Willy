"""Timing helpers shared across pipelines and models."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

__all__ = ["now_ms", "timed"]


def now_ms() -> float:
    """Monotonic milliseconds since an arbitrary epoch.

    Use ``now_ms`` for measuring durations — not ``time.time() * 1000``,
    which is wall-clock and can jump backwards on NTP corrections.
    """
    return time.perf_counter() * 1000.0


@contextmanager
def timed(
    label: str = "",
    *,
    logger: logging.Logger | None = None,
    level: int = logging.DEBUG,
) -> Iterator[dict]:
    """Context manager that measures wall-clock elapsed time.

    Yields a dict that exposes ``elapsed_ms`` after the block exits.
    Optionally logs a single line at ``level`` on the given logger.

    Example::

        with timed("detect", logger=self.logger) as t:
            result = self._run()
        result.inference_time_s = t["elapsed_ms"] / 1000.0
    """
    start = time.perf_counter()
    stats: dict = {"elapsed_ms": 0.0}
    try:
        yield stats
    finally:
        stats["elapsed_ms"] = (time.perf_counter() - start) * 1000.0
        if logger is not None:
            logger.log(level, "%s took %.1f ms", label or "<block>", stats["elapsed_ms"])
