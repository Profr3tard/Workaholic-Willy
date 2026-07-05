"""Bounded polling helper for the Whisper streaming STT loop.

Kept torch / sounddevice-free and in its own module so the loop bound is
unit-testable on any host (the parent ``speech_to_text`` module imports torch).
"""

from __future__ import annotations

from collections.abc import Callable


def poll_until_text(
    *,
    drain: Callable[[], str | None],
    sleep: Callable[[int], None],
    now: Callable[[], float],
    timeout_s: float | None = None,
    max_attempts: int | None = None,
    poll_interval_ms: int = 100,
) -> str | None:
    """Poll ``drain`` until it yields truthy text or a bound is reached.

    Parameters
    ----------
    drain
        One transcription attempt; returns text or ``None`` / empty string.
    sleep
        Sleep callback taking milliseconds (e.g. ``sounddevice.sleep``).
    now
        Monotonic clock in seconds (e.g. ``time.monotonic``).
    timeout_s
        Wall-clock budget; ``None`` disables the time bound.
    max_attempts
        Maximum drain attempts; ``None`` disables the attempt bound.
    poll_interval_ms
        Milliseconds to sleep between attempts.

    Returns
    -------
    str | None
        The first truthy text, or ``None`` once a bound is hit. If BOTH bounds
        are ``None`` this can still block indefinitely (preserves the legacy
        contract for callers that explicitly opt out of bounding).
    """
    start = now()
    attempts = 0
    while True:
        sleep(poll_interval_ms)
        text = drain()
        if text:
            return text
        attempts += 1
        if max_attempts is not None and attempts >= max_attempts:
            return None
        if timeout_s is not None and (now() - start) >= timeout_s:
            return None