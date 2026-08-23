"""Find the MediaPipe `.task` bundle, or refuse with something an operator can act on."""

from __future__ import annotations

from pathlib import Path
from typing import Final

__all__ = [
    "MEDIAPIPE_AVAILABLE",
    "MEDIAPIPE_INSTALL_HINT",
    "require_mediapipe",
    "resolve_model_file",
]

try:  # pragma: no cover - the branch taken depends on the install, not on the tests
    import mediapipe as _mp  # noqa: F401

    _available = True
except ImportError:  # pragma: no cover - only on installs without the optional extra
    _available = False

#: Whether the optional extra is importable.
MEDIAPIPE_AVAILABLE: Final[bool] = _available

MEDIAPIPE_INSTALL_HINT: Final[str] = (
    "Install the optional hand-detection extra: pip install -r requirements-cpu.txt"
)


def require_mediapipe(what: str) -> None:
    """Raise a named `ImportError` when the optional extra is absent."""
    if not MEDIAPIPE_AVAILABLE:
        raise ImportError(f"mediapipe is required for {what} but is not installed. {MEDIAPIPE_INSTALL_HINT}")


def resolve_model_file(path: str, *, config_key: str, download_url: str) -> str:
    """Return `path` as an absolute string, or raise naming the key, the path and the download."""
    if not path or not path.strip():
        raise FileNotFoundError(
            f"{config_key} is empty, so there is no MediaPipe model to load.\n"
            f"  Set it to the path of the downloaded bundle.\n"
            f"  Download: {download_url}"
        )

    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved

    if not resolved.is_file():
        raise FileNotFoundError(
            f"MediaPipe model file not found: {resolved}\n"
            f"  Configured by: {config_key} = {path!r}\n"
            f"  Download it and put it there:\n"
            f"    {download_url}\n"
            f"  The bundles are NOT committed to this repository (they are binaries, and the\n"
            f"  operator chooses which revision to run)."
        )
    return str(resolved)
