"""What happens when the VLM cannot be loaded refuse the prompt, or degrade with a warning."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from src.models.constants import MODELS_LOG_DIR, VLM_AVAILABILITY_LOG_FILE
from src.models.perception_backend import PerceivedObject
from src.utility.log_cfg import create_logger

__all__ = ["VlmUnavailableError", "GuardedVlmBackend"]

#: The default sink for this module's two lines.
_LOG = create_logger(__name__, log_file=VLM_AVAILABILITY_LOG_FILE, log_dir=MODELS_LOG_DIR)


class VlmUnavailableError(RuntimeError):
    """The VLM route was required and the model could not be loaded."""

    def __init__(self, model_id: str, cause: BaseException | None = None) -> None:
        detail = f": {type(cause).__name__}: {cause}" if cause is not None else ""
        super().__init__(
            f"the prompt needs the VLM route and {model_id!r} could not be loaded{detail}. "
            f"Install the weights, or set models.pipeline.zero_shot.vlm.on_unavailable=degrade to fall "
            f"back to the phrase grounder (which will be WRONG on prompts of this kind, loudly logged)."
        )
        self.model_id = model_id
        self.cause = cause


class GuardedVlmBackend:
    """Wraps the VLM backend with the ``on_unavailable`` contract."""

    def __init__(
        self,
        *,
        vlm: Any,
        model_id: str,
        degrade: bool = False,
        fallback_factory: Callable[[], Any] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if degrade and fallback_factory is None:
            raise ValueError(
                "on_unavailable='degrade' needs a fallback backend to degrade TO, refusing to "
                "construct a backend whose degraded path is also unavailable."
            )
        self._vlm = vlm
        self._model_id = model_id
        self._degrade = degrade
        self._fallback_factory = fallback_factory
        self._fallback: Any = None
        self._log = logger or _LOG
        #: Set once the VLM has failed to load, so we do not retry a multi-second load per pick.
        self._unavailable: BaseException | None = None

    @property
    def degraded(self) -> bool:
        """True once this backend has fallen back, surfaced by the console and the run report."""
        return self._unavailable is not None

    def perceive(self, image_bgr: Any, prompt: str) -> tuple[PerceivedObject, ...]:
        if self._unavailable is None:
            try:
                return tuple(self._vlm.perceive(image_bgr, prompt))
            except VlmUnavailableError as exc:
                self._unavailable = exc.cause or exc
            except (ImportError, OSError, RuntimeError) as exc:
                # The three shapes a missing/unloadable model actually takes: no dependency, no
                # weights on disk, no VRAM.
                self._unavailable = exc
            if self._unavailable is not None:
                self._log.error(
                    "VLM %s became unavailable: %s: %s",
                    self._model_id, type(self._unavailable).__name__, self._unavailable,
                )

        if self._unavailable is None:  # pragma: no cover - unreachable; kept for narrowing
            return ()
        if not self._degrade:
            raise VlmUnavailableError(self._model_id, self._unavailable)

        self._log.warning(
            "DEGRADED: %r needs the VLM route but %s is unavailable, grounding with the phrase "
            "detector instead.",
            prompt, self._model_id,
        )
        if self._fallback is None:
            assert self._fallback_factory is not None  # guaranteed by __init__
            self._fallback = self._fallback_factory()
        return tuple(self._fallback.perceive(image_bgr, prompt))
