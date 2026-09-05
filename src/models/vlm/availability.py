"""What happens when the VLM cannot be loaded: refuse the prompt, or degrade with a warning.

``refuse`` means the prompt is not taken at all; ``degrade`` means the fallback answers it and every
such answer is warned about. Both halves are the owner's stated rule, and the default is ``refuse``,
because silently degrading is the failure this whole route exists to prevent.

The complex prompts that reach the VLM route are the ones the phrase grounder gets confidently
wrong: it returns a high-scoring box for the wrong object rather than admitting defeat. So a quiet
fallback does not mean "slightly worse perception"; it means the cell grasps something the operator
did not ask for, with nothing in the log to say why. Refusing is the safe reading of an ambiguous
situation; degrading is a deliberate choice made in config and always announced.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from src.models.constants import MODELS_LOG_DIR, VLM_AVAILABILITY_LOG_FILE
from src.models.perception_backend import PerceivedObject
from src.utility.log_cfg import create_logger

__all__ = ["VlmUnavailableError", "GuardedVlmBackend"]

#: The default sink for this module's two lines. Under a bare ``getLogger`` the degrade warning
#: below, the one thing this module exists to make impossible to miss, has no handler and is
#: discarded. The name stays ``__name__``, and an injected ``logger=`` still overrides it per
#: instance.
_LOG = create_logger(__name__, log_file=VLM_AVAILABILITY_LOG_FILE, log_dir=MODELS_LOG_DIR)


class VlmUnavailableError(RuntimeError):
    """The VLM route was required and the model could not be loaded.

    Carries the underlying cause so an operator sees why: weights missing, dependency absent, out of
    VRAM, rather than only that something failed. Raised rather than returned as an empty result,
    because "no objects found" and "I could not look" are different answers and the pick loop must
    not treat the second as the first.
    """

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
    """Wraps the VLM backend with the ``on_unavailable`` contract.

    Delegates to ``vlm`` while it works. The first time loading fails it either raises
    :class:`VlmUnavailableError` (``refuse``) or hands over to the fallback (``degrade``), and the
    degrade path logs a warning every time it is used, not once. A one-shot warning scrolls out of a
    terminal and the run then looks normal for an hour; the cost of a repeated line is nothing next
    to a silently wrong grasp.

    The fallback arrives as a factory, not an instance, and is built only if degradation actually
    happens. Constructing it eagerly would load GroundingDINO and SAM2 on every cell that merely
    might degrade, spending seconds and gigabytes of VRAM on a path a healthy cell never runs.
    """

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
                "on_unavailable='degrade' needs a fallback backend to degrade to; refusing to "
                "construct a backend whose degraded path is also unavailable."
            )
        self._vlm = vlm
        self._model_id = model_id
        self._degrade = degrade
        self._fallback_factory = fallback_factory
        self._fallback: Any = None
        self._log = logger or _LOG
        #: Set once the VLM has failed to load, so a multi-second load is not retried per pick.
        self._unavailable: BaseException | None = None

    @property
    def degraded(self) -> bool:
        """True once this backend has fallen back; surfaced by the console and the run report."""
        return self._unavailable is not None

    def perceive(self, image_bgr: Any, prompt: str) -> tuple[PerceivedObject, ...]:
        if self._unavailable is None:
            try:
                return tuple(self._vlm.perceive(image_bgr, prompt))
            except VlmUnavailableError as exc:
                self._unavailable = exc.cause or exc
            except (ImportError, OSError, RuntimeError) as exc:
                # The three shapes a missing or unloadable model takes: no dependency, no weights
                # on disk, no VRAM. Anything else is a real bug and must not be swallowed into a
                # fallback that hides it.
                self._unavailable = exc
            if self._unavailable is not None:
                self._log.error(
                    "VLM %s became unavailable: %s: %s",
                    self._model_id, type(self._unavailable).__name__, self._unavailable,
                )

        if self._unavailable is None:  # pragma: no cover (unreachable; kept for narrowing)
            return ()
        if not self._degrade:
            raise VlmUnavailableError(self._model_id, self._unavailable)

        self._log.warning(
            "DEGRADED: %r needs the VLM route but %s is unavailable; grounding with the phrase "
            "detector instead. It does not fail loudly on prompts of this kind; it returns a "
            "confident box for the WRONG object. Treat this result as unverified.",
            prompt, self._model_id,
        )
        if self._fallback is None:
            assert self._fallback_factory is not None  # guaranteed by __init__
            self._fallback = self._fallback_factory()
        return tuple(self._fallback.perceive(image_bgr, prompt))
