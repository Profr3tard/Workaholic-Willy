"""Route each prompt, then delegate, the piece that makes three parts into a pipeline."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from src.models.constants import MODELS_LOG_DIR, PERCEPTION_ROUTING_LOG_FILE
from src.models.perception_backend import PerceivedObject
from src.models.routing import (
    PromptRouter,
    Route,
    RouteDecision,
    RuleBasedRouter,
    normalize_simple_prompt,
)
from src.utility.log_cfg import create_logger

__all__ = ["RoutedPerceptionBackend"]

#: One file for the routing verdicts.
_LOG = create_logger(__name__, log_file=PERCEPTION_ROUTING_LOG_FILE, log_dir=MODELS_LOG_DIR)


class RoutedPerceptionBackend:
    """Pick a perception route per prompt and delegate to it.

    Satisfies :class:`~src.models.perception_backend.PerceptionBackend`, so every caller that
    already takes a backend takes this one unchanged.
    """

    def __init__(
        self,
        *,
        simple_factory: Callable[[], Any],
        vlm_factory: Callable[[], Any],
        router: PromptRouter | None = None,
        normalize: bool = True,
        on_decision: Callable[[RouteDecision], None] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._router: PromptRouter = router or RuleBasedRouter()
        self._factories: dict[Route, Callable[[], Any]] = {
            Route.SIMPLE: simple_factory,
            Route.VLM: vlm_factory,
        }
        self._built: dict[Route, Any] = {}
        self._normalize = normalize
        self._on_decision = on_decision
        self._log = logger or _LOG
        self._last_decision: RouteDecision | None = None

    @property
    def last_decision(self) -> RouteDecision | None:
        """The most recent routing verdict, for telemetry. ``None`` before the first ``perceive``."""
        return self._last_decision

    def built_routes(self) -> tuple[Route, ...]:
        """Which routes have actually been constructed, what is holding VRAM right now."""
        return tuple(self._built)

    def _backend_for(self, route: Route) -> Any:
        backend = self._built.get(route)
        if backend is None:
            self._log.info("building the %s perception route (first use)", route)
            backend = self._factories[route]()
            self._built[route] = backend
        return backend

    def perceive(self, image_bgr: Any, prompt: str) -> tuple[PerceivedObject, ...]:
        decision = self._router.route(prompt)
        self._last_decision = decision
        if self._on_decision is not None:
            try:
                self._on_decision(decision)
            except Exception:  # noqa: BLE001 - telemetry must never break a pick
                self._log.debug("route-decision callback raised; continuing", exc_info=True)

        self._log.info("prompt %r -> %s", prompt, decision.describe())
        text = prompt
        if decision.route is Route.SIMPLE and self._normalize:
            text = normalize_simple_prompt(prompt)

        # NOT guarded: if the chosen route cannot run, that is the route's own contract to enforce.
        return tuple(self._backend_for(decision.route).perceive(image_bgr, text))
