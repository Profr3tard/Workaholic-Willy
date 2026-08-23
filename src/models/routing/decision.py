"""The typed answer to "which perception route should this prompt take?"."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Protocol

__all__ = [
    "Route",
    "RouteReason",
    "PromptSignals",
    "RouteDecision",
    "PromptRouter",
]


class Route(StrEnum):
    """Which perception stack handles this prompt."""

    SIMPLE = "simple"  #: GroundingDINO -> SAM2, the fast open-vocabulary path
    VLM = "vlm"        #: a vision-language model that reasons about the phrase, then SAM2


class RouteReason(StrEnum):
    """WHY a prompt was routed, in the router's own words."""

    PLAIN_NOUN_PHRASE = "plain_noun_phrase"      #: -> SIMPLE. Nothing here the phrase grounder cannot do.
    EMPTY_PROMPT = "empty_prompt"                #: -> SIMPLE. Routing is not validation; let it fail honestly.
    NON_ENGLISH = "non_english"                  #: GroundingDINO is trained on English phrases.
    RELATIVE_CLAUSE = "relative_clause"          #: "the cube that is on the box" reference, not description.
    NEGATION = "negation"                        #: "not the red one" the phrase grounder has no NOT.
    STATE_WORD = "state_word"                    #: "broken", "kaputt" a judgement about condition.
    COMPARATIVE = "comparative"                  #: "the largest" requires comparing candidates.
    SPATIAL_RELATION = "spatial_relation"        #: "left of the tray" relates two objects.
    QUANTIFIER = "quantifier"                    #: "every cube" scope over a set.
    CONJUNCTION = "conjunction"                  #: "the cube and the cup" two targets in one phrase.
    MULTI_ATTRIBUTE = "multi_attribute"          #: enough attributes that BINDING them matters.
    TOO_LONG = "too_long"                        #: past any plausible noun phrase: treat as an instruction.


@dataclass(frozen=True, slots=True)
class PromptSignals:
    """What the router measured about a prompt. Typed, not a bare dict, because it is logged."""

    words: int
    attributes: int
    conjunctions: int
    non_english: bool
    has_relative_clause: bool
    has_negation: bool
    has_state_word: bool
    has_comparative: bool
    has_spatial_relation: bool
    has_quantifier: bool

    def to_dict(self) -> dict[str, int | bool]:
        """Flat JSON-safe mapping for the event envelope and ``GraspAttemptRecord.extra``."""
        return dict(asdict(self))


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """One routing verdict: the route, the reason, and the evidence behind it."""

    route: Route
    reason: RouteReason
    signals: PromptSignals

    @property
    def is_vlm(self) -> bool:
        return self.route is Route.VLM

    def to_dict(self) -> dict[str, object]:
        """JSON-safe form for telemetry. Stable keys — consumers key off these."""
        return {"route": str(self.route), "reason": str(self.reason), "signals": self.signals.to_dict()}

    def describe(self) -> str:
        """One operator-readable line, e.g. ``vlm (relative_clause)``."""
        return f"{self.route} ({self.reason})"


class PromptRouter(Protocol):
    """The seam. A learned judge implements this and every caller keeps working."""

    def route(self, prompt: str) -> RouteDecision: ...
