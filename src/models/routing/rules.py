"""The deterministic rules behind :class:`RuleBasedRouter`: vocabulary in, route out, no model.

Rules rather than a classifier, because the router runs before any weights load, on every pick, and
its mistakes are expensive in both directions: a false simple grasps the wrong object, a false VLM
costs seconds and VRAM. A rule set is auditable, in that when it is wrong you can see exactly which
word did it, and it is free. The :class:`~.decision.PromptRouter` seam exists for the day a learned
judge earns its place with measurements.

The order matters because several rules can fire on one prompt ("nicht der kaputte Würfel" is
non-English and negated and a state word). The first match wins, and the order runs most-specific
first, so the reason an operator sees is the most informative one available rather than an accident.

The vocabularies are English and German only, German because that is the language the cell is
operated in. Any other language is detected only if it carries non-ASCII letters, so a non-ASCII-free
prompt in, say, Dutch is routed simple and grounds badly. That is a deliberate limit: widening it
means either a language-ID model (a weight load to decide whether to load weights) or a vocabulary
per language, and neither is justified before the second operator language exists.
"""

from __future__ import annotations

import re
import unicodedata

from .decision import PromptSignals, Route, RouteDecision, RouteReason

__all__ = ["RuleBasedRouter", "analyse", "route", "MAX_SIMPLE_WORDS", "MULTI_ATTRIBUTE_THRESHOLD"]

#: Past this many words a prompt has stopped being a noun phrase and started being an instruction.
#: Judgement, not measurement, as the module docstring of :mod:`.decision` says. Six covers every
#: realistic target description in this cell ("the blue plastic box" is four); "pick up the thing i
#: pointed at before" is eight and is not a noun phrase.
MAX_SIMPLE_WORDS = 6

#: How many attributes before binding them becomes the hard part. GroundingDINO is reliable on "small
#: red cube"; it degrades when the phrase carries enough modifiers that which-adjective-goes-with-which
#: -noun starts to matter. Set at 3 deliberately: 2 would route most ordinary prompts to the VLM and
#: throw away the fast path's whole reason for existing.
MULTI_ATTRIBUTE_THRESHOLD = 3

# --- vocabularies ---------------------------------------------------------------------------------
# Tokens that are unambiguously German. Ambiguous ones are excluded on purpose: "die" is a German
# article and an English verb, "in"/"an"/"so"/"war" collide too. A false NON_ENGLISH silently doubles
# the cost of every English pick, so this set only holds tokens with no English reading at all.
_GERMAN_MARKERS = frozenset({
    "der", "das", "den", "dem", "des", "ein", "eine", "einen", "einem", "einer", "eines",
    "und", "oder", "nicht", "kein", "keine", "keinen", "alle", "jeder", "jede", "jedes",
    "welche", "welcher", "welches", "welchen", "mit", "ohne", "auf", "unter", "neben", "hinter",
    # "war", "hat" and "leg" are German words and ordinary English ones ("the war memorial"), so they
    # are deliberately absent. This set stays strictly free of English readings.
    "zwischen", "links", "rechts", "oben", "unten", "ist", "sind", "waren", "haben",
    "greif", "greife", "nimm", "hole", "lege", "bitte", "dann", "noch", "auch",
    "kaputt", "kaputte", "kaputten", "kaputter", "beschaedigt", "beschaedigte", "defekt", "defekte",
    "wuerfel", "teil", "teile", "objekt", "objekte", "kiste", "kisten", "becher", "schachtel",
    "rote", "roten", "roter", "gelbe", "gelben", "blaue", "blauen", "gruene", "gruenen",
    "schwarze", "schwarzen", "weisse", "weissen", "grosse", "grossen", "kleine", "kleinen",
    "groesste", "groessten", "kleinste", "kleinsten", "linke", "rechte", "vordere", "hintere",
})

_RELATIVE_CLAUSE = frozenset({
    "that", "which", "who", "whose", "where", "welche", "welcher", "welches", "welchen", "wo",
})
_NEGATION = frozenset({
    "not", "no", "without", "except", "neither", "nor", "avoid", "ignore", "skip",
    "nicht", "kein", "keine", "keinen", "keiner", "ohne", "ausser", "weder",
})
_STATE_WORDS = frozenset({
    "broken", "damaged", "cracked", "faulty", "defective", "dented", "bent", "chipped", "torn",
    "intact", "undamaged", "empty", "full", "open", "closed", "dirty", "clean", "wet",
    "kaputt", "kaputte", "kaputten", "kaputter", "kaputtes", "beschaedigt", "beschaedigte",
    "beschaedigten", "defekt", "defekte", "defekten", "heil", "leer", "voll", "offen",
    "geschlossen", "schmutzig", "sauber", "verbogen", "gerissen",
})
_COMPARATIVE = frozenset({
    "largest", "smallest", "biggest", "tallest", "shortest", "longest", "widest", "narrowest",
    "heaviest", "lightest", "nearest", "closest", "furthest", "farthest", "leftmost", "rightmost",
    "topmost", "larger", "smaller", "bigger", "taller", "shorter", "longer", "heavier", "lighter",
    # "next" is absent: on its own it is an ordinal, but "next to" is spatial, and the bigram is the
    # commoner phrasing at a bench. It is handled in _SPATIAL_PHRASES instead.
    "nearer", "closer", "most", "least", "first", "last",
    "groesste", "groessten", "kleinste", "kleinsten", "hoechste", "laengste", "kuerzeste",
    "schwerste", "leichteste", "naechste", "vorderste", "hinterste", "oberste", "unterste",
    "groesser", "kleiner", "hoeher", "laenger", "schwerer", "leichter", "naeher",
})
_SPATIAL = frozenset({
    "above", "below", "beneath", "under", "underneath", "behind", "beside", "between", "inside",
    "outside", "near", "onto", "atop", "adjacent", "opposite", "front",
    "ueber", "unter", "hinter", "neben", "zwischen", "innerhalb", "ausserhalb", "vor", "darunter",
    "darauf", "daneben", "dahinter", "gegenueber",
})
#: Spatial relations that only exist as several words. Matched against the joined token stream, because
#: their parts are individually harmless: "next" is an ordinal, "front" and "top" are nouns.
_SPATIAL_PHRASES = (
    "next to", "in front of", "on top of", "close to", "far from", "away from", "to the left of",
    "to the right of",
)

_QUANTIFIER = frozenset({
    "all", "every", "each", "both", "any", "several", "many", "few", "none",
    "alle", "jeder", "jede", "jedes", "jeden", "beide", "mehrere", "viele", "einige", "keines",
})
_CONJUNCTION = frozenset({"and", "or", "plus", "then", "also", "und", "oder", "sowie", "dann", "auch"})

#: Attribute-bearing modifiers the simple path does handle individually. Counting them is only about
#: how many must be bound at once, so this list is colours + sizes + common shapes/materials rather
#: than an attempt at a full adjective lexicon.
_ATTRIBUTES = frozenset({
    "red", "green", "blue", "yellow", "orange", "purple", "pink", "brown", "black", "white", "grey",
    "gray", "silver", "golden", "transparent", "shiny", "matte", "striped", "spotted",
    "big", "large", "small", "tiny", "huge", "long", "short", "tall", "flat", "thin", "thick",
    "wide", "narrow", "round", "square", "rectangular", "cylindrical", "curved", "pointed",
    "metal", "metallic", "plastic", "wooden", "cardboard", "rubber", "glass", "paper",
    "rot", "rote", "roten", "roter", "gruen", "gruene", "gruenen", "blau", "blaue", "blauen",
    "gelb", "gelbe", "gelben", "schwarz", "schwarze", "schwarzen", "weiss", "weisse", "weissen",
    "gross", "grosse", "grossen", "klein", "kleine", "kleinen", "lang", "lange", "kurz", "kurze",
    "rund", "runde", "flach", "flache", "duenn", "duenne", "dick", "dicke", "breit", "schmal",
    "metall", "metallene", "kunststoff", "holz", "holzerne", "glas", "papier", "pappe",
})

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

#: German umlauts and eszett folded to their ASCII digraphs, so the vocabularies above can be written
#: in plain ASCII and still match "Würfel" / "beschädigt" / "größte" as typed.
_FOLD = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", "Ä": "ae", "Ö": "oe", "Ü": "ue"})


def _tokens(prompt: str) -> tuple[list[str], bool]:
    """``(folded lowercase words, sawNonAscii)``.

    The non-ASCII flag is the only cross-language signal here: a prompt carrying accented letters is
    not English, whatever language it actually is. It is computed before folding, because folding is
    exactly what would erase it.
    """
    lowered = prompt.lower()
    saw_non_ascii = any(
        not char.isascii() and unicodedata.category(char).startswith("L") for char in lowered
    )
    return _WORD_RE.findall(lowered.translate(_FOLD)), saw_non_ascii


def analyse(prompt: str) -> PromptSignals:
    """Measure a prompt. Pure, total, and free: no model, no I/O, no exceptions for bad input."""
    words, saw_non_ascii = _tokens(prompt)
    unique = set(words)
    joined = " ".join(words)
    return PromptSignals(
        words=len(words),
        attributes=sum(1 for word in words if word in _ATTRIBUTES),
        conjunctions=sum(1 for word in words if word in _CONJUNCTION),
        non_english=saw_non_ascii or bool(unique & _GERMAN_MARKERS),
        has_relative_clause=bool(unique & _RELATIVE_CLAUSE),
        has_negation=bool(unique & _NEGATION),
        has_state_word=bool(unique & _STATE_WORDS),
        has_comparative=bool(unique & _COMPARATIVE),
        has_spatial_relation=bool(unique & _SPATIAL) or any(p in joined for p in _SPATIAL_PHRASES),
        has_quantifier=bool(unique & _QUANTIFIER),
    )


def _reason_for(signals: PromptSignals) -> RouteReason | None:
    """The first complex property this prompt has, most-specific first, or ``None`` for the fast path."""
    if signals.non_english:
        return RouteReason.NON_ENGLISH
    if signals.has_relative_clause:
        return RouteReason.RELATIVE_CLAUSE
    if signals.has_negation:
        return RouteReason.NEGATION
    if signals.has_state_word:
        return RouteReason.STATE_WORD
    if signals.has_comparative:
        return RouteReason.COMPARATIVE
    if signals.has_spatial_relation:
        return RouteReason.SPATIAL_RELATION
    if signals.has_quantifier:
        return RouteReason.QUANTIFIER
    if signals.conjunctions:
        return RouteReason.CONJUNCTION
    if signals.attributes >= MULTI_ATTRIBUTE_THRESHOLD:
        return RouteReason.MULTI_ATTRIBUTE
    if signals.words > MAX_SIMPLE_WORDS:
        return RouteReason.TOO_LONG
    return None


def route(prompt: str) -> RouteDecision:
    """Route one prompt. The module-level entry point; :class:`RuleBasedRouter` wraps it for the seam."""
    signals = analyse(prompt)
    if signals.words == 0:
        # Routing is not validation. An empty prompt is a caller bug, and sending it to the expensive
        # route would hide that; the simple path fails on it loudly and cheaply.
        return RouteDecision(Route.SIMPLE, RouteReason.EMPTY_PROMPT, signals)
    reason = _reason_for(signals)
    if reason is None:
        return RouteDecision(Route.SIMPLE, RouteReason.PLAIN_NOUN_PHRASE, signals)
    return RouteDecision(Route.VLM, reason, signals)


class RuleBasedRouter:
    """The default :class:`~.decision.PromptRouter`: stateless, deterministic, model-free."""

    __slots__ = ()

    def route(self, prompt: str) -> RouteDecision:
        return route(prompt)
