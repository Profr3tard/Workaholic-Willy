"""What a verb hands back: the two halves of a result, named once.

These name one rule: a result is computed in one place and formatted nowhere
else. `src/config/explain.py` splits `KeyExplanation` out from its text for that
reason, since re-deriving "which layer won" in the browser would be a second
implementation of the one question that module answers, and the browser and the
terminal would then disagree about a cell. That reasoning is applied there to
exactly one class; here the shape covers every result.

Two protocols rather than one, because `render()` and `to_dict()` are separate
obligations and a class legitimately carries only one. Four classes already
define `render()` (`KeyExplanation`, `IOSnapshot`, `TransitionResult`,
`PreflightReport`) and not one of them defines `to_dict()`, so a combined
protocol would be satisfied by zero existing classes and would owe a method at
each of those four sites. `IOSnapshot` is a bench reading and `TransitionResult`
is one measured edge: both describe themselves to a person, and neither owes
anyone a wire format. The halves compose structurally, so a class that has both
satisfies both, with no base class, no registration and no import.

Nothing imports this at run time. Both are `@runtime_checkable` Protocols, so an
implementer writes the method and is done, which is what lets this package add
no edge to the dependency stack. This module imports `typing` and nothing else;
anything needing a second import does not belong here.

`runtime_checkable` checks presence and never signatures, so
`isinstance(x, Rendered)` is true for a class whose `render` takes four required
arguments. That is a limit of the language.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = ["Rendered", "Structured"]


@runtime_checkable
class Rendered(Protocol):
    """Can describe itself to a person, in full, in one call.

    `render()` returns everything the reader needs. A method that describes one
    field is a `summary` or a `describe`, keeps that name, and is not this:
    `src/robot/safety/planning/environment.py` holds one-line `@property`
    fragments that its own whole-object method composes out of, and one name for
    all three would make that composition unreadable.

    Three rules:

    Zero arguments. A renderer that takes options has two outputs, and then the
    CLI prints one and the console the other, which is the drift this contract
    stops. Options belong in the object before it is rendered.

    ASCII only. This terminal is cp1252, where one decorated glyph in printed
    text raises `UnicodeEncodeError` instead of printing, and rendered text is
    printed. Markers belong in the docstring, which is read and never encoded.

    No trailing newline. The caller owns the line break, because a caller
    placing the text inside a larger block cannot remove one it did not ask for.
    `PreflightReport.render()` ends on `"\\n".join(lines)`; match it.
    """

    def render(self) -> str: ...


@runtime_checkable
class Structured(Protocol):
    """Can hand its content to another program.

    `to_dict` is defined at 54 places in the in-scope tree, which makes it this
    repository's established name for this half. It is stated here so that a
    contract mentioning only `render()` does not read as a ruling that the
    machine half is unspecified, leaving the 55th class to invent `as_dict`,
    `payload` or `json()` in good faith.

    Plain data, all the way down. The returned mapping must survive `json.dumps`
    without a custom encoder: no dataclasses, no `StrEnum` members (use
    `.value`), no `Path`, no numpy scalar. A dict that serialises only by
    accident breaks the first time a field is added, and consumers here write
    bytes that other tools compare.

    A view, never a second computation. `to_dict()` reports what the object
    already holds; a number derived here that `render()` does not derive makes
    the two halves two answers, which is the defect this contract is against.
    """

    def to_dict(self) -> dict[str, Any]: ...
