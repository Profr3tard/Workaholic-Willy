"""What a verb hands back: the two halves of a result, named once.

This file exists to prevent a result computed in one place and formatted in two.
`config/explain.py` already makes the argument: `KeyExplanation` was split out
from its text so the operator console and the terminal could not drift, because
re-deriving "which layer won" in the browser would be a second implementation of
the one question that module answers. That reasoning applied to one class; this
names it for all of them.

Two protocols rather than one, because `render()` and `to_dict()` are separate
obligations and classes legitimately carry only one. `IOSnapshot` is a bench
reading and `TransitionResult` is one measured edge: both describe themselves to
a person, and neither owes anyone a wire format. A combined protocol would have
charged them for a consumer that does not exist. The halves compose structurally
instead, so a class that has both satisfies both, with no base class, no
registration and no import.

Nothing imports this at run time, which is the point. Both are
`@runtime_checkable` Protocols, so an implementer writes the method and is done.
That is why this package adds no edge to the dependency stack. This module
imports `typing` and nothing else; if it ever needs a second import, the thing
being added does not belong here.

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
    field is a `summary` or a `describe`, keeps that name, and is not this. The
    distinction is load-bearing rather than stylistic:
    `src/robot/safety/planning/environment.py` holds one-line `@property`
    fragments that its own whole-object method composes out of, and naming all
    three the same would make the composition unreadable.

    Three rules:

    Zero arguments. A renderer that takes options has two outputs, and then the
    CLI prints one and the console prints the other, which is the drift this
    file exists to stop. Options belong in the object before it is rendered.

    ASCII only. This terminal is cp1252, where one decorated glyph in printed
    text raises `UnicodeEncodeError` instead of printing. Rendered text is
    printed, so it is under that rule. Put markers in the docstring, where they
    are read and never encoded.

    No trailing newline. The caller owns the line break, because a caller that
    wants the text inside a larger block cannot remove one it did not ask for.
    `PreflightReport.render()` ends on `"\\n".join(lines)`; match it.
    """

    def render(self) -> str: ...


@runtime_checkable
class Structured(Protocol):
    """Can hand its content to another program.

    `to_dict` is the established name for this half throughout the repository. It
    is named here so that a contract mentioning only `render()` does not read as
    a ruling that the machine half is unspecified, leaving the next class to
    invent `as_dict`, `payload` or `json()` in good faith.

    Plain data, all the way down. The returned mapping must survive
    `json.dumps` without a custom encoder: no dataclasses, no `StrEnum` members
    (use `.value`), no `Path`, no numpy scalar. A dict that only serialises by
    accident is a wire format that breaks the first time a field is added, and
    several consumers here write bytes that other tools compare.

    A view, never a second computation. `to_dict()` reports what the object
    already holds. The moment it derives a number that `render()` does not, the
    two halves are two answers and the contract has bought the defect it was
    written against.
    """

    def to_dict(self) -> dict[str, Any]: ...
