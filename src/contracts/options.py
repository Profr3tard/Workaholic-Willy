""""The caller did not choose this", as data: the one sentinel and the one precedence rule.

Why a sentinel and not `None`: for several settings here `None` is a real,
chosen value. `train_units=None` means every unit, `control=None` means the
corpus labels, `labelled_unit_share=None` means no rebalancing,
`baseline_pick_rate=None` means skip the non-regression key, `profile=None`
means the base tree with no overlays. If "not chosen" were also `None`, a named
bundle could not tell "train on everything" from "I said nothing", and would
overwrite the first while believing it was filling in the second.

Why not a comparison against the default: it cannot answer the question. A
setting that is chosen and happens to equal the default is indistinguishable
from one nobody touched. Inspecting `sys.argv` instead works from a shell and is
meaningless anywhere else, because under pytest or inside a service it holds the
runner's arguments and the flag never appears. The state has to be data.

The argparse rule is `default=UNSET`, not `argparse.SUPPRESS`. The namespace
then always carries the attribute, holding `UNSET` when nobody typed the flag,
so the CLI hands the same data to the same resolver as a Python caller.
`SUPPRESS` removes the attribute, turning every unmigrated read into an
`AttributeError`, and combined with `parents=` and `set_defaults` it
reintroduces the shared Action mutation that `src/config/__main__.py` documents.

`x is UNSET` is correct at run time and narrows nothing for a type checker,
because `_Unset` is a plain class rather than a singleton mypy can reason about.
Use :func:`chosen`, which is a `TypeGuard`, or :func:`resolve`, which is the
same test written once.
"""

from __future__ import annotations

from typing import Final, TypeGuard, TypeVar, Union

__all__ = ["UNSET", "Maybe", "chosen", "resolve"]


class _Unset:
    """The type of :data:`UNSET`. One value, compared by identity, never by equality.

    Kept private so `UNSET` is the only name a caller needs. The type stays
    importable for an annotation such as `int | _Unset`.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover (debugging aid)
        return "UNSET"

    def __bool__(self) -> bool:
        # False, so an unchosen value behaves like an empty one in a boolean context. Truthiness
        # cannot tell an unchosen field from a chosen 0 or 0.0, and both are meaningful here: a
        # 0.0 threshold means "do not apply it". Use chosen() for the question this cannot answer.
        return False


#: "The caller did not choose this." Compare with `is`, never with `==` or truthiness.
UNSET: Final[_Unset] = _Unset()

_T = TypeVar("_T")

#: A value that may not have been chosen. `Maybe[int]` is `int | _Unset`.
Maybe = Union[_T, _Unset]


def chosen(value: "Maybe[_T]") -> "TypeGuard[_T]":
    """Did the caller actually choose this? Narrows the value for a type checker when true.

    Keeps the identity test in one place so nobody writes `!= UNSET`, which
    works today and stops working the moment anything defines `__eq__`.

    It is a `TypeGuard` because `x is UNSET` narrows nothing: that form only
    works for a singleton a type checker can reason about, and `_Unset` is a
    plain class, so a value stays `str | _Unset | None` in the branch where it
    provably cannot be `UNSET`. `isinstance(value, _Unset)` narrows correctly
    but would force every caller to import a private name to use a public
    sentinel. This function is the public form of that test.

    `TypeGuard` narrows where it is true and not where it is false, so write the
    `if chosen(x):` arm as the one that uses the value::

        if chosen(profile):
            chain = validate(profile)      # profile is str | None here
        else:
            chain = from_environment()

    The reversed shape type-checks no better than `is UNSET` did. `TypeIs`,
    which narrows both arms, needs Python 3.13 or `typing_extensions`; this
    project is on 3.11 with neither.
    """
    return not isinstance(value, _Unset)


def resolve(name: str, *layers: "Maybe[_T]") -> _T:
    """The first layer that was actually chosen, most significant first.

    The precedence of this repository, in one function so it cannot be written
    down twice with two different answers::

        max_attempts = resolve("max_attempts", explicit, from_config, 5)
        #                                      ^^^^^^^^  ^^^^^^^^^^^  ^
        #                                      caller    YAML         code default

    `AutonomousGraspService.from_robot_config` resolves `max_attempts` in that
    order already: the sentinel first, then the config block, then a literal.

    The last layer must be a real value. Falling off the end is a programming
    error rather than a runtime condition, so it raises instead of returning
    `None`: a caller handed `None` would carry it into a plan and fail somewhere
    unrelated with no trace of which setting was missing. `name` exists so the
    refusal can say which one.
    """
    for layer in layers:
        if not isinstance(layer, _Unset):
            return layer
    raise ValueError(
        f"nothing supplied a value for {name!r}: every layer was UNSET. The last layer passed to "
        f"resolve() is the code default and must always be a real value."
    )
