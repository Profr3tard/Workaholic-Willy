"""How much a config field is anyone's business — derived, never hand-maintained.

Each field gets a TIER. Two rules govern what a tier is allowed to be:

  1. **A tier is a display filter, never a gate.** Every field stays settable, ``extra='forbid'`` keeps
     accepting it, and hiding it from a listing removes nothing. Removing a value from a FILE removes
     nothing; removing a FIELD removes a capability, a tier must never be mistaken for the second.
  2. **A required field is never demoted.** 63 fields have no default at all; treating one as "long
     tail" invites defaulting it later, and that would trade a hard load failure for a silent value in
     the FAIL-OPEN direction.

The tiers are DERIVED from what the schema and the loaded cell already say, not declared per field.
"""

from __future__ import annotations

from typing import Any

__all__ = ["TIERS", "tier_for"]

#: Ordered widest-first: a listing shows everything up to and including the requested tier.
TIERS: tuple[str, ...] = ("safety", "site", "tuned", "advanced")

#: Fail-closed switches that live outside ``robot.safety`` but decide whether a guard REFUSES on doubt.
_SAFETY_ELSEWHERE = frozenset({
    "robot.grasping.decision.fail_closed_on_real_hardware",
    "robot.grasping.verification.fail_closed",
    "robot.grasping.uncertainty.fail_closed_threshold",
})


def tier_for(
    path: str,
    *,
    required: bool = False,
    decided: bool = False,
    gated_off: bool = False,
) -> str:
    """The tier of one field.
    """
    if path.startswith("robot.safety.") or path in _SAFETY_ELSEWHERE:
        return "safety"
    if required or decided:
        return "site"
    if gated_off:
        return "advanced"
    return "tuned"


def gate_state(path: str, index: dict[str, Any], values: dict[str, Any] | None) -> bool:
    """Is ``path`` under a block whose ``enabled`` switch is off?

    Walks the ancestors and asks each for its ``enabled`` sibling.
    """
    parts = path.split(".")
    for i in range(1, len(parts)):
        gate = ".".join(parts[:i]) + ".enabled"
        if gate == path or gate not in index:
            continue
        if values is not None and gate in values:
            if values[gate] is False:
                return True
            continue
        if index[gate].default is False:
            return True
    return False
