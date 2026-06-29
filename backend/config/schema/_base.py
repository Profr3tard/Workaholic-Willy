"""Shared base classes and validators for every config schema.

All configuration models inherit from :class:`StrictModel`, which enforces
two project-wide invariants:

* ``extra="forbid"`` — typos in YAML files are rejected at load time
  instead of being silently ignored.
* ``frozen=True`` — config objects are immutable after construction.
  This prevents accidental cross-component mutation; configs are
  constructed once at startup and treated as values, not state.

Use :class:`StrictModel` for every new config class.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base class for every configuration schema in this package.

    See module docstring for the rationale behind ``extra="forbid"`` and
    ``frozen=True``. Subclasses can opt out individually by overriding
    ``model_config``, but should justify the decision in a comment.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )
