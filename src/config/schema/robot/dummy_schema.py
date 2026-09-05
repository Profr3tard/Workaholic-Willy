"""Dummy driver config schema."""

from __future__ import annotations



from .._base import StrictModel


class DummyConfig(StrictModel):
    """Settings for the in-process dummy driver.

    It carries no fields: the dummy driver is featureless by intent, so
    that the rest of the stack can be exercised with neither hardware nor
    a simulator. It is a class rather than ``None`` so a field can be
    added later without changing the shape of the YAML.
    """

    pass
