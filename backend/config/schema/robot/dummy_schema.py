"""Dummy driver config schema for robot."""

from __future__ import annotations



from .._base import StrictModel


class DummyConfig(StrictModel):
    """Settings for the in-process dummy driver.

    No knobs today, the dummy driver is intentionally featureless
    and exists so the rest of the stack can be exercised without any
    hardware or simulator. Declared as a class (rather than ``None``)
    so future fields can be added without breaking the YAML shape.
    """

    pass