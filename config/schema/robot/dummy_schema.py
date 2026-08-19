"""Dummy driver config schema."""

from __future__ import annotations



from .._base import StrictModel


class DummyConfig(StrictModel):
    """Settings for the in-process dummy driver.

    No knobs today the dummy driver is intentionally featureless
    and exists so the rest of the stack can be exercised without any
    hardware or simulator.
    """

    pass
