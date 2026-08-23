"""
Pure-Python sim drivers :class:`DummyRobotArm` (and friends).

Used by tests and offline development. No external SDKs.
"""

from __future__ import annotations

from .arm import DUMMY_CAPABILITIES, DummyRobotArm

__all__ = ["DUMMY_CAPABILITIES", "DummyRobotArm"]
