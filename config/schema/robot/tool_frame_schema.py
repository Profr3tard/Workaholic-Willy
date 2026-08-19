"""The FLANGE -> TCP transform: where the grasp centre sits on the flange, and who owns that fact."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, model_validator

from .._base import StrictModel

__all__ = ["ToolFrameConfig", "ToolFrameSource"]

#: Who holds the flange->TCP transform on a real cell.
#:
#: ``undeclared`` -- nobody has said. The driver refuses to move the arm on real hardware.
#: ``willy``      -- the DRIVER composes flange<->TCP inside its own boundary
#:   and the controller runs a bare flange.
#: ``polyscope``  -- the CONTROLLER holds it (an operator set it on the teach pendant).
ToolFrameSource = Literal["undeclared", "willy", "polyscope"]


class ToolFrameConfig(StrictModel):
    """Flange -> TCP for the mounted end-effector.

    PER GRIPPER, NOT PER ROBOT. This block hangs off ``robot.gripper`` because the transform is a
    property of the end-effector: the repo already carries four of them (the 2F-85 at 132 mm, the
    Schunk EZU-35 at 140 mm, the EGU-50 at ~144 mm, and the real suction cup at 161 mm) and swaps
    between two of them mid-run in the tool-change demos.
    """

    #: Who owns the transform. See :data:`ToolFrameSource`. Default ``undeclared`` keeps every existing
    #: config loadable and byte-identical; it is the real-arm driver, not the schema, that refuses it.
    source: ToolFrameSource = "undeclared"
    offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    verify_tolerance_mm: float = Field(default=10.0, gt=0.0)

    @model_validator(mode="after")
    def _check_declared(self) -> "ToolFrameConfig":
        if self.source == "undeclared":
            return self  # nothing to be consistent with yet; the driver is what refuses

        q = self.rotation_quat_xyzw
        norm = math.sqrt(sum(v * v for v in q))
        if abs(norm - 1.0) > 1e-6:
            raise ValueError(
                f"gripper.tool_frame.rotation_quat_xyzw must be a unit quaternion; "
                f"got norm {norm:.9f} for {q}. An unnormalised quaternion silently scales every "
                f"transformed pose."
            )
        if all(v == 0.0 for v in self.offset_mm) and q == (0.0, 0.0, 0.0, 1.0):
            raise ValueError(
                f"gripper.tool_frame declares source: {self.source!r} but leaves the transform at "
                "identity (offset_mm [0,0,0] + rotation_quat_xyzw [0,0,0,1]). That is the shape of an "
                "unmeasured cell, not of a mounted tool: it says the grasp centre IS the flange face. "
                "Measure the coupling + gripper on the bench and state both halves, or set "
                "source: undeclared while the cell is still being built."
            )
        return self
