"""The FLANGE -> TCP transform: where the grasp centre sits on the flange, and who owns that fact.

Every ``Pose`` this stack commands is the **TCP**: the grasp centre, between the fingers. Where that
point sits relative to the robot's flange, and how it is turned, is a property of the MOUNTED TOOL.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, model_validator

from .._base import StrictModel

__all__ = ["ToolFrameConfig", "ToolFrameSource"]

#: Who holds the flange->TCP transform on a real cell.
#:
#: ``undeclared``: nobody has said. A real-arm driver REFUSES to connect: silence here is not a
#:   default, it is an unmeasured cell, and the failure it produces is a crash rather than an error.
#: ``willy``: the DRIVER composes flange<->TCP inside its own boundary and the controller runs a
#:   bare flange. This is exactly what the Isaac driver already does (``drivers/sim/arm.py``), it needs
#:   no vendor SDK call whatsoever, and it is the only mode that also fixes the cuRobo path, which
#:   plans to ``tool0`` and executes ``moveJ`` on joint waypoints, bypassing any controller-side tool
#:   register entirely (``drivers/ur/curobo_motion.py``).
#: ``polyscope``: the CONTROLLER holds it (an operator set it on the teach pendant). The driver
#:   passes poses through untouched and only VERIFIES the controller agrees.
#:
#: Both live modes are checked by the same measurement, and they predict different answers
#: (~identity for ``willy``, the declared transform for ``polyscope``), so the check also proves
#: which mode the cell is really in, not merely that some number matched.
ToolFrameSource = Literal["undeclared", "willy", "polyscope"]


class ToolFrameConfig(StrictModel):
    """Flange -> TCP for the mounted end-effector.

    **(3) NOT VALIDATED ON REAL HARDWARE.** Every number here must be MEASURED on the actual cell:
    the coupling plate PLUS the gripper, not the gripper alone. Stating it as config is what lets the
    path be built and tested before the hardware exists; the defaults are a "nothing measured yet"
    marker, not a claim about any cell.

    WHY IT EXISTS. ``GraspExecutionPolicy`` drives ``grasp.position`` verbatim, and the UR driver hands
    poses straight to ``moveL``, in whatever frame the controller's tool register holds. At the UR
    factory default (identity at the flange), a top-down grasp commanding z=37 mm drives the FLANGE
    to 37 mm and the fingertips to z=-95 mm, i.e. 95 mm below the table. Nothing in the stack catches
    it: the WorkspaceGuard bounds the COMMANDED number and the September cell ships ``z_min: 0.0``,
    and ``fixtures: []`` is the only fixture declaration in any profile, so there is no table to hit.

    THE ROTATIONAL HALF IS THE DANGEROUS HALF, because it is invisible. An offset wrong by 100 mm is a
    crash on run 1 and you fix it. A rotation wrong by 90 degrees is a cell that logs 10/10 successes
    while closing the jaws along the wrong object axis. Hand-eye calibration cannot catch it,
    because the calibration solves for whatever frame ``get_tcp_pose()`` reports and reports an
    excellent RMSE either way.

    PER GRIPPER, NOT PER ROBOT. This block hangs off ``robot.gripper`` because the transform is a
    property of the end-effector: the repo already carries four of them (the 2F-85 at 132 mm, the
    Schunk EZU-35 at 140 mm, the EGU-50 at ~144 mm, and the real suction cup at 161 mm) and swaps
    between two of them mid-run in the tool-change demos.
    """

    #: Who owns the transform. See :data:`ToolFrameSource`. Default ``undeclared`` keeps every existing
    #: config loadable and byte-identical; it is the real-arm driver, not the schema, that refuses it.
    source: ToolFrameSource = "undeclared"

    #: Flange -> grasp-centre translation, mm, in the FLANGE frame. Same origin, axes and bench step as
    #: ``safety.payload.cog_mm``; measure them together.
    offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)

    #: Flange -> TCP rotation as XYZW. Identity means the grasp frame is the flange frame, which is
    #: true of no real gripper: the Willy grasp frame is X=closing, Y=binormal, Z=approach, and a tool
    #: is bolted on in whatever orientation its coupling dictates.
    #:
    #: For reference, the value the Isaac driver hardcodes for the 2F-85 on a UR flange is
    #: ``(-0.70710678, 0.0, 0.0, 0.70710678)``, a -90 deg rotation about flange X, which maps the
    #: TCP's +Z (approach) onto flange +Y. It is NOT the default here: a default that happens to be
    #: right for one gripper is how the next cell inherits a wrong frame silently.
    rotation_quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    #: How far the controller's ACTIVE tool frame may differ from what this block declares before the
    #: driver refuses to connect, in mm.
    #:
    #: The active frame is derived, not read: ``inv(base->flange from the bundled DH table) @
    #: getForwardKinematics(q)``. Both halves already exist and are already on the safety-critical path,
    #: so this needs no new vendor SDK surface. MEASURED offline over 200 random configurations per
    #: model, the derivation is exact to 1e-13 mm on nominal kinematics; the tolerance exists for the
    #: per-unit kinematic calibration a physical arm carries, which spreads a correctly-configured arm
    #: by ~1.4 mm at a 0.5 mm/5e-4 rad perturbation and ~4.6 mm at a pessimistic 1.0 mm/1e-3 rad.
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
