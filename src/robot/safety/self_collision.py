"""
SelfCollisionGuard: link/tool/fixture self-collision guard.

Two backends are supported:

* ``capsule`` (default, always available) the arm is approximated as
  a chain of capsules (one per link) plus a tool capsule rooted at the
  TCP and a base column capsule. Fixtures are axis-aligned boxes
  declared in :class:`FixtureBoxConfig`. Distance is closed-form so the
  per-move cost stays sub-millisecond.

  - For UR robots the arm-vs-arm capsule chain is derived from a
    bundled DH table (see :mod:`_ur_kinematics`). Non-adjacent link
    pairs are checked against each other and against every fixture.
  - For non-UR vendors (KUKA / SIM) the chain is not available; the
    guard checks only **tool vs base** and **tool vs fixtures**.
    Arm-vs-arm coverage requires a DH / URDF entry not bundled for KUKA.

* ``fcl`` exact mesh-vs-mesh distance on the bundled per-model
  collision meshes.

Configuration
-------------
* ``min_distance_mm`` minimum allowed signed distance between any
  monitored shape pair; under this is "collision".
* ``link_radii_mm`` per-link capsule radius. When omitted the guard
  uses a 60 mm default (typical UR link cross-section).
* ``fixtures`` list of :class:`FixtureBoxConfig`.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from ._capsule import (
    AxisAlignedBox,
    Capsule,
    capsule_box_distance_mm,
    capsule_capsule_distance_mm,
)
from ._ur_kinematics import ur_link_origins_mm, ur_link_transforms_mm
from .decision import SafetyDecision, SafetyReason
from .guard import SafetyContext

if TYPE_CHECKING:  # pragma: no cover - typing only
    from config.schema.robot import SelfCollisionSafetyConfig

__all__ = ["SelfCollisionGuard"]


_LOGGER = logging.getLogger(__name__)

# Default capsule radii / lengths used when the operator does not
# override them. These match a typical 60 mm-cross-section UR link and
# a 70 mm gripper envelope.
_DEFAULT_LINK_RADIUS_MM = 60.0
_DEFAULT_TOOL_RADIUS_MM = 70.0
_DEFAULT_TOOL_LENGTH_MM = 150.0
_DEFAULT_BASE_RADIUS_MM = 80.0
_DEFAULT_BASE_HEIGHT_MM = 150.0


class SelfCollisionGuard:
    """Self-collision guard.

    Stateless across calls. Constructed by
    :meth:`SafetyPreflight.from_safety_config` when
    ``safety.self_collision.enforce`` is ``True``.
    """

    name = "self_collision"

    def __init__(self, config: "SelfCollisionSafetyConfig") -> None:
        self._config = config
        self._min_distance_mm = float(config.min_distance_mm)
        # What the PLANNER should keep clear so it stops proposing configs this guard will
        # reject. Separate from _min_distance_mm on purpose: see SelfCollisionConfig.
        self._planner_margin_mm = float(getattr(config, 'planner_margin_mm', 0.0) or 0.0)
        self._backend = config.backend
        # Lazily-built exact-mesh backend (only for backend='fcl'); None == not yet built or unavailable.
        self._fcl_backend: object | None = None
        self._fcl_backend_built = False
        #: Why the exact-mesh backend is (un)available once built "ok" | "unknown_model" | "no_bundle" |
        #: "no_engine" | None (not built yet). Surfaced so a DEGRADED cell is inspectable, not silent.
        self._fcl_status: str | None = None
        # Tool/base capsule geometry (config-derived; defaults == the module constants).
        self._tool_length_mm = float(getattr(config, "tool_length_mm", _DEFAULT_TOOL_LENGTH_MM))
        self._tool_radius_mm = float(getattr(config, "tool_radius_mm", _DEFAULT_TOOL_RADIUS_MM))
        # Tool model. "capsule" (default) = approach-axis bounding cylinder; "finger" =
        # a thin capsule along the grasp closing axis (the faithful 2F-85 descending-finger footprint).
        self._tool_model = str(getattr(config, "tool_model", "capsule"))
        self._tool_finger_radius_mm = float(getattr(config, "tool_finger_radius_mm", 16.0))
        self._tool_finger_span_mm = float(getattr(config, "tool_finger_span_mm", 150.0))
        self._base_radius_mm = float(getattr(config, "base_radius_mm", _DEFAULT_BASE_RADIUS_MM))
        self._base_height_mm = float(getattr(config, "base_height_mm", _DEFAULT_BASE_HEIGHT_MM))
        self._fixtures = tuple(
            AxisAlignedBox(
                center_mm=np.asarray(fx.center_mm, dtype=np.float64),
                half_extents_mm=np.asarray(fx.half_extents_mm, dtype=np.float64),
            )
            for fx in config.fixtures
        )
        self._fixture_names = tuple(fx.name for fx in config.fixtures)

    # ------------------------------------------------------------------
    # Capsule construction
    # ------------------------------------------------------------------

    def _link_radius(self, axis_idx: int) -> float:
        radii = self._config.link_radii_mm
        if radii is None:
            return _DEFAULT_LINK_RADIUS_MM
        if axis_idx < len(radii):
            return float(radii[axis_idx])
        return _DEFAULT_LINK_RADIUS_MM

    def _base_capsule(self) -> Capsule:
        """Fixed base column reaching from floor to mounting flange."""
        return Capsule(
            p0=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            p1=np.array([0.0, 0.0, self._base_height_mm], dtype=np.float64),
            radius_mm=self._base_radius_mm,
        )

    def _tool_capsule(self, ctx: SafetyContext) -> Capsule | None:
        """Tool capsule rooted at the commanded TCP pose.

        ``tool_model='capsule'`` (default) extends a capsule from the TCP along the
        TCP's local +Z (approach) axis by ``tool_length_mm`` a rotation-invariant
        bounding cylinder for a tool protruding from the flange. ``tool_model='finger'``
        instead lays a thin capsule along the grasp's CLOSING axis (the faithful
        parallel-jaw footprint).
        """
        pose = ctx.target_pose
        if pose is None:
            return None
        tcp_mm = np.asarray(pose.position_mm, dtype=np.float64)
        from src.geometry.quaternion import to_rotation_matrix
        R = to_rotation_matrix(pose.quaternion_xyzw)
        if self._tool_model == "finger":
            closing = R[:, 0]
            half = 0.5 * self._tool_finger_span_mm
            return Capsule(
                p0=tcp_mm - closing * half,
                p1=tcp_mm + closing * half,
                radius_mm=self._tool_finger_radius_mm,
            )
        tool_z = R[:, 2]
        flange_side_mm = tcp_mm - tool_z * self._tool_length_mm
        return Capsule(p0=tcp_mm, p1=flange_side_mm, radius_mm=self._tool_radius_mm)

    def _ur_arm_capsules(self, ctx: SafetyContext) -> list[Capsule] | None:
        """Per-link capsules for a UR(-kinematics) arm, or ``None`` if not derivable.

        An explicit ``self_collision.kinematics_model`` selects the bundled DH table and OVERRIDES
        the vendor gate, this lets a non-UR vendor that is physically a UR arm (the Isaac sim, a UR5e)
        opt into real arm-vs-arm capsules. Without it, only a real ``vendor == "ur"`` arm qualifies,
        keyed by its own model.
        """
        if ctx.target_joints is None or ctx.arm is None:
            return None
        model = self._config.kinematics_model
        if model is None:
            if ctx.arm.capabilities.vendor != "ur":
                return None
            model = ctx.arm.capabilities.model
        origins_mm = ur_link_origins_mm(
            model, ctx.target_joints.values,
        )
        if origins_mm is None or len(origins_mm) < 2:
            return None
        yaw_deg = float(self._config.kinematics_base_yaw_deg)
        if yaw_deg != 0.0:
            origins_mm = self._rotate_origins_z(origins_mm, yaw_deg)
        capsules: list[Capsule] = []
        for i in range(len(origins_mm) - 1):
            capsules.append(
                Capsule(
                    p0=origins_mm[i],
                    p1=origins_mm[i + 1],
                    radius_mm=self._link_radius(i),
                )
            )
        return capsules

    @staticmethod
    def _rotate_origins_z(origins_mm: list[np.ndarray], yaw_deg: float) -> list[np.ndarray]:
        """Rotate base-frame origins about +Z by ``yaw_deg`` (base-frame reconcile).

        A pure base-frame rotation: it preserves every inter-link distance (so arm-vs-arm checks are
        unchanged) and only relocates the chain relative to the base/tool/fixture capsules.
        """
        yaw = math.radians(yaw_deg)
        c, s = math.cos(yaw), math.sin(yaw)
        return [
            np.array([c * o[0] - s * o[1], s * o[0] + c * o[1], o[2]], dtype=np.float64)
            for o in origins_mm
        ]

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------

    def evaluate(self, ctx: SafetyContext) -> SafetyDecision:
        if self._backend == "fcl":
            # Exact mesh-vs-mesh self-collision over the real UR5e collision meshes. Returns a
            # decision when it runs; ``None`` when python-fcl or the mesh bundle is absent / the model is
            # not derivable -> fall through to the capsule path (optional-dependency clean fallback).
            dec = self._evaluate_fcl(ctx)
            if dec is not None:
                return dec

        base = self._base_capsule()
        tool = self._tool_capsule(ctx)
        arm_links = self._ur_arm_capsules(ctx)

        # Build the full capsule registry for pairwise checks. Track
        # names so a rejection report can say *which* pair collided.
        labelled: list[tuple[str, Capsule]] = [("base", base)]
        if arm_links is not None:
            for i, cap in enumerate(arm_links):
                labelled.append((f"link_{i}", cap))
        if tool is not None:
            labelled.append(("tool", tool))

        # ---- capsule vs capsule (skip adjacent arm links) -----------
        for i in range(len(labelled)):
            name_i, cap_i = labelled[i]
            for j in range(i + 1, len(labelled)):
                name_j, cap_j = labelled[j]
                if self._should_skip_pair(name_i, name_j):
                    continue
                d = capsule_capsule_distance_mm(cap_i, cap_j)
                if d < self._min_distance_mm:
                    return SafetyDecision.reject(
                        self.name,
                        SafetyReason.SELF_COLLISION,
                        message=(
                            f"{name_i} vs {name_j}: signed distance "
                            f"{d:.3f} mm < {self._min_distance_mm:.3f} mm"
                        ),
                        detail={
                            "pair": f"{name_i}|{name_j}",
                            "signed_distance_mm": f"{d:.6f}",
                            "min_distance_mm": (
                                f"{self._min_distance_mm:.6f}"
                            ),
                        },
                    )

        # ---- capsule vs fixture --------------------------------------
        for k, fixture in enumerate(self._fixtures):
            fname = self._fixture_names[k] or f"fixture_{k}"
            for name_i, cap_i in labelled:
                d = capsule_box_distance_mm(cap_i, fixture)
                if d < self._min_distance_mm:
                    return SafetyDecision.reject(
                        self.name,
                        SafetyReason.SELF_COLLISION,
                        message=(
                            f"{name_i} vs fixture {fname!r}: signed "
                            f"distance {d:.3f} mm < "
                            f"{self._min_distance_mm:.3f} mm"
                        ),
                        detail={
                            "pair": f"{name_i}|fixture:{fname}",
                            "signed_distance_mm": f"{d:.6f}",
                            "min_distance_mm": (
                                f"{self._min_distance_mm:.6f}"
                            ),
                        },
                    )

        return SafetyDecision.accept(self.name)

    def _evaluate_fcl(self, ctx: SafetyContext) -> SafetyDecision | None:
        """Exact mesh self-collision. ``None`` if it cannot run (-> capsule fallback)."""
        if ctx.target_joints is None or ctx.arm is None:
            return None
        model = self._config.kinematics_model
        if model is None:
            if ctx.arm.capabilities.vendor != "ur":
                return None
            model = ctx.arm.capabilities.model
        if not self._fcl_backend_built:  # build once (BVH models), cache the result (incl. None)
            from ._fcl_self_collision import make_backend, mesh_backend_status
            variant = getattr(self._config, "collision_mesh_variant", None)
            self._fcl_backend = make_backend(model, self._config.mesh_dir, variant)
            self._fcl_backend_built = True
            # The config ASKED for the exact-mesh backend. If it could not be built we silently ran the
            # coarser capsule proxy which over-rejects (see _DEFAULT_LINK_RADIUS_MM) and would look like
            # "the robot cannot grasp anything" rather than "the safety authority is missing".
            self._fcl_status = mesh_backend_status(model, self._config.mesh_dir, variant)
            if self._fcl_backend is None:
                _LOGGER.warning(
                    "SelfCollisionGuard: backend='fcl' requested for model %r but the exact-mesh backend is "
                    "unavailable (%s) running the CAPSULE fallback for this cell.", model, self._fcl_status,
                )
        backend = self._fcl_backend
        if backend is None:
            return None  # python-fcl / mesh bundle absent -> capsule fallback
        transforms = ur_link_transforms_mm(model, ctx.target_joints.values)
        if transforms is None:
            return None
        hit = backend.evaluate(  # type: ignore[attr-defined]
            transforms, float(self._config.kinematics_base_yaw_deg), self._fixtures, self._min_distance_mm
        )
        if hit is None:
            return SafetyDecision.accept(self.name)
        pair, dmm = hit
        engine = getattr(backend, "engine", "fcl")  # 'coal' (preferred) | 'fcl' (fallback)
        return SafetyDecision.reject(
            self.name,
            SafetyReason.SELF_COLLISION,
            message=f"{pair}: mesh distance {dmm:.3f} mm < {self._min_distance_mm:.3f} mm",
            detail={
                "pair": pair,
                "signed_distance_mm": f"{dmm:.6f}",
                "min_distance_mm": f"{self._min_distance_mm:.6f}",
                "backend": engine,
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _should_skip_pair(name_a: str, name_b: str) -> bool:
        """Skip pairs that always overlap by construction.

        Adjacent links share an end-point. Non-adjacent wrist links
        (``link_3``/``link_4``/``link_5`` on a UR-style spherical
        wrist) are physically too close for a capsule approximation
        with realistic radii, we exclude any pair within two indices
        of each other to avoid spurious "wrist collides with wrist"
        rejections. Real collisions in this region would require an
        FCL/mesh backend.
        """
        if name_a.startswith("link_") and name_b.startswith("link_"):
            ia = int(name_a.split("_", 1)[1])
            ib = int(name_b.split("_", 1)[1])
            if abs(ia - ib) <= 2:
                return True
        if {name_a, name_b} == {"base", "link_0"}:
            return True
        # link_1 starts at the shoulder joint, which sits on top of
        # the base column by construction; the start-point of the
        # capsule always lies within the base capsule's envelope. The
        # link_1 capsule cannot swing into the base (it pivots
        # *about* the base axis), so skip this pair too. Real upper-
        # arm collisions are caught by ``link_2`` (upper arm shaft).
        if {name_a, name_b} == {"base", "link_1"}:
            return True
        # The wrist-mounted tool inevitably overlaps the last link.
        if {name_a, name_b} == {"tool", "link_5"}:
            return True
        return False
