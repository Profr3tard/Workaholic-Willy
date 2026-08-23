"""Exact mesh-vs-mesh self-collision backend for the ``backend='fcl'`` path of
:class:`SelfCollisionGuard`. It replaces the capsule proxy, measured wrong in BOTH directions
(false-negative ``forearm|wrist_3`` 25 mm SKIPPED by ``_should_skip_pair``; false-positive
``link_2|link_5`` from the fat default radius) — with exact distance on the real UR5e meshes.

Engine: **Coal preferred, python-fcl fallback** (resolved once, centrally, through
:func: `src.robot.safety.planning.environment.import_collision_engine`).

Design (all default-off byte-identical: this path is reached ONLY when ``self_collision.backend='fcl'``):
  * Per-link collision meshes are bundled (``data/ur5e_collision_meshes.npz``), pre-baked into the
    bundled-DH link frame: ``M_dh = inv(T_dh[frame](q0)) @ inv(R_base) @ world_usd(q0)`` (validated
    vertex-exact, <0.15 mm, against the Isaac USD at 6 configs). One BVH model is built per link ONCE;
    at each evaluate only the CollisionObject TRANSFORM is updated (no rebuild).
  * The world transform of link *L* at joints *q* is ``R_base(yaw) @ T_dh[L.frame](q)`` the same
    base-frame yaw reconcile (``kinematics_base_yaw_deg``) the capsule path uses, so meshes, fixtures
    and tool agree.
  * Pairwise self-collision skips only links within ONE DH frame of each other (adjacent / rigid wrist
    cluster, which always touch by construction); every farther pair is checked EXACTLY so the wrist
    pairs the capsule had to skip are now covered without the capsule's false positives.
  * Each link is also checked against the fixtures (axis-aligned boxes).

Run ``python -m src.robot.safety.planning --check`` to see which engine resolves on this box.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ._ur_kinematics import UR_DH_TABLES_M
from .planning.environment import (
    collision_mesh_bundle,
    import_collision_engine,
)


_LOGGER = logging.getLogger(__name__)

#: Mirrors ``self_collision._DEFAULT_LINK_RADIUS_MM`` quoted in the fallback warning so the operator sees
#: HOW much coarser the capsule guard is without having to go read the other module.
_CAPSULE_FALLBACK_RADIUS_HINT_MM = 60.0

_STATUS_HINTS = {
    "unknown_model": "No bundled DH chain for this model add it to safety/_ur_kinematics.UR_DH_TABLES_M.",
    "no_bundle": (
        "Bake the per-link collision meshes into safety/data/{model}_collision_meshes.npz "
        "(same DH-frame recipe as the ur5e bundle; the 2F-85 tool0 meshes are model-independent and can be "
        "copied verbatim). Until then this cell has NO exact-mesh self-collision authority."
    ),
    "no_engine": "Install Coal (WILLY_COAL_PREFIX) or python-fcl; expected/accepted on macOS + CI.",
}


def _yaw_matrix(yaw_deg: float) -> np.ndarray:
    """3x3 rotation about +Z by ``yaw_deg`` (the kinematics_base_yaw_deg reconcile)."""
    import math
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


class _EngineAdapter:
    """Uniform wrapper over Coal OR python-fcl. The two APIs differ only in a few names/types; the
    queries (BVH build + closest-distance) are behavior-identical (proven byte-identical)."""

    def __init__(self, mod: Any, kind: str) -> None:
        self._m = mod
        self.kind = kind

    def build_object(self, verts: np.ndarray, faces: np.ndarray) -> Any:
        m = self._m
        if self.kind == "coal":
            model = m.BVHModelOBBRSS()
            vv = m.StdVec_Vec3s()
            for v in np.asarray(verts, dtype=np.float64):
                vv.append(v)
            tt = m.StdVec_Triangle()
            for f in faces:
                tt.append(m.Triangle(int(f[0]), int(f[1]), int(f[2])))
            model.beginModel(len(tt), len(vv))
            model.addSubModel(vv, tt)
            model.endModel()
            return m.CollisionObject(model, m.Transform3s())
        model = m.BVHModel()
        model.beginModel(len(verts), len(faces))
        model.addSubModel(np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64))
        model.endModel()
        return m.CollisionObject(model, m.Transform())

    def _transform(self, R: np.ndarray, t: np.ndarray) -> Any:
        m = self._m
        R = np.ascontiguousarray(R, dtype=np.float64)
        t = np.ascontiguousarray(t, dtype=np.float64)
        return m.Transform3s(R, t) if self.kind == "coal" else m.Transform(R, t)

    def set_transform(self, obj: Any, R: np.ndarray, t: np.ndarray) -> None:
        obj.setTransform(self._transform(R, t))

    def box_object(self, half_extents: np.ndarray, center: np.ndarray) -> Any:
        m = self._m
        size = 2.0 * np.asarray(half_extents, dtype=np.float64)
        return m.CollisionObject(m.Box(*size), self._transform(np.eye(3), np.asarray(center, dtype=np.float64)))

    def distance(self, a: Any, b: Any) -> float:
        m = self._m
        return float(m.distance(a, b, m.DistanceRequest(), m.DistanceResult()))


class MeshSelfCollisionBackend:
    """Holds the per-link BVH models + runs the exact pairwise + fixture closest-distance checks."""

    def __init__(self, adapter: _EngineAdapter, meshes: dict[str, tuple[np.ndarray, np.ndarray, int]]) -> None:
        self._a = adapter
        self.engine = adapter.kind  # 'coal' | 'fcl' surfaced for telemetry / the migration proof
        self._models: dict[str, Any] = {}
        self._frame: dict[str, int] = {}
        # Per-link bounding sphere (centroid + radius, in the link frame) for the optional broadphase
        # cull. Conservative (the sphere bounds the mesh), so it NEVER changes a verdict.
        self._sph_c: dict[str, np.ndarray] = {}
        self._sph_r: dict[str, float] = {}
        for name, (verts, faces, frame) in meshes.items():
            self._models[name] = adapter.build_object(verts, faces)
            self._frame[name] = int(frame)
            v = np.asarray(verts, dtype=np.float64)
            c = v.mean(axis=0)
            self._sph_c[name] = c
            self._sph_r[name] = float(np.linalg.norm(v - c, axis=1).max())
        self._names = list(self._models)

    def evaluate(
        self,
        transforms_dh_mm: list[np.ndarray],
        yaw_deg: float,
        fixtures: tuple,
        min_distance_mm: float,
        broadphase: bool = False,
    ) -> tuple[str, float] | None:
        """Return ``(pair, signed_distance_mm)`` for the first violating pair, else ``None``.

        ``transforms_dh_mm`` is the full per-frame DH FK (``ur_link_transforms_mm``); ``fixtures`` are
        :class:`AxisAlignedBox` (center/half-extents mm) in the system base frame. ``broadphase`` (used by
        the continuous monitor) skips a pair via its bounding spheres BEFORE the exact query when they
        cannot be within ``min_distance_mm`` a CONSERVATIVE cull (spheres bound the meshes), so the
        verdict is byte-identical to the brute path (default ``False`` keeps the one-shot guard unchanged).
        """
        a = self._a
        Rb = _yaw_matrix(yaw_deg)
        wc: dict[str, np.ndarray] = {}  # world-frame sphere centroids (broadphase only)
        # place every link mesh in the SYSTEM base frame: R_base @ T_dh[frame]
        for name in self._names:
            T = transforms_dh_mm[self._frame[name]]
            R = Rb @ T[:3, :3]
            t = Rb @ T[:3, 3]
            a.set_transform(self._models[name], R, t)
            if broadphase:
                wc[name] = R @ self._sph_c[name] + t
        # ---- link vs link (skip frames within 1: adjacent joints + the rigid wrist/gripper cluster) ----
        for i in range(len(self._names)):
            ni = self._names[i]
            for j in range(i + 1, len(self._names)):
                nj = self._names[j]
                if abs(self._frame[ni] - self._frame[nj]) <= 1:
                    continue
                if broadphase and (float(np.linalg.norm(wc[ni] - wc[nj]))
                                   - self._sph_r[ni] - self._sph_r[nj] > min_distance_mm):
                    continue  # spheres too far apart to possibly violate -> skip the exact query
                d = a.distance(self._models[ni], self._models[nj])
                if d < min_distance_mm:
                    return (f"{ni}|{nj}", d)
        # ---- link vs fixture (axis-aligned boxes) ----
        for fx in fixtures:
            fc = np.asarray(fx.center_mm, dtype=np.float64)
            fr = float(np.linalg.norm(np.asarray(fx.half_extents_mm, dtype=np.float64)))
            box = a.box_object(np.asarray(fx.half_extents_mm, dtype=np.float64), fc)
            for name in self._names:
                if broadphase and (float(np.linalg.norm(wc[name] - fc))
                                   - self._sph_r[name] - fr > min_distance_mm):
                    continue
                d = a.distance(self._models[name], box)
                if d < min_distance_mm:
                    fname = getattr(fx, "name", "") or "fixture"
                    return (f"{name}|fixture:{fname}", d)
        return None


def mesh_backend_status(
    model: str, mesh_dir: str | None = None, mesh_name: str | None = None
) -> str:
    """Why the exact-mesh backend can (or cannot) run for ``model`` one stable, loggable token.

    Returns ``"ok"`` | ``"unknown_model"`` (no bundled DH chain, so link meshes cannot be placed) |
    ``"no_bundle"`` (no ``{model}_collision_meshes.npz``) | ``"no_engine"`` (neither Coal nor python-fcl
    importable the accepted macOS/CI condition).
    """
    if model.lower() not in UR_DH_TABLES_M:
        return "unknown_model"
    default = collision_mesh_bundle(model, mesh_name)
    path = (Path(mesh_dir) / default.name) if mesh_dir else default
    if not path.exists():
        return "no_bundle"
    mod, kind = import_collision_engine()
    if mod is None or kind is None:
        return "no_engine"
    return "ok"


def make_backend(
    model: str, mesh_dir: str | None = None, mesh_name: str | None = None
) -> MeshSelfCollisionBackend | None:
    """Build the mesh backend (Coal preferred, python-fcl fallback) for ``model``, or ``None``."""
    status = mesh_backend_status(model, mesh_dir, mesh_name)
    if status != "ok":
        _LOGGER.warning(
            "exact-mesh self-collision UNAVAILABLE for model %r (%s): falling back to the CAPSULE guard, "
            "which is coarser and over-rejects (default %.0f mm link radius). %s",
            model, status, _CAPSULE_FALLBACK_RADIUS_HINT_MM,
            _STATUS_HINTS.get(status, ""),
        )
        return None
    mod, kind = import_collision_engine()
    if mod is None or kind is None:  # narrowing only: status == "ok" already proved the engine imports
        return None
    default = collision_mesh_bundle(model, mesh_name)
    fname = default.name
    path = (Path(mesh_dir) / fname) if mesh_dir else default
    data = np.load(path)
    names = sorted({k.split("__")[0] for k in data.files})
    meshes: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for n in names:
        meshes[n] = (data[f"{n}__v"], data[f"{n}__f"], int(data[f"{n}__frame"][0]))
    try:
        return MeshSelfCollisionBackend(_EngineAdapter(mod, kind), meshes)
    except Exception:  # noqa: BLE001 - any engine construction failure -> capsule fallback
        return None
