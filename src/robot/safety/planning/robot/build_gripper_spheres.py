"""Generate the Robotiq 2F-85 gripper collision-sphere map for cuRobo, Isaac-free, repo-derived.

The reference robot's collision geometry is the vertex-exact, version-controlled mesh authority
``src/robot/safety/data/ur5e_collision_meshes.npz`` (the SAME source that feeds the Coal
self-collision guard AND the on-box cuRobo sphere fit). This script grid-fits the gripper
(``gripper`` body + ``lfinger`` + ``rfinger``) — all in the tool0 frame, into a cuRobo-format
collision-sphere map, so the gripper "exists as a labeled file" in the repo without any Isaac / GPU
dependency.

It writes ``ur5e_gripper_spheres.yml`` next to this script. Run it with the project venv:

    .venv/Scripts/python.exe src/robot/safety/planning/robot/build_gripper_spheres.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

_MESH_NPZ = (
    Path(__file__).resolve().parents[3] / "safety" / "data" / "ur5e_collision_meshes.npz"
)
_OUT = Path(__file__).with_name("ur5e_gripper_spheres.yml")


def _grid_fit_spheres(verts_mm: np.ndarray, cell_mm: float, rmax_mm: float) -> list[dict]:
    """One tight sphere per occupied voxel (centre = voxel centroid, r = max vert dist, capped).

    The union covers the real mesh with minimal over-reach. Returns spheres in METRES, tool0 frame.
    Mirrors ``docs/curobo/build_ur5e_config.py`` so the committed gripper spheres match the on-box fit.
    """
    v = np.asarray(verts_mm, dtype=np.float64)
    keys = np.floor(v / cell_mm).astype(np.int64)
    out: list[dict] = []
    for key in sorted({tuple(k) for k in keys}):
        pts = v[np.all(keys == np.asarray(key), axis=1)]
        c = pts.mean(0)
        r = min(float(np.max(np.linalg.norm(pts - c, axis=1))), rmax_mm)
        out.append(
            {
                "center": [round(float(x) / 1000.0, 4) for x in c],
                "radius": round(max(r, 6.0) / 1000.0, 4),
            }
        )
    return out


def build() -> dict:
    mesh = np.load(_MESH_NPZ, allow_pickle=True)
    tool0 = (
        _grid_fit_spheres(mesh["gripper__v"], cell_mm=44.0, rmax_mm=24.0)
        + _grid_fit_spheres(mesh["lfinger__v"], cell_mm=34.0, rmax_mm=17.0)
        + _grid_fit_spheres(mesh["rfinger__v"], cell_mm=34.0, rmax_mm=17.0)
    )
    return {
        "_provenance": {
            "robot": "Universal Robots UR5e",
            "gripper": "Robotiq 2F-85",
            "frame": "tool0 (Y=approach, X=closing, Z=depth), metres",
            "source": "src/robot/safety/data/ur5e_collision_meshes.npz (vertex-exact)",
            "generated_by": "src/robot/safety/planning/robot/build_gripper_spheres.py",
            "note": (
                "tool0 gripper spheres only (frame-correct, directly cuRobo-usable). Arm-link "
                "spheres + the Lula-tuned complete ur5e.yml are on-box via "
                "docs/curobo/build_ur5e_config.py. See PROVENANCE.md."
            ),
        },
        "collision_spheres": {"tool0": tool0},
    }


def main() -> None:
    cfg = build()
    _OUT.write_text(yaml.safe_dump(cfg, default_flow_style=False, sort_keys=False), encoding="utf-8")
    print(f"wrote {_OUT}  ({len(cfg['collision_spheres']['tool0'])} tool0 spheres from {_MESH_NPZ.name})")


if __name__ == "__main__":
    main()
