"""Teaching the cuRobo planner the clearance the SAFETY GUARD will demand.

Two engines decide whether an arm configuration is acceptable, and they do not use the same geometry:

  * cuRobo plans against a SPHERE model of the links and returns the first path it believes is
    collision-free;
  * ``SelfCollisionGuard`` then re-checks that path's final configuration against the EXACT link MESHES
    and rejects anything closer than ``safety.self_collision.min_distance_mm``.

HONEST LIMIT: this reduces the disagreement, it does not eliminate it. Spheres are not meshes, so the
two models will still differ locally, the buffer is a cushion, not a proof. It is the guard, never the
planner, that decides what actually executes.
"""

from __future__ import annotations

import copy
from typing import Any

import yaml

__all__ = [
    "ENV_SELF_COLLISION_MARGIN_MM",
    "apply_self_collision_margin",
    "derive_margin_config_file",
]

#: Env var the client uses to hand the guard's margin to the sidecar. Unset / ``0`` -> untouched config.
ENV_SELF_COLLISION_MARGIN_MM = "WILLY_CUROBO_SELF_COLLISION_MARGIN_MM"

_BUFFER_PATH = ("robot_cfg", "kinematics", "self_collision_buffer")


def apply_self_collision_margin(config: dict[str, Any], margin_mm: float) -> tuple[dict[str, Any], int]:
    """Return ``(config_copy, links_adjusted)`` with each self-collision buffer raised by half ``margin_mm``.

    Half per link, because cuRobo subtracts BOTH links' buffers from a pair's sphere distance so half
    each yields exactly ``margin_mm`` of demanded pairwise clearance.
    """
    out = copy.deepcopy(config)
    if margin_mm <= 0.0:
        return out, 0
    node: Any = out
    for key in _BUFFER_PATH[:-1]:
        node = node.get(key) if isinstance(node, dict) else None
        if node is None:
            return out, 0
    buffers = node.get(_BUFFER_PATH[-1]) if isinstance(node, dict) else None
    if not isinstance(buffers, dict) or not buffers:
        return out, 0

    half_m = float(margin_mm) / 2000.0  # mm -> m, split across the pair
    adjusted = 0
    for link, value in list(buffers.items()):
        if isinstance(value, (int, float)):
            buffers[link] = float(value) + half_m
            adjusted += 1
    return out, adjusted


def derive_margin_config_file(source_path: str, dest_path: str, margin_mm: float) -> int:
    """Write ``source_path``'s robot config to ``dest_path`` with the margin applied. Returns links adjusted.

    Kept separate from :func:`apply_self_collision_margin` so the transform stays a pure function; only
    this wrapper touches the filesystem.
    """
    with open(source_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    adjusted_config, adjusted = apply_self_collision_margin(config, margin_mm)
    with open(dest_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(adjusted_config, handle, sort_keys=True)
    return adjusted
