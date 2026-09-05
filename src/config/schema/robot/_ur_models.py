"""The UR models Willy can drive: one list, shared by every vendor block that names one.

The canonical lower-case model key ("ur5e", "ur3e", ...) is the single string the whole stack keys
off: the safety DH table (``safety/_ur_kinematics.py``), the exact-mesh bundle
(``{key}_collision_meshes.npz``), the cuRobo robot config (``{key}.yml``), the Isaac USD + Lula config,
and ``safety.self_collision.kinematics_model``.

It lives here rather than in one vendor's schema because both the simulated cell and the real UR cell
have to name a model, and they must agree about which names exist. A UR3e whose config says "ur5e"
resolves another robot's link lengths in every one of the lookups above, silently. On real hardware
that is not a cosmetic error.

Kept in the config layer (the bottom of the dependency stack, so it must not import the driver
package) and held in lockstep with the driver-side registry
``src.robot.drivers.sim.robot_models._UR_MODELS``.
"""

from __future__ import annotations

__all__ = ["UR_MODEL_KEYS"]

UR_MODEL_KEYS: tuple[str, ...] = ("ur3e", "ur5e", "ur10e")
