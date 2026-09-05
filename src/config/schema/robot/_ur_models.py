"""The UR models Willy can drive: one list, shared by every block that names a model.

The canonical lower-case key ("ur5e", "ur3e", ...) is the single string the whole stack keys off:
the safety DH table (``safety/_ur_kinematics.py``), the exact-mesh bundle
(``{key}_collision_meshes.npz``), the cuRobo robot config (``{key}.yml``), the Isaac USD and Lula
config, and ``safety.self_collision.kinematics_model``. The simulated cell and the real UR cell
each name a model and must agree on which names exist, so the set lives here rather than inside
one vendor's schema. A UR3e whose config says "ur5e" resolves another robot's link lengths in
every one of those lookups, without a word, on real hardware.

The list sits in the config layer, the bottom of the dependency stack, so it must not import the
driver package. Keep it in lockstep with ``src.robot.drivers.sim.robot_models._UR_MODELS``.
"""

from __future__ import annotations

__all__ = ["UR_MODEL_KEYS"]

UR_MODEL_KEYS: tuple[str, ...] = ("ur3e", "ur5e", "ur10e")
