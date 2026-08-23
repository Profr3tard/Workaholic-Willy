"""Universal Robots model registry for the Isaac sim driver (multi-robot de-lock).

The **canonical model key** ("ur5e", "ur3e", ...) is the single string shared across the whole stack:
  * the safety self-collision DH table (:mod:`src.robot.safety._ur_kinematics`, ``UR_DH_TABLES_M``),
  * the cuRobo robot config filename (``{key}.yml``, built on-box by ``docs/curobo/build_{key}_config.py``),
  * the safety config's ``kinematics_model``.
This registry adds the two ISAAC-SPECIFIC bits that key can't express: the Lula supported-config name
(capitalised, e.g. "UR3e") and the Isaac asset-pack USD relpath (the combined arm+gripper asset).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "URModelSpec",
    "curobo_robot_yml",
    "ur_model_spec",
]


@dataclass(frozen=True, slots=True)
class URModelSpec:
    """The Isaac / Lula model spec for one UR variant."""

    key: str
    lula_key: str
    usd_relpath: str
    #: The ``Gripper`` USD variant this model's asset BAKES IN, or ``None`` when the asset ships bare.
    baked_gripper_variant: str | None = None
    #: Vendor-datasheet REACH (mm), the radius of the working sphere about the shoulder and the payload
    #: cap (kg).
    max_reach_mm: float = 850.0
    max_payload_kg: float = 5.0
    #: The link the eye-in-hand camera and the tool hang from, RELATIVE to the robot root prim.
    wrist_link_name: str = "wrist_3_link"
    workspace_center_mm: tuple[float, float] = (450.0, 0.0)
    workspace_half_extents_mm: tuple[float, float] = (150.0, 150.0)


# The UR e-series models Isaac ships a USD for (verified on-box: the ur3e/ur5e asset dirs exist under
# ``.../Isaac/Robots/UniversalRobots/``). Each key also has a bundled DH row in ``_ur_kinematics`` and a
# buildable ``{key}.yml`` cuRobo config. Extend as more UR variants are validated.
_UR_MODELS: dict[str, URModelSpec] = {
    # ur3e ships BARE (no baked gripper) -> a ur3e cell must set gripper_mount (e.g. "robotiq_2f85").
    "ur3e": URModelSpec("ur3e", "UR3e", "/Isaac/Robots/UniversalRobots/ur3e/ur3e.usd",
                        max_reach_mm=500.0, max_payload_kg=3.0,
                        workspace_center_mm=(300.0, 0.0),
                        workspace_half_extents_mm=(100.0, 100.0)),
    "ur5e": URModelSpec("ur5e", "UR5e", "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd",
                        baked_gripper_variant="Robotiq_2f_85", max_reach_mm=850.0, max_payload_kg=5.0),
    "ur10e": URModelSpec("ur10e", "UR10e", "/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd",
                         baked_gripper_variant="Robotiq_2f_85", max_reach_mm=1300.0, max_payload_kg=12.5),
}


def ur_model_spec(model: str) -> URModelSpec:
    """The :class:`URModelSpec` for ``model`` (case-insensitive). Raises ``ValueError`` for an unknown key."""
    spec = _UR_MODELS.get(model.lower())
    if spec is None:
        raise ValueError(
            f"unsupported sim robot_model {model!r}; known: {sorted(_UR_MODELS)}. Add a URModelSpec "
            "(Lula key + Isaac USD relpath) and build the matching cuRobo {key}.yml to support it."
        )
    return spec


def curobo_robot_yml(model: str) -> str:
    """The cuRobo robot-config filename for ``model`` (built on-box by ``docs/curobo/build_{key}_config.py``)."""
    return f"{ur_model_spec(model).key}.yml"
