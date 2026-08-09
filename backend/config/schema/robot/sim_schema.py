"""Isaac-Sim driver + scene config schema."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from .._base import StrictModel

from ._ur_models import UR_MODEL_KEYS  # re-exported: the sim cell and the real UR cell share one list


class SimCameraSchema(StrictModel):
    """Per-camera entry inside :class:`SimConfig.cameras`.

    Pydantic mirror of the driver-side dataclass
    ``backend.src.robot.drivers.sim.config.SimCameraConfig`` - the
    runtime converts this Pydantic model into the driver dataclass at
    arm-construction time. Kept intentionally minimal: any future
    knobs SHOULD land on the driver dataclass first, then mirror here.
    """

    prim_path: str = Field(default="", min_length=0)
    mounting_mode: Literal["eye_to_hand", "eye_in_hand"] = "eye_to_hand"
    render_product_name: str | None = None
    # --- Isaac authoring extras (read by backend.src.willy_sim, not the bare driver) ---
    near_clip_m: float | None = Field(default=None, gt=0.0)
    resolution: tuple[int, int] | None = None
    # Horizontal field of view (deg) to author the LENS from -- i.e. the FOV of the real sensor this
    # camera stands in for (an Intel RealSense D435's RGB stream is 69.4 deg). None keeps Isaac's default
    # lens, which is NARROWER than a D435 and therefore frames a smaller slice of the table than the real
    # cell will: a silent sim2real gap. See willy_sim/scene/cameras.py (D435_RGB_HFOV_DEG).
    hfov_deg: float | None = Field(default=None, gt=0.0, lt=180.0)
    # Fixed (eye-to-hand) cameras: world position in mm (None -> authored elsewhere).
    position_mm: tuple[float, float, float] | None = None
    # Eye-in-hand cameras: mount in the wrist-link LOCAL frame (mm) + look-at aim + up hint.
    mount_offset_mm: tuple[float, float, float] | None = None
    mount_aim_mm: tuple[float, float, float] | None = None
    mount_up_hint: tuple[float, float, float] | None = None


class SimObjectConfig(StrictModel):
    """Graspable object authored into the sim scene (willy_sim scene-authoring)."""

    name: str = "cube"  # identity handle for prompt-based selection among clutter ("the green cube")
    # Procedural primitive shape (ignored when usd_asset_path is set). "cube" -> DynamicCuboid (size_mm = the
    # box extents). "cylinder" -> DynamicCylinder (size_mm = (diameter, diameter, height); the round body a
    # 3-finger centric gripper like the Schunk EZU-35 wraps firmly). Default "cube" -> byte-identical.
    shape: str = "cube"
    size_mm: tuple[float, float, float] = (30.0, 30.0, 50.0)
    position_mm: tuple[float, float, float] = (450.0, 0.0, 25.0)
    mass_kg: float = Field(default=0.05, gt=0.0)
    static_friction: float = Field(default=1.5, ge=0.0)
    dynamic_friction: float = Field(default=1.3, ge=0.0)
    color: tuple[float, float, float] = (0.9, 0.2, 0.1)  # RGB 0-1; the VL colour disambiguation handle
    usd_asset_path: str | None = None  # relative /Isaac/... USD asset (YCB mesh); None -> DynamicCuboid
    # Spawn orientation (world, WXYZ); None -> identity. A referenced YCB authored lying (e.g. the soup
    # can) is spawned UPRIGHT so it settles stable + reliably graspable. Ignored for procedural cubes.
    orientation_wxyz: tuple[float, float, float, float] | None = None
    # Collision approximation for a referenced NON-physics YCB mesh (the
    # /Isaac/Props/YCB/Axis_Aligned/ variants carry no collider/rigid-body, unlike Axis_Aligned_Physics/).
    # None (default) -> author NOTHING (byte-identical: cubes + the pre-rigged Axis_Aligned_Physics YCB are
    # untouched). Set to "convexHull" (Isaac's own recipe for 003-006) or "convexDecomposition" to author
    # UsdPhysics on the mesh at runtime so it becomes a graspable rigid body. Ignored for procedural cubes.
    usd_collision_approximation: str | None = None


class SimTableConfig(StrictModel):
    """Static table authored into the sim scene."""

    size_mm: tuple[float, float, float] = (800.0, 800.0, 400.0)
    position_mm: tuple[float, float, float] = (450.0, 0.0, -200.0)


class SimMarkerConfig(StrictModel):
    """Dedicated hand-eye calibration marker. Marker length + ArUco dict come from
    ``camera.hand_eye.eye_in_hand`` (single source of truth); this block holds placement + kind."""

    kind: Literal["none", "flat", "aruco"] = "none"
    position_mm: tuple[float, float, float] = (450.0, -150.0, 6.0)
    flat_size_mm: tuple[float, float, float] = (80.0, 80.0, 10.0)
    aruco_plate_size_mm: float = Field(default=72.0, gt=0.0)
    aruco_marker_id: int = Field(default=0, ge=0)


class SimViewpointConfig(StrictModel):
    """Eye-in-hand look-at viewpoint pattern (hemisphere around the marker)."""

    radii_mm: list[float] = Field(default_factory=lambda: [235.0, 275.0])
    elevations_deg: list[float] = Field(default_factory=lambda: [16.0, 30.0])
    azimuths_deg: list[float] = Field(
        default_factory=lambda: [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
    )


class SimGateConfig(StrictModel):
    """Pick/lift gate thresholds for the sim validation runners."""

    lift_threshold_mm: float = Field(default=50.0, gt=0.0)
    pass_fraction: float = Field(default=0.8, gt=0.0, le=1.0)


class SimSceneConfig(StrictModel):
    """Willy-sim scene-authoring + experiment params (read by backend.src.willy_sim only)."""

    object: SimObjectConfig = Field(default_factory=SimObjectConfig)
    objects: list[SimObjectConfig] = Field(default_factory=list)  # multi-object clutter; empty -> [object]
    table: SimTableConfig = Field(default_factory=SimTableConfig)
    marker: SimMarkerConfig = Field(default_factory=SimMarkerConfig)
    eih_viewpoints: SimViewpointConfig = Field(default_factory=SimViewpointConfig)
    gate: SimGateConfig = Field(default_factory=SimGateConfig)
    render_warmup_steps: int = Field(default=20, ge=0)


class SimConfig(StrictModel):
    """Isaac-Sim (or pure-Python mock) driver settings.

    Only consulted when ``robot.vendor == "sim"``. Mirrors the driver
    dataclass
    ``backend.src.robot.drivers.sim.config.SimRobotConfig`` so YAML
    edits stay validated even on hosts without the Isaac SDK installed.
    """

    backend: Literal["isaac"] = "isaac"
    enabled: bool = False
    mock_mode: bool = False
    # Which UR model this sim cell drives (de-locks the historically UR5e-hardcoded driver): "ur5e" (default,
    # byte-identical), "ur3e", "ur10e". Selects the Lula config + Isaac USD (scene builder) + the driver's
    # cuRobo {key}.yml. Keep the safety block's kinematics_model in sync with this. See sim/robot_models.py.
    robot_model: str = "ur5e"
    scene: str | None = None
    robot_prim_path: str | None = None
    gripper_prim_path: str | None = None
    calibration_target_prim_path: str | None = None
    home_joint_positions: list[float] | None = None
    cameras: dict[str, SimCameraSchema] = Field(default_factory=dict)
    step_dt_s: float = Field(default=1.0 / 60.0, gt=0.0, le=1.0)
    settle_timeout_s: float = Field(default=5.0, gt=0.0, le=120.0)
    headless: bool = True
    reset_on_connect: bool = True
    exec_host: str | None = None
    # --- Isaac scene-authoring extras (read by backend.src.willy_sim, NOT forwarded to the bare
    # driver dataclass; the "robot.sim" namespace owns the whole simulated work-cell). ---
    assets_root: str | None = None              # Isaac asset pack root (carb asset_root + scene build)
    gripper_variant: str = "Robotiq_2f_85"      # USD Gripper variant selection on the UR5e asset
    # When set (a key in willy_sim.grippers.MOUNTED_GRIPPERS, e.g.
    # "schunk_egu50"), the scene builder selects the "None" UR5e Gripper variant and MOUNTS that standalone
    # vendor gripper on the wrist instead (and the runtime drives it via the matching GripperProfile). None
    # (default) keeps the baked ``gripper_variant`` -> byte-identical for every existing cell.
    gripper_mount: str | None = None
    # Which suction cup to mount for a suction pick: a key in willy_sim.grippers.SUCTION_CUPS
    # (e.g. "slim" for a finer cup that reaches tighter gaps). None (default) uses the standard cup.
    # Only consulted by the suction runners; the jaw cells ignore it.
    suction_cup: str | None = None
    park_joint_positions: list[float] | None = None  # arm parked clear of an overhead cam for perception
    scene_setup: SimSceneConfig = Field(default_factory=SimSceneConfig)

    @field_validator("robot_model")
    @classmethod
    def _known_robot_model(cls, v: str) -> str:
        """Fail config validation (<1 s) on an unknown/typo'd model instead of ~60 s into an Isaac boot.

        Without this, ``ur3``/``UR3E ``/``ur-3e`` validate fine and only raise inside ``ur_model_spec`` when
        the driver brings up the articulation (drivers/sim/arm.py) or the scene builder references the USD.
        """
        if v not in UR_MODEL_KEYS:
            raise ValueError(f"unknown robot_model {v!r}; supported: {list(UR_MODEL_KEYS)}")
        return v
