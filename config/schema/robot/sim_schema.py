"""Isaac-Sim driver + scene config schema."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from .._base import StrictModel

from ._ur_models import UR_MODEL_KEYS  # re-exported: the sim cell and the real UR cell share one list


class SimCameraSchema(StrictModel):
    """Per-camera entry inside :class:`SimConfig.cameras`."""

    prim_path: str = Field(default="", min_length=0)
    mounting_mode: Literal["eye_to_hand", "eye_in_hand"] = "eye_to_hand"
    render_product_name: str | None = None
    # --- Isaac authoring extras (read by backend.src.willy_sim, not the bare driver) ---
    near_clip_m: float | None = Field(default=None, gt=0.0)
    resolution: tuple[int, int] | None = None
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
    shape: str = "cube"
    size_mm: tuple[float, float, float] = (30.0, 30.0, 50.0)
    position_mm: tuple[float, float, float] = (450.0, 0.0, 25.0)
    mass_kg: float = Field(default=0.05, gt=0.0)
    static_friction: float = Field(default=1.5, ge=0.0)
    dynamic_friction: float = Field(default=1.3, ge=0.0)
    color: tuple[float, float, float] = (0.9, 0.2, 0.1)  # RGB 0-1; the VL colour disambiguation handle
    usd_asset_path: str | None = None  # relative /Isaac/... USD asset (YCB mesh); None -> DynamicCuboid
    orientation_wxyz: tuple[float, float, float, float] | None = None
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
    """Isaac-Sim (or pure-Python mock) driver settings."""

    backend: Literal["isaac"] = "isaac"
    enabled: bool = False
    mock_mode: bool = False
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
    assets_root: str | None = None              # Isaac asset pack root (carb asset_root + scene build)
    gripper_variant: str = "Robotiq_2f_85"      # USD Gripper variant selection on the UR5e asset
    gripper_mount: str | None = None
    suction_cup: str | None = None
    park_joint_positions: list[float] | None = None  # arm parked clear of an overhead cam for perception
    scene_setup: SimSceneConfig = Field(default_factory=SimSceneConfig)

    @field_validator("robot_model")
    @classmethod
    def _known_robot_model(cls, v: str) -> str:
        """Fail config validation (<1 s) on an unknown/typo'd model instead of ~60 s into an Isaac boot."""
        if v not in UR_MODEL_KEYS:
            raise ValueError(f"unknown robot_model {v!r}; supported: {list(UR_MODEL_KEYS)}")
        return v
