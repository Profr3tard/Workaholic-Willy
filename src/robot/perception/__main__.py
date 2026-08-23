"""Smoke-exercise the live-camera PerceptionSource against a real RGB-D camera.

    (S) Needs a real Intel RealSense (D435/D435i) + pyrealsense2
    + the perception models (GroundingDINO + SAM2, on the GPU box).

    python -m src.robot.perception --prompt "a red cube"
    python -m src.robot.perception --prompt "cube ; screwdriver ; mug"   # multi-phrase clutter
    python -m src.robot.perception --rig realsense_d435 --warmup 10
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from config.schema.camera import RGBDDeviceRigConfig


def _find_rgbd_rig(camera_cfg: Any, rig_id: str | None) -> RGBDDeviceRigConfig:
    """Pick the RGB-D rig to open: the named one, or the single realsense rig if unambiguous."""
    rigs = list(getattr(getattr(camera_cfg, "cameras", None), "rigs", []) or [])
    rgbd = [r for r in rigs if getattr(r, "rgbd_backend", None) == "realsense"]
    if rig_id is not None:
        for r in rgbd:
            if getattr(r, "rig_id", None) == rig_id:
                return cast("RGBDDeviceRigConfig", r)
        raise SystemExit(
            f"no realsense rig with rig_id={rig_id!r}; rgbd rigs: {[getattr(r, 'rig_id', '?') for r in rgbd]}"
        )
    if len(rgbd) == 1:
        return cast("RGBDDeviceRigConfig", rgbd[0])
    raise SystemExit(
        f"found {len(rgbd)} realsense rigs; pass --rig <rig_id>. "
        f"candidates: {[getattr(r, 'rig_id', '?') for r in rgbd]}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Grab one RGB-D frame, detect+segment, print stats. No robot.")
    ap.add_argument("--prompt", default="an object", help="GroundingDINO phrase(s); ' ; '-separate for clutter")
    ap.add_argument("--rig", default=None, help="rig_id of the RGB-D rig (default: the only realsense rig)")
    ap.add_argument("--warmup", type=int, default=5, help="throwaway grabs so auto-exposure settles")
    args = ap.parse_args(argv)

    # Deferred: config is light, but the streamer pulls pyrealsense2 and the factory pulls torch.
    from config import load_config
    from src.camera.setup.image_taking.rgbd import RealSenseRGBDStreamer
    from src.models.factory import build_object_detector, build_segmenter
    from src.robot.perception import RealSenseVisionPerceptionSource

    cfg = load_config()
    rig = _find_rgbd_rig(cfg.camera, args.rig)
    streamer = RealSenseRGBDStreamer(rig)
    detector = build_object_detector(cfg.models)
    segmenter = build_segmenter(cfg.models)
    source = RealSenseVisionPerceptionSource(
        streamer=streamer, detector=detector, segmenter=segmenter,
        prompt=args.prompt, warmup_grabs=args.warmup,
    )

    print(f"opening rig {getattr(rig, 'rig_id', '?')!r} ...", flush=True)
    streamer.open()
    try:
        frame = source.acquire()
    finally:
        streamer.release()

    depth = np.asarray(frame.depth_map)
    total = depth.size or 1
    print(f"intrinsics ({source.intrinsics_source}):\n{np.asarray(frame.intrinsics)}")
    print(f"depth {depth.shape}: {int((depth == 0).sum())}/{total} holes "
          f"({100.0 * (depth == 0).mean():.1f}%), non-hole range "
          f"[{float(depth[depth > 0].min()) if (depth > 0).any() else 0:.0f}, {float(depth.max()):.0f}] mm")
    print(f"segmentations: {len(frame.segmentations)}")
    for i, seg in enumerate(frame.segmentations):
        mask = np.asarray(seg.mask).astype(bool)
        if mask.any():
            in_mask = depth[mask]
            holes = int((in_mask == 0).sum())
            print(f"  [{i}] label={getattr(seg, 'label', '?')!r} mask_px={int(mask.sum())} "
                  f"depth-holes-in-mask={holes}/{int(mask.sum())} "
                  f"({100.0 * holes / max(1, int(mask.sum())):.1f}%)")
        else:
            print(f"  [{i}] label={getattr(seg, 'label', '?')!r} mask_px=0 (empty)")
    if not frame.segmentations:
        print("  (nothing grounded check the prompt, the lighting, and that the object is in view)")
    print("OK - grabbed, detected, segmented. No robot touched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
