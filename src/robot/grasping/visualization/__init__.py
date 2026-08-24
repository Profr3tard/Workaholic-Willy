"""
Debug visualisation primitives for the grasping pipeline.

Two surfaces:

* :mod:`debug_draw` pure 2D. Renders RGB + mask overlay +
  projected gripper boxes + contact arrows + score bar.
* :mod:`open3d_viewer` —D Open3D viewer for offline debugging.

Both surfaces accept the same ``GraspPoint`` / point-cloud inputs so the
ops surface stays consistent regardless of where you inspect the scene.
"""

from .debug_draw import (
    DebugDrawConfig,
    draw_grasp_debug_image,
    save_grasp_debug_image,
)
from .open3d_viewer import (
    GraspViewerConfig,
    build_grasp_scene,
    show_grasp_scene,
)

__all__ = [
    "DebugDrawConfig",
    "GraspViewerConfig",
    "build_grasp_scene",
    "draw_grasp_debug_image",
    "save_grasp_debug_image",
    "show_grasp_scene",
]
