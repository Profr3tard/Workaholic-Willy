"""An RGB-D ArUco marker source for the hand-eye routine's ``marker_source`` seam.

``CalibrationRoutine`` accepts an injected ``MarkerPoseProvider = Callable[[], Optional[np.ndarray]]``
(``robot/execution/calibration.py``) that overrides the default stereo-rectified path. The two sim
runners fill it with an Isaac source; a real eye-to-hand d435 rig had nothing to fill it with,
because the shipped default assumes a stereo rig and ``FrameProvider`` raises for an RGB-D one. This is that
missing piece: grab a colour frame, detect the board with the generic mono ``ArucoPoseEstimator``
(detect + ``SOLVEPNP_IPPE_SQUARE``), return ``T_cam_to_marker`` (4x4) or ``None``.

It is glue, not new maths: ``ArucoPoseEstimator`` already exists and is unit-tested against synthetic
markers; the routine's consumption side is finished and fail-closed. The streamer is duck-typed
(``grab`` / ``get_intrinsics`` / ``get_distortion``), so this module imports no camera code and no
``pyrealsense2``, since the real streamer is injected by the bring-up runner.

Honesty: bucket (2). Unit-tested offline against a rendered marker (real evidence before September), but
never run against a physical d435. The distortion path (decision D1) is exercised but the on-aligned-
stream coefficients are ~0, so a real bench must confirm the residual.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from src.calibration.stereo.sub_modules.aruco_esti import ArucoPoseEstimator

__all__ = ["RGBDArucoMarkerSource"]


class RGBDArucoMarkerSource:
    """Callable ``MarkerPoseProvider``: one grab -> one ``T_cam_to_marker`` (4x4) or ``None``.

    Parameters
    ----------
    streamer
        Anything with ``grab() -> RGBDFrame`` (``.color`` BGR uint8), ``get_intrinsics() -> 3x3 K`` and
        ``get_distortion() -> dist | None``. In production the ``RealSenseRGBDStreamer``.
    marker_length_mm, dict_name
        The physical board: the ArUco square edge in mm and the dictionary. A mismatch here scales every
        sample uniformly, so the solve converges and is uniformly wrong. Measure the printed edge.
    target_id
        The marker id to pose. A single id keeps the provider's return an ``Optional[np.ndarray]``.
    intrinsics, distortion
        Decision **D1**: leave ``None`` to use the streamer's factory k/dist, or pass a bench-calibrated
        pair (e.g. from :func:`camera.setup.image_taking.intrinsics.load_intrinsics`) to override. The
        source is recorded on ``intrinsics_source`` for provenance.
    warmup_grabs
        Throwaway grabs so a real camera's auto-exposure settles before the pose read.
    """

    def __init__(
        self,
        *,
        streamer: Any,
        marker_length_mm: float = 50.0,
        dict_name: str = "DICT_5X5_100",
        target_id: int = 0,
        intrinsics: np.ndarray | None = None,
        distortion: np.ndarray | None = None,
        warmup_grabs: int = 3,
    ) -> None:
        self._streamer = streamer
        self._estimator = ArucoPoseEstimator(marker_length_mm=marker_length_mm, dict_name=dict_name)
        self._target_id = int(target_id)
        self._k_override = None if intrinsics is None else np.asarray(intrinsics, dtype=np.float64)
        self._dist_override = None if distortion is None else np.asarray(distortion, dtype=np.float64)
        self._warmup = max(0, int(warmup_grabs))
        self.intrinsics_source = "override" if intrinsics is not None else "factory"

    def __call__(self) -> Optional[np.ndarray]:
        for _ in range(self._warmup):
            self._streamer.grab()
        frame = self._streamer.grab()
        bgr = np.ascontiguousarray(np.asarray(frame.color))

        if self._k_override is not None:
            k = self._k_override
        else:
            k = self._streamer.get_intrinsics()
            if k is None:
                raise RuntimeError(
                    "streamer.get_intrinsics() is None; open() the streamer first, or pass a "
                    "calibrated intrinsics= (the D1 override)."
                )
        if self._dist_override is not None:
            dist = self._dist_override
        else:
            dist = self._streamer.get_distortion()
            if dist is None:
                dist = np.zeros(5)  # device reported none -> honest zero, not a fabricated value

        result = self._estimator.estimate(bgr, np.asarray(k, dtype=np.float64), dist, target_id=self._target_id)
        # With a target_id, estimate() returns a single 4x4 or None, exactly the MarkerPoseProvider shape.
        return result if result is None else np.asarray(result, dtype=np.float64)
