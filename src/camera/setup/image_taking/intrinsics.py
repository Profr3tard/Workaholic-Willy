"""Read a stored camera intrinsics file back into ``(K, dist)``.

The RealSense streamer writes ``intrinsics.json`` (``_export_intrinsics``) but, until now, nothing read
it back: ``grep intrinsics_file`` found only the schema and the writer. This is the missing loader, and
the read side of decision **D1** (offer both factory and calibrated intrinsics):

  * Factory: the device's own ``get_intrinsics()`` / ``get_distortion()``, live after ``open()``.
  * Calibrated: an ``intrinsics.json`` produced either by the streamer's factory export or by a bench
    ``cv2.calibrateCamera`` run. This loader reads whichever is on disk.

A consumer that wants the factory K uses the streamer directly; one that wants a bench-calibrated K
points this loader at the file. Neither is imposed, so the operator chooses, and can compare the two by
the reprojection residual the calibration run reports.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = ["load_intrinsics"]


def load_intrinsics(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ``intrinsics.json`` into a 3x3 ``K`` and a distortion vector.

    Accepts the streamer's ``fx/fy/cx/cy(+dist)`` schema. A missing or empty ``dist`` yields ``zeros(5)``
    -- the honest "no distortion known" default a PnP solve expects. Raises ``FileNotFoundError`` if the
    file is absent (fail loud: a silently-defaulted K is worse than a stop) and ``ValueError`` on a
    malformed one.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"intrinsics file not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"intrinsics file {p} is not valid JSON: {exc}") from exc

    try:
        fx, fy, cx, cy = (float(data[k]) for k in ("fx", "fy", "cx", "cy"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"intrinsics file {p} must carry numeric fx/fy/cx/cy: {exc}") from exc
    if not all(np.isfinite(v) for v in (fx, fy, cx, cy)) or fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"intrinsics file {p} has non-physical fx/fy: {fx}, {fy}")

    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist_raw = data.get("dist") or []
    dist = np.asarray(dist_raw, dtype=np.float64).reshape(-1) if dist_raw else np.zeros(5)
    if not np.all(np.isfinite(dist)):
        raise ValueError(f"intrinsics file {p} has non-finite distortion coefficients")
    return k, dist
