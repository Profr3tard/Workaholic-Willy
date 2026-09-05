"""
Calibration solver layer: pure numpy, with no OpenCV dependency.

Exported classes:

* :class:`~src.calibration.solver.umeyama_rigid.UmeyamaRigid`
    closed-form SVD rigid point-set registration (dst ~= R @ src + t).
* :class:`~src.calibration.solver.hand_eye_axxb.HandEyeAXXB`
    closed-form AX = XB hand-eye calibration (Kronecker and SVD method).
  Supports both eye-to-hand and eye-in-hand configurations.
"""

from __future__ import annotations

from .hand_eye_axxb import HandEyeAXXB
from .umeyama_rigid import UmeyamaRigid

__all__ = [
    "HandEyeAXXB",
    "UmeyamaRigid",
]
