"""
The closed-form calibration solvers: numpy only, with no OpenCV dependency.

* :class:`~src.calibration.solver.umeyama_rigid.UmeyamaRigid` registers two
  corresponding point sets by SVD, solving ``dst ~= R @ src + t``.
* :class:`~src.calibration.solver.hand_eye_axxb.HandEyeAXXB` solves
  ``AX = XB`` by the Kronecker-product and SVD route. It serves eye-to-hand
  and eye-in-hand alike; which one it is follows from the ``A`` and ``B`` the
  caller supplies, not from anything this layer holds.
"""

from __future__ import annotations

from .hand_eye_axxb import HandEyeAXXB
from .umeyama_rigid import UmeyamaRigid

__all__ = [
    "HandEyeAXXB",
    "UmeyamaRigid",
]
