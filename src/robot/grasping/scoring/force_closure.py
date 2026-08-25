"""Analytical force-closure certificate for parallel-jaw grasps.

Evaluates the two-contact Coulomb friction-cone condition from contact points,
surface normals, and friction coefficient. The certificate is contact-local
and analytical; it indicates force closure under the model but does not
guarantee physical grasp success or replace hardware force feedback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.robot.grasping.contacts import ContactPair

__all__ = [
    "ForceClosureCertificate",
    "certify_contact_pair",
    "friction_threshold",
]


# Sentinel value returned for any inner-product cosine that cannot be
# computed (zero-length closing axis, NaN normals, ...).
_INVALID_COS = -1.0


def friction_threshold(friction_coefficient: float) -> float:
    """
    Return ``cos(atan(mu)) = 1 / sqrt(1 + mu**2)``;
    raises :class:`ValueError` for non-finite or negative ``mu``.
    """
    value = float(friction_coefficient)
    if not np.isfinite(value):
        raise ValueError(
            f"friction_coefficient must be finite; got {friction_coefficient!r}"
        )
    if value < 0.0:
        raise ValueError(
            f"friction_coefficient must be >= 0; got {friction_coefficient!r}"
        )
    return float(1.0 / np.sqrt(1.0 + value * value))


@dataclass(frozen=True, slots=True)
class ForceClosureCertificate:
    """
    Analytical 2-contact force-closure certificate: ``is_certified`` iff both friction-cone cosines
    (``cos_alpha_a = -dot(v, n_a)``, ``cos_alpha_b = dot(v, n_b)``, clipped to ``[-1, 1]``) are
    ``>= cos_threshold``, and ``margin`` is ``min(cos_alpha_a, cos_alpha_b) - cos_threshold``.
    """

    is_certified: bool
    friction_coefficient: float
    cos_threshold: float
    cos_alpha_a: float
    cos_alpha_b: float
    margin: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly snapshot suitable for pose metadata."""
        return {
            "is_certified": bool(self.is_certified),
            "friction_coefficient": float(self.friction_coefficient),
            "cos_threshold": float(self.cos_threshold),
            "cos_alpha_a": float(self.cos_alpha_a),
            "cos_alpha_b": float(self.cos_alpha_b),
            "margin": float(self.margin),
        }


def certify_contact_pair(
    pair: ContactPair, friction_coefficient: float
) -> ForceClosureCertificate:
    """Return the analytical (deterministic) force-closure certificate for a contact pair."""
    if not isinstance(pair, ContactPair):
        raise TypeError("pair must be a ContactPair")

    cos_threshold = friction_threshold(friction_coefficient)

    delta = np.asarray(pair.point_b, dtype=np.float64) - np.asarray(
        pair.point_a, dtype=np.float64
    )
    distance = float(np.linalg.norm(delta))
    if distance < 1e-9:
        cos_alpha_a = _INVALID_COS
        cos_alpha_b = _INVALID_COS
    else:
        closing = delta / distance
        # Outward normals at each contact, by ContactPair convention.
        n_a = np.asarray(pair.normal_a, dtype=np.float64)
        n_b = np.asarray(pair.normal_b, dtype=np.float64)
        cos_alpha_a = float(np.clip(-float(np.dot(closing, n_a)), -1.0, 1.0))
        cos_alpha_b = float(np.clip(float(np.dot(closing, n_b)), -1.0, 1.0))

    worst = min(cos_alpha_a, cos_alpha_b)
    margin = worst - cos_threshold
    is_certified = bool(
        cos_alpha_a >= cos_threshold and cos_alpha_b >= cos_threshold
    )
    return ForceClosureCertificate(
        is_certified=is_certified,
        friction_coefficient=float(friction_coefficient),
        cos_threshold=float(cos_threshold),
        cos_alpha_a=cos_alpha_a,
        cos_alpha_b=cos_alpha_b,
        margin=float(margin),
    )
