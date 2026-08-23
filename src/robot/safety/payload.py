"""
PayloadGuard tool / payload envelope safety guard.

Per-move evaluation is intentionally cheap: it verifies that the
*currently configured* payload still fits the operator-declared
envelope. The heavy lift pushing the payload mass + CoG into the UR
controller via RTDE ``setPayload`` happens in
:meth:`URRobotArm.connect` so the controller's protective-stop
calculations see the same values Willy believes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .decision import SafetyDecision, SafetyReason
from .guard import SafetyContext

if TYPE_CHECKING:  # pragma: no cover - typing only
    from config.schema.robot import PayloadSafetyConfig

__all__ = ["PayloadGuard"]


class PayloadGuard:
    """Mass / CoG / inertia envelope guard.

    Stateless across calls. Constructed by
    :meth:`SafetyPreflight.from_safety_config` when
    ``safety.payload.enforce`` is ``True``.
    """

    name = "payload"

    def __init__(self, config: "PayloadSafetyConfig") -> None:
        self._config = config

    def evaluate(self, ctx: SafetyContext) -> SafetyDecision:
        # Defence-in-depth: ``PayloadSafetyConfig`` (Pydantic) is the PRIMARY
        # validator and already rejects mass<0 / mass>max / inertia<0 at construction
        # (see ``test_safety_payload`` schema cases). The three runtime checks below
        # are an intentional second line, they still fail CLOSED for a frozen config
        # assembled by a path that bypassed schema validation (a grasping-preset
        # overlay, or a hand-built config in a test) rather than letting a bad payload
        # reach the controller. They are NOT redundant; do not delete them.
        cfg = self._config

        if cfg.mass_kg < 0.0:
            return SafetyDecision.reject(
                self.name,
                SafetyReason.PAYLOAD,
                message=(
                    f"payload.mass_kg ({cfg.mass_kg:.6f}) must be >= 0"
                ),
                detail={
                    "reason": "negative_mass",
                    "mass_kg": f"{cfg.mass_kg:.6f}",
                },
            )

        if cfg.mass_kg > cfg.max_mass_kg:
            return SafetyDecision.reject(
                self.name,
                SafetyReason.PAYLOAD,
                message=(
                    f"payload.mass_kg ({cfg.mass_kg:.6f}) exceeds "
                    f"max_mass_kg ({cfg.max_mass_kg:.6f})"
                ),
                detail={
                    "reason": "over_mass",
                    "mass_kg": f"{cfg.mass_kg:.6f}",
                    "max_mass_kg": f"{cfg.max_mass_kg:.6f}",
                },
            )

        for axis_label, val in zip(("Ixx", "Iyy", "Izz"), cfg.inertia_kgm2):
            if val < 0.0:
                return SafetyDecision.reject(
                    self.name,
                    SafetyReason.PAYLOAD,
                    message=(
                        f"payload.inertia_kgm2.{axis_label} "
                        f"({val:.6f}) must be >= 0"
                    ),
                    detail={
                        "reason": "negative_inertia",
                        "axis": axis_label,
                        "value": f"{val:.6f}",
                    },
                )

        return SafetyDecision.accept(self.name)
