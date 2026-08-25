"""Suction/vacuum grasp modality for sealable, flat-ish surfaces.

Defines suction grasping as a separate end-effector modality from parallel
jaws. Candidates are ranked using analytical seal-formation and wrench-
resistance physics through the ``SuctionScorer`` seam, with no learned
implementation included. Pure NumPy; Isaac handles attach/lift simulation,
while final seal physics remains hardware-dependent.
"""

from __future__ import annotations

from src.robot.grasping.suction.scorer import (
    AnalyticalSuctionScorer,
    SuctionQuality,
    SuctionScorer,
)
from src.robot.grasping.suction.seal import (
    SealConfig,
    SealResult,
    evaluate_seal,
)
from src.robot.grasping.suction.synthesis import (
    SuctionConfig,
    SuctionGrasp,
    synthesize_suction_grasps,
)
from src.robot.grasping.suction.wrench import (
    WrenchConfig,
    WrenchResult,
    evaluate_wrench_resistance,
)

__all__ = [
    # seal
    "SealConfig",
    "SealResult",
    "evaluate_seal",
    # wrench
    "WrenchConfig",
    "WrenchResult",
    "evaluate_wrench_resistance",
    # scorer seam
    "SuctionQuality",
    "SuctionScorer",
    "AnalyticalSuctionScorer",
    # synthesis
    "SuctionConfig",
    "SuctionGrasp",
    "synthesize_suction_grasps",
]
