"""Frozen schema constants + ArtifactSchemaError for the success-probability model."""

from __future__ import annotations


FEATURE_SCHEMA_VERSION: int = 1

#: Artifact version of the logistic-regression + isotonic family (the shipped synthetic bootstrap).
MODEL_ARTIFACT_VERSION: int = 1
#: Artifact version of the gradient-boosted-trees + isotonic family (the more expressive model, intended
#: for a real grasp-outcome corpus on the logistic-structured synthetic bootstrap it does not beat
#: logistic, so the shipped bootstrap stays logistic; train GBT explicitly with real data).
GBT_ARTIFACT_VERSION: int = 2
#: Artifact version of the multi-layer-perceptron + isotonic family (the strongest family; standardised
#: inputs -> ReLU MLP -> single raw logit -> isotonic). Like GBT it is intended for a REAL grasp-outcome
#: corpus and only ships if it wins the A/B (Brier/ECE/AUROC) against the incumbent.
MLP_ARTIFACT_VERSION: int = 3

#: Model families the artifact can carry.
MODEL_FAMILY_LOGISTIC: str = "logistic_regression"
MODEL_FAMILY_GBT: str = "gradient_boosted_trees"
MODEL_FAMILY_MLP: str = "mlp"

#: Every family the runtime knows, in increasing expressiveness.
MODEL_FAMILIES: tuple[str, ...] = (MODEL_FAMILY_LOGISTIC, MODEL_FAMILY_GBT, MODEL_FAMILY_MLP)

#: Artifact versions the runtime predictor can load (it branches on ``model_family``).
SUPPORTED_ARTIFACT_VERSIONS: frozenset[int] = frozenset(
    {MODEL_ARTIFACT_VERSION, GBT_ARTIFACT_VERSION, MLP_ARTIFACT_VERSION}
)

#: Locked order of the 23 features the model consumes. Position is part of
#: the artifact contract adding, removing, or reordering requires bumping
#: :data:`FEATURE_SCHEMA_VERSION` and producing a new model directory under
#: ``assets/models/success_probability/v<N>/``.
FEATURE_NAMES: tuple[str, ...] = (
    # Top-level scalars from GraspScoreBreakdown (4)
    "geometric_score",
    "stability_score",
    "reachability_score",
    "feasibility_score",
    # Geometric subscores (4)
    "geo_antipodal",
    "geo_normal_opposition",
    "geo_axis_alignment",
    "geo_width_fit",
    # Stability subscores (4)
    "stab_confidence",
    "stab_contact_stability",
    "stab_collision_free",
    "stab_table_clearance",
    # Reachability subscores (2)
    "reach_workspace",
    "reach_approach_alignment",
    # Feasibility subscores (4)
    "feas_ik_quality",
    "feas_joint_margin",
    "feas_approach_clearance",
    "feas_corridor_risk",
    # Pose-level numerics (2)
    "pose_grip_width_mm",
    "pose_confidence",
    # Mode one-hot (3)
    "mode_easy",
    "mode_dense",
    "mode_auto",
)

#: The three mode buckets the v1 schema collapses every raw mode string into.
MODE_BUCKETS: tuple[str, ...] = ("easy", "dense", "auto")

_MODE_BUCKET_INDEX: dict[str, int] = {name: idx for idx, name in enumerate(MODE_BUCKETS)}

#: Index of each feature in the canonical vector; used by extractor + loader.
_FEATURE_INDEX: dict[str, int] = {name: idx for idx, name in enumerate(FEATURE_NAMES)}

_N_FEATURES: int = len(FEATURE_NAMES)


class ArtifactSchemaError(ValueError):
    """Raised when a model artifact does not match the locked schema."""
