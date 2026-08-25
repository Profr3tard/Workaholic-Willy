"""Pure-NumPy success-probability prediction and artifact loading.

Supports logistic regression and gradient-boosted tree artifacts through a
shared runtime schema, with training-mean imputation and isotonic
calibration. Runtime inference is NumPy-only; scikit-learn is used only
offline for training.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np

from src.robot.grasping.constants import (
    SUCCESS_MODEL_LOG_FILE,
    create_grasping_logger,
)

from ..feasibility_score import feasibility_score_components
from ..geometric_score import geometric_score_components
from ..reachability_score import reachability_score_components
from ..stability_score import stability_score_components
from ._features import extract_features
from ._schema import (
    MODEL_FAMILY_GBT,
    MODEL_FAMILY_LOGISTIC,
    MODEL_FAMILY_MLP,
    SUPPORTED_ARTIFACT_VERSIONS,
    ArtifactSchemaError,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    _N_FEATURES,
)


# Logging for this module.
logger = create_grasping_logger("SuccessProbabilityModel", SUCCESS_MODEL_LOG_FILE)


def _sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def _isotonic_interp(raw: np.ndarray, x_thresholds: np.ndarray, y_thresholds: np.ndarray) -> np.ndarray:
    # Vectorised isotonic interpolation: monotone 1-D mapping from raw -> calibrated.
    return np.interp(raw, x_thresholds, y_thresholds)


@dataclass(frozen=True, slots=True)
class GbtTree:
    """One serialized regression tree. Leaves have ``feature == -1``; internal nodes split on
    ``x[feature] <= threshold`` (go ``left``, else ``right``)."""

    feature: np.ndarray  # (n_nodes,) int64, -1 at leaves
    threshold: np.ndarray  # (n_nodes,) float64
    left: np.ndarray  # (n_nodes,) int64
    right: np.ndarray  # (n_nodes,) int64
    value: np.ndarray  # (n_nodes,) float64 leaf contribution


@dataclass(frozen=True, slots=True)
class GbtEnsemble:
    """A gradient-boosted tree ensemble: ``raw = init_score + learning_rate * sum(tree leaf values)``."""

    learning_rate: float
    init_score: float
    trees: tuple[GbtTree, ...]


def _gbt_tree_leaf_values(tree: GbtTree, x: np.ndarray) -> np.ndarray:
    """Vectorised traversal: the leaf value each row of ``x`` falls into."""
    node = np.zeros(x.shape[0], dtype=np.int64)
    active = tree.feature[node] >= 0
    while np.any(active):
        idx = np.nonzero(active)[0]
        nd = node[idx]
        go_left = x[idx, tree.feature[nd]] <= tree.threshold[nd]
        node[idx] = np.where(go_left, tree.left[nd], tree.right[nd])
        active = tree.feature[node] >= 0
    return tree.value[node]


def _gbt_raw_predict(ensemble: GbtEnsemble, x: np.ndarray) -> np.ndarray:
    total = np.full(x.shape[0], float(ensemble.init_score), dtype=np.float64)
    for tree in ensemble.trees:
        total = total + ensemble.learning_rate * _gbt_tree_leaf_values(tree, x)
    return total


@dataclass(frozen=True, slots=True)
class MlpLayer:
    """One dense layer: ``out = h @ weight + bias``; ``weight`` has shape ``(n_in, n_out)``."""

    weight: np.ndarray
    bias: np.ndarray


@dataclass(frozen=True, slots=True)
class MlpNetwork:
    """ReLU multi-layer perceptron over STANDARDISED features; the last layer emits one raw logit."""

    layers: tuple[MlpLayer, ...]
    activation: str = "relu"


def _mlp_raw_predict(network: MlpNetwork, z: np.ndarray) -> np.ndarray:
    """Forward pass on standardised inputs -> one raw logit per row."""

    h = z
    for layer in network.layers[:-1]:
        h = np.maximum(h @ layer.weight + layer.bias, 0.0)
    out = network.layers[-1]
    return np.asarray(h @ out.weight + out.bias, dtype=np.float64)[:, 0]


@dataclass(frozen=True, slots=True)
class SuccessProbabilityModel:
    """Immutable in-memory success-probability artifact (all tensors ``float64``)."""

    schema_version: int
    artifact_version: int
    model_family: str
    feature_names: tuple[str, ...]
    feature_means: np.ndarray  # shape (n_features,)
    isotonic_x: np.ndarray  # shape (k,)
    isotonic_y: np.ndarray  # shape (k,)
    manifest: Mapping[str, Any]
    feature_stds: np.ndarray | None = None  # logistic + mlp
    logistic_coefficients: np.ndarray | None = None  # logistic only
    logistic_intercept: float | None = None  # logistic only
    gbt: GbtEnsemble | None = None  # gradient-boosted trees only
    mlp: MlpNetwork | None = None  # mlp only

    def predict_proba_one(
        self,
        breakdown: Any,
        *,
        mode: str | None,
        component_overrides: Mapping[str, Mapping[str, float]] | None = None,
    ) -> float:
        """Convenience: extract -> impute -> predict for one pose."""
        features = extract_features(breakdown, mode=mode, component_overrides=component_overrides)
        return float(predict_proba(self, features.reshape(1, -1))[0])

    def to_dict(self) -> dict[str, Any]:
        """Round-trip-safe JSON serialisation used by the trainer exporter."""
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "artifact_version": self.artifact_version,
            "model_family": self.model_family,
            "feature_names": list(self.feature_names),
            "feature_means": self.feature_means.tolist(),
            "isotonic_x": self.isotonic_x.tolist(),
            "isotonic_y": self.isotonic_y.tolist(),
        }
        if self.model_family == MODEL_FAMILY_GBT:
            assert self.gbt is not None
            payload["gbt"] = {
                "learning_rate": float(self.gbt.learning_rate),
                "init_score": float(self.gbt.init_score),
                "trees": [
                    {
                        "feature": t.feature.tolist(),
                        "threshold": t.threshold.tolist(),
                        "left": t.left.tolist(),
                        "right": t.right.tolist(),
                        "value": t.value.tolist(),
                    }
                    for t in self.gbt.trees
                ],
            }
        elif self.model_family == MODEL_FAMILY_MLP:
            assert self.feature_stds is not None and self.mlp is not None
            payload["feature_stds"] = self.feature_stds.tolist()
            payload["mlp"] = {
                "activation": self.mlp.activation,
                "layers": [
                    {"weight": layer.weight.tolist(), "bias": layer.bias.tolist()}
                    for layer in self.mlp.layers
                ],
            }
        else:
            assert self.feature_stds is not None and self.logistic_coefficients is not None
            payload["feature_stds"] = self.feature_stds.tolist()
            payload["logistic_coefficients"] = self.logistic_coefficients.tolist()
            payload["logistic_intercept"] = float(cast(float, self.logistic_intercept))
        return payload


def predict_proba(model: SuccessProbabilityModel, features: np.ndarray) -> np.ndarray:
    """Vectorised calibrated predict for a ``(n_samples, n_features)`` matrix (NaNs -> per-feature training mean; output in ``[0, 1]``)."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("features must be 2-D (n_samples, n_features)")
    if x.shape[1] != model.feature_means.shape[0]:
        raise ValueError(
            f"feature dim mismatch: got {x.shape[1]}, model expects {model.feature_means.shape[0]}"
        )

    means = model.feature_means
    nan_mask = ~np.isfinite(x)
    if nan_mask.any():
        x = np.where(nan_mask, means, x)

    if model.model_family == MODEL_FAMILY_GBT:
        assert model.gbt is not None
        raw = _gbt_raw_predict(model.gbt, x)
    else:
        # logistic + mlp share the standardisation step
        assert model.feature_stds is not None
        stds = np.where(model.feature_stds > 1e-12, model.feature_stds, 1.0)
        z = (x - means) / stds
        if model.model_family == MODEL_FAMILY_MLP:
            assert model.mlp is not None
            raw = _mlp_raw_predict(model.mlp, z)
        else:
            assert model.logistic_coefficients is not None
            raw = z @ model.logistic_coefficients + float(cast(float, model.logistic_intercept))

    raw_proba = cast(np.ndarray, _sigmoid(raw))
    calibrated = _isotonic_interp(raw_proba, model.isotonic_x, model.isotonic_y)
    return np.clip(calibrated, 0.0, 1.0)


def load_success_probability_model(artifact_dir: str | Path) -> SuccessProbabilityModel:
    """
    Load an on-disk artifact (``model.json`` + ``manifest.json``);
    schema mismatches raise :class:`ArtifactSchemaError` so callers can fail safely.
    """
    base = Path(artifact_dir)
    model_path = base / "model.json"
    manifest_path = base / "manifest.json"
    if not model_path.is_file():
        raise ArtifactSchemaError(f"missing model.json under {base}")
    if not manifest_path.is_file():
        raise ArtifactSchemaError(f"missing manifest.json under {base}")

    with model_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    model = _model_from_payload(payload, manifest)
    # Only the shipped model is a *synthetic bootstrap*; a run that silently keeps
    # predicting with it is a known trap, so the manifest provenance goes in the log.
    logger.info(
        "Loaded success-probability model from %s: family=%s artifact_version=%d "
        "features=%d model_id=%s dataset=%s lifecycle=%s",
        base,
        model.model_family,
        model.artifact_version,
        len(model.feature_names),
        manifest.get("model_id", "?"),
        manifest.get("dataset_id", "?"),
        manifest.get("lifecycle_phase", "?"),
    )
    return model


def _require_isotonic(payload: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    iso_x = np.asarray(payload["isotonic_x"], dtype=np.float64)
    iso_y = np.asarray(payload["isotonic_y"], dtype=np.float64)
    if iso_x.ndim != 1 or iso_y.ndim != 1 or iso_x.shape != iso_y.shape:
        raise ArtifactSchemaError("isotonic_x and isotonic_y must be 1-D and same length")
    if iso_x.size < 2:
        raise ArtifactSchemaError("isotonic calibration needs >= 2 knot points")
    if np.any(np.diff(iso_x) < 0) or np.any(np.diff(iso_y) < 0):
        raise ArtifactSchemaError("isotonic_x and isotonic_y must be monotonically non-decreasing")
    return iso_x, iso_y


def _parse_gbt(payload: Mapping[str, Any]) -> GbtEnsemble:
    raw = payload.get("gbt")
    if not isinstance(raw, Mapping) or "trees" not in raw:
        raise ArtifactSchemaError("gradient_boosted_trees artifact missing a 'gbt' block")
    trees: list[GbtTree] = []
    for node in raw["trees"]:
        feature = np.asarray(node["feature"], dtype=np.int64)
        arrays = {k: np.asarray(node[k], dtype=np.int64) for k in ("left", "right")}
        threshold = np.asarray(node["threshold"], dtype=np.float64)
        value = np.asarray(node["value"], dtype=np.float64)
        n = feature.shape[0]
        if any(a.shape != (n,) for a in (*arrays.values(), threshold, value)):
            raise ArtifactSchemaError("gbt tree arrays must share one length")
        trees.append(GbtTree(feature, threshold, arrays["left"], arrays["right"], value))
    return GbtEnsemble(
        learning_rate=float(raw["learning_rate"]),
        init_score=float(raw["init_score"]),
        trees=tuple(trees),
    )


def _parse_mlp(payload: Mapping[str, Any]) -> MlpNetwork:
    raw = payload.get("mlp")
    if not isinstance(raw, Mapping) or "layers" not in raw:
        raise ArtifactSchemaError("mlp artifact missing an 'mlp' block")
    activation = str(raw.get("activation", "relu"))
    if activation != "relu":
        raise ArtifactSchemaError(f"unsupported mlp activation {activation!r}")
    layers: list[MlpLayer] = []
    for node in raw["layers"]:
        weight = np.asarray(node["weight"], dtype=np.float64)
        bias = np.asarray(node["bias"], dtype=np.float64)
        if weight.ndim != 2 or bias.ndim != 1 or weight.shape[1] != bias.shape[0]:
            raise ArtifactSchemaError("mlp layer weight/bias shapes are inconsistent")
        if layers and layers[-1].bias.shape[0] != weight.shape[0]:
            raise ArtifactSchemaError("mlp layer widths do not chain")
        layers.append(MlpLayer(weight, bias))
    if not layers:
        raise ArtifactSchemaError("mlp needs at least one layer")
    if layers[0].weight.shape[0] != _N_FEATURES:
        raise ArtifactSchemaError(f"mlp input dim must be {_N_FEATURES}")
    if layers[-1].bias.shape[0] != 1:
        raise ArtifactSchemaError("mlp output layer must have exactly 1 unit")
    return MlpNetwork(layers=tuple(layers), activation=activation)


def _model_from_payload(
    payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> SuccessProbabilityModel:
    for key in ("schema_version", "artifact_version", "feature_names", "feature_means", "isotonic_x", "isotonic_y"):
        if key not in payload:
            raise ArtifactSchemaError(f"model payload missing key: {key!r}")

    if int(payload["schema_version"]) != FEATURE_SCHEMA_VERSION:
        raise ArtifactSchemaError(
            f"feature schema version mismatch: artifact={payload['schema_version']}, runtime={FEATURE_SCHEMA_VERSION}"
        )
    artifact_version = int(payload["artifact_version"])
    if artifact_version not in SUPPORTED_ARTIFACT_VERSIONS:
        raise ArtifactSchemaError(
            f"unsupported model artifact_version={artifact_version}; runtime supports {sorted(SUPPORTED_ARTIFACT_VERSIONS)}"
        )

    names = tuple(str(name) for name in payload["feature_names"])
    if names != FEATURE_NAMES:
        raise ArtifactSchemaError("feature_names in artifact does not match the locked v1 schema")

    means = np.asarray(payload["feature_means"], dtype=np.float64)
    if means.shape != (_N_FEATURES,):
        raise ArtifactSchemaError(f"feature_means must have shape ({_N_FEATURES},)")
    iso_x, iso_y = _require_isotonic(payload)

    # Older logistic artifacts predate model_family -> default to logistic.
    family = str(payload.get("model_family") or MODEL_FAMILY_LOGISTIC)

    if family == MODEL_FAMILY_GBT:
        return SuccessProbabilityModel(
            schema_version=FEATURE_SCHEMA_VERSION,
            artifact_version=artifact_version,
            model_family=family,
            feature_names=FEATURE_NAMES,
            feature_means=means,
            isotonic_x=iso_x,
            isotonic_y=iso_y,
            manifest=dict(manifest),
            gbt=_parse_gbt(payload),
        )

    if family == MODEL_FAMILY_MLP:
        if "feature_stds" not in payload:
            raise ArtifactSchemaError("mlp model payload missing key: 'feature_stds'")
        mlp_stds = np.asarray(payload["feature_stds"], dtype=np.float64)
        if mlp_stds.shape != (_N_FEATURES,):
            raise ArtifactSchemaError(f"mlp feature_stds must have shape ({_N_FEATURES},)")
        return SuccessProbabilityModel(
            schema_version=FEATURE_SCHEMA_VERSION,
            artifact_version=artifact_version,
            model_family=family,
            feature_names=FEATURE_NAMES,
            feature_means=means,
            isotonic_x=iso_x,
            isotonic_y=iso_y,
            manifest=dict(manifest),
            feature_stds=mlp_stds,
            mlp=_parse_mlp(payload),
        )

    if family == MODEL_FAMILY_LOGISTIC:
        for key in ("feature_stds", "logistic_coefficients", "logistic_intercept"):
            if key not in payload:
                raise ArtifactSchemaError(f"logistic model payload missing key: {key!r}")
        stds = np.asarray(payload["feature_stds"], dtype=np.float64)
        coefs = np.asarray(payload["logistic_coefficients"], dtype=np.float64)
        if stds.shape != (_N_FEATURES,) or coefs.shape != (_N_FEATURES,):
            raise ArtifactSchemaError(f"logistic feature_stds/coefficients must have shape ({_N_FEATURES},)")
        intercept = float(payload["logistic_intercept"])
        if not math.isfinite(intercept):
            raise ArtifactSchemaError("logistic_intercept must be finite")
        return SuccessProbabilityModel(
            schema_version=FEATURE_SCHEMA_VERSION,
            artifact_version=artifact_version,
            model_family=family,
            feature_names=FEATURE_NAMES,
            feature_means=means,
            isotonic_x=iso_x,
            isotonic_y=iso_y,
            manifest=dict(manifest),
            feature_stds=stds,
            logistic_coefficients=coefs,
            logistic_intercept=intercept,
        )

    raise ArtifactSchemaError(f"unknown model_family {family!r}")


# Re-exported for tests that want to confirm the predictor uses the SAME subscore extractors as the
# runtime scorers (i.e. no schema drift).
_RUNTIME_COMPONENT_EXTRACTORS = (
    geometric_score_components,
    stability_score_components,
    reachability_score_components,
    feasibility_score_components,
)


__all__ = [
    "GbtEnsemble",
    "GbtTree",
    "SuccessProbabilityModel",
    "load_success_probability_model",
    "predict_proba",
]
