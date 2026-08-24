"""Model promotion gate for the success-probability artifact.

This module is the *offline* gatekeeper between a trained artifact and the
runtime lifecycle phases that grant the model behavioural influence
(``canary`` and ``active``). It does three things:

1. **Evaluate** an artifact against a frozen, deterministic synthetic
   validation slice (seed :data:`PROMOTION_VALIDATION_SEED`, distinct from the
   training and CV seeds).
2. **Emit** a structured :class:`PromotionReport` containing the bound
   thresholds, eval metrics, a SHA-256 attestation chain over the
   artifact bytes, and the evaluator's tooling version. The report is
   written to ``promotion.json`` next to ``model.json`` and
   ``manifest.json``.
3. **Verify** at runtime load time that an artifact carries a valid
   promotion report (``verdict == "pass"`` AND artifact bytes match the
   attestation AND thresholds are not weaker than the plan-locked
   limits).

Signing
-------
The :class:`PromotionReport` carries a ``signature`` field, ``"none"`` by default. The SHA-256
attestation chain is the trust root and is ALWAYS checked; the signature is an OPTIONAL layer on top:
:func:`build_promotion_report` accepts a ``signer`` and :func:`verify_promotion` a ``verifier``
(see :mod:`signing`). With no signer injected the signature stays ``"none"`` and no fake trust root is
implied.
"""

from __future__ import annotations

import getpass
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np

from src.robot.grasping.scoring.success_probability import (
    SuccessProbabilityModel,
    load_success_probability_model,
    predict_proba,
)
from src.robot.grasping.calibration.signing import (
    SIGNATURE_NONE,
    Signer,
    Verifier,
)
from src.robot.grasping.constants import (
    MODEL_PROMOTION_LOG_FILE,
    create_grasping_logger,
)

# Logging for the model promotion gate.
logger = create_grasping_logger("ModelPromotion", MODEL_PROMOTION_LOG_FILE)


__all__ = [
    "PROMOTION_FILENAME",
    "PROMOTION_SCHEMA_VERSION",
    "PROMOTION_TOOLING_VERSION",
    "PROMOTION_VALIDATION_ATTEMPTS",
    "PROMOTION_VALIDATION_SEED",
    "PromotionLoadError",
    "PromotionReport",
    "PromotionThresholds",
    "PromotionVerdict",
    "ValidationSlice",
    "build_promotion_report",
    "evaluate_for_promotion",
    "load_promotion_report",
    "promote_artifact",
    "verify_promotion",
    "write_promotion_report",
]


# Module-level constants (locked at ship bumping these requires a new
# promotion-report schema version + retraining the artifact).

#: Filename of the promotion attestation, written next to ``model.json``.
PROMOTION_FILENAME: str = "promotion.json"

#: Schema version for the on-disk attestation.
PROMOTION_SCHEMA_VERSION: int = 1

#: Tooling identifier embedded in every report so future evaluator versions stay distinguishable.
PROMOTION_TOOLING_VERSION: str = "promotion-v1"

#: Independent seed for the validation slice.
PROMOTION_VALIDATION_SEED: int = 20260519

#: Sample budget for the validation slice.
PROMOTION_VALIDATION_ATTEMPTS: int = 2000

#: Lifecycle phases that require a valid promotion.
_PROMOTED_LIFECYCLE_PHASES: frozenset[str] = frozenset({"canary", "active"})


@dataclass(frozen=True, slots=True)
class PromotionThresholds:
    """Plan-honest maxima for the promotion gate."""
    brier_max: float = 0.09
    log_loss_max: float = 0.31

    def to_dict(self) -> dict[str, float]:
        return {"brier_max": float(self.brier_max), "log_loss_max": float(self.log_loss_max)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PromotionThresholds":
        return cls(
            brier_max=float(payload["brier_max"]),
            log_loss_max=float(payload["log_loss_max"]),
        )


@dataclass(frozen=True, slots=True)
class ValidationSlice:
    """Describes the deterministic validation slice the gate used."""

    kind: str
    seed: int
    n_attempts: int
    dataset_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "seed": int(self.seed),
            "n_attempts": int(self.n_attempts),
            "dataset_sha256": self.dataset_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ValidationSlice":
        return cls(
            kind=str(payload["kind"]),
            seed=int(payload["seed"]),
            n_attempts=int(payload["n_attempts"]),
            dataset_sha256=str(payload["dataset_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class PromotionVerdict:
    """In-memory pass/fail verdict + the reasons that produced it."""

    verdict: str  # "pass" | "fail"
    metrics: Mapping[str, float]
    thresholds: PromotionThresholds
    reasons: tuple[str, ...] = ()

    def passed(self) -> bool:
        return self.verdict == "pass"


@dataclass(frozen=True, slots=True)
class PromotionReport:
    """On-disk attestation payload (mirrors ``promotion.json``)."""

    schema_version: int
    artifact_basename: str
    model_json_sha256: str
    manifest_json_sha256: str
    artifact_sha256: str
    validation: ValidationSlice
    metrics: Mapping[str, float]
    gate_thresholds: PromotionThresholds
    verdict: str
    promoted_at: str
    promoted_by: str
    tooling_version: str
    signature: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "artifact_basename": self.artifact_basename,
            "model_json_sha256": self.model_json_sha256,
            "manifest_json_sha256": self.manifest_json_sha256,
            "artifact_sha256": self.artifact_sha256,
            "validation": self.validation.to_dict(),
            "metrics": {k: float(v) for k, v in self.metrics.items()},
            "gate_thresholds": self.gate_thresholds.to_dict(),
            "verdict": self.verdict,
            "promoted_at": self.promoted_at,
            "promoted_by": self.promoted_by,
            "tooling_version": self.tooling_version,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PromotionReport":
        return cls(
            schema_version=int(payload["schema_version"]),
            artifact_basename=str(payload["artifact_basename"]),
            model_json_sha256=str(payload["model_json_sha256"]),
            manifest_json_sha256=str(payload["manifest_json_sha256"]),
            artifact_sha256=str(payload["artifact_sha256"]),
            validation=ValidationSlice.from_dict(payload["validation"]),
            metrics={k: float(v) for k, v in payload["metrics"].items()},
            gate_thresholds=PromotionThresholds.from_dict(payload["gate_thresholds"]),
            verdict=str(payload["verdict"]),
            promoted_at=str(payload["promoted_at"]),
            promoted_by=str(payload["promoted_by"]),
            tooling_version=str(payload["tooling_version"]),
            signature=str(payload.get("signature", "none")),
        )


class PromotionLoadError(ValueError):
    """
    Raised when ``promotion.json`` is missing, malformed, or schema-incompatible.
    NOTE: Runtime callers should catch this *and* ``OSError`` to remain fail-safe.
    """


def _sha256_of_file(path: Path) -> str:
    """Return the lowercase hex SHA-256 of a file's raw bytes."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifact_sha256(model_sha: str, manifest_sha: str) -> str:
    """Combine the two per-file SHAs into one deterministic chain SHA (order fixed: model, then manifest)."""
    return hashlib.sha256(
        (model_sha + ":" + manifest_sha).encode("ascii")
    ).hexdigest()


def _dataset_sha256(x: np.ndarray, y: np.ndarray) -> str:
    """Hash of the canonical (X, y) bytes used for the validation eval."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(x, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(y, dtype=np.float64).tobytes())
    return h.hexdigest()


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Stable JSON serialisation (sorted keys, trailing newline) matching the trainer."""
    return (
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _log_loss(y: np.ndarray, p: np.ndarray, *, eps: float = 1e-15) -> float:
    """Binary cross-entropy log-loss (pure numpy, clipped to avoid log(0))."""
    # Kept here rather than in the trainer so this gate does not change the
    # locked ``evaluate_metrics`` key set {brier, ece, auroc}.
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    if y.shape != p.shape:
        raise ValueError("shape mismatch between y and p")
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def evaluate_for_promotion(
    artifact_dir: str | Path,
    *,
    validation_seed: int = PROMOTION_VALIDATION_SEED,
    n_attempts: int = PROMOTION_VALIDATION_ATTEMPTS,
    thresholds: PromotionThresholds = PromotionThresholds(),
) -> tuple[PromotionVerdict, dict[str, float], ValidationSlice]:
    """Load an artifact, score it on the validation slice, and return the verdict + metrics."""
    from src.robot.grasping.calibration.success_model_calibration import (
        DatasetSpec,
        build_synthetic_dataset,
        evaluate_metrics,
    )

    spec = DatasetSpec(seed=validation_seed, n_attempts=n_attempts)
    x, y, _modes = build_synthetic_dataset(spec)
    dataset_sha = _dataset_sha256(x, y)

    model: SuccessProbabilityModel = load_success_probability_model(artifact_dir)
    p = predict_proba(model, x)

    base = evaluate_metrics(y, p)  # {brier, ece, auroc}
    metrics: dict[str, float] = {
        "brier": float(base["brier"]),
        "log_loss": _log_loss(y, p),
        "ece": float(base["ece"]),
        "auroc": float(base["auroc"]),
    }

    reasons: list[str] = []
    if metrics["brier"] > thresholds.brier_max:
        reasons.append(
            f"brier={metrics['brier']:.4f} exceeds max {thresholds.brier_max:.4f}"
        )
    if metrics["log_loss"] > thresholds.log_loss_max:
        reasons.append(
            f"log_loss={metrics['log_loss']:.4f} exceeds max "
            f"{thresholds.log_loss_max:.4f}"
        )
    verdict = "pass" if not reasons else "fail"

    logger.info(
        "Promotion eval: verdict=%s brier=%.4f log_loss=%.4f ece=%.4f auroc=%.4f "
        "(seed=%d n=%d, artifact=%s)",
        verdict,
        metrics["brier"],
        metrics["log_loss"],
        metrics["ece"],
        metrics["auroc"],
        validation_seed,
        n_attempts,
        Path(artifact_dir).name,
    )
    if reasons:
        logger.warning("Promotion gate refused: %s", "; ".join(reasons))

    slice_ = ValidationSlice(
        kind="synthetic_split",
        seed=validation_seed,
        n_attempts=n_attempts,
        dataset_sha256=dataset_sha,
    )
    return (
        PromotionVerdict(
            verdict=verdict,
            metrics=metrics,
            thresholds=thresholds,
            reasons=tuple(reasons),
        ),
        metrics,
        slice_,
    )


def _default_promoted_by() -> str:
    # ``getpass.getuser`` honours ``USER`` / ``LOGNAME`` and falls back to
    # the pwd database.
    try:
        return getpass.getuser() or "unknown"
    except Exception:  # pragma: no cover - environment-dependent
        return "unknown"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_promotion_report(
    artifact_dir: str | Path,
    verdict: PromotionVerdict,
    validation: ValidationSlice,
    *,
    promoted_at: str | None = None,
    promoted_by: str | None = None,
    signature: str = SIGNATURE_NONE,
    signer: Signer | None = None,
) -> PromotionReport:
    """
    Assemble a :class:`PromotionReport` from an evaluated artifact (``promoted_at``/``promoted_by``
    override UTC-now/OS-user so the report regenerates byte-deterministically;
    an injected ``signer`` signs the chain SHA, else the signature stays ``"none"``
    with the SHA-256 chain the sole trust root).
    """
    a = Path(artifact_dir)
    model_sha = _sha256_of_file(a / "model.json")
    manifest_sha = _sha256_of_file(a / "manifest.json")
    chain_sha = _artifact_sha256(model_sha, manifest_sha)
    effective_signature = (
        signer.sign(chain_sha.encode("utf-8")) if signer is not None else signature
    )

    return PromotionReport(
        schema_version=PROMOTION_SCHEMA_VERSION,
        artifact_basename=a.name,
        model_json_sha256=model_sha,
        manifest_json_sha256=manifest_sha,
        artifact_sha256=chain_sha,
        validation=validation,
        metrics=dict(verdict.metrics),
        gate_thresholds=verdict.thresholds,
        verdict=verdict.verdict,
        promoted_at=promoted_at if promoted_at is not None else _utc_now_iso(),
        promoted_by=promoted_by if promoted_by is not None else _default_promoted_by(),
        tooling_version=PROMOTION_TOOLING_VERSION,
        signature=effective_signature,
    )


def write_promotion_report(
    report: PromotionReport,
    artifact_dir: str | Path,
) -> Path:
    """Write ``promotion.json`` next to ``model.json`` (canonical bytes)."""
    target = Path(artifact_dir) / PROMOTION_FILENAME
    payload = _canonical_json_bytes(report.to_dict())
    target.write_bytes(payload)
    logger.info(
        "Wrote %s (%d bytes, verdict=%s, chain_sha=%s...)",
        target,
        len(payload),
        report.verdict,
        report.artifact_sha256[:12],
    )
    return target


def load_promotion_report(artifact_dir: str | Path) -> PromotionReport:
    """Read ``promotion.json`` from disk and validate the schema header (raises :class:`PromotionLoadError`)."""
    target = Path(artifact_dir) / PROMOTION_FILENAME
    if not target.is_file():
        raise PromotionLoadError(
            f"missing {PROMOTION_FILENAME} under {artifact_dir}"
        )
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PromotionLoadError(f"could not parse {target}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PromotionLoadError(f"{target} is not a JSON object")
    sv = payload.get("schema_version")
    if int(cast(Any, sv)) != PROMOTION_SCHEMA_VERSION:
        raise PromotionLoadError(
            f"unsupported promotion schema_version={sv!r}, "
            f"runtime expects {PROMOTION_SCHEMA_VERSION}"
        )
    try:
        return PromotionReport.from_dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise PromotionLoadError(f"malformed promotion payload: {exc}") from exc


def verify_promotion(
    artifact_dir: str | Path,
    *,
    thresholds: PromotionThresholds = PromotionThresholds(),
    verifier: Verifier | None = None,
    require_signature: bool = False,
) -> tuple[bool, list[str]]:
    """Verify the on-disk artifact is *promoted* and *untampered*; returns ``(ok, reasons)`` and never raises."""
    a = Path(artifact_dir)
    reasons: list[str] = []
    try:
        report = load_promotion_report(a)
    except (PromotionLoadError, OSError) as exc:
        # Returned, not raised so this is the only place it is ever visible.
        logger.error("Promotion verify failed to load %s: %s", a, exc)
        return False, [f"promotion_load_failed: {exc}"]

    if report.verdict != "pass":
        reasons.append(f"verdict={report.verdict!r} != 'pass'")

    if report.gate_thresholds.brier_max > thresholds.brier_max:
        reasons.append(
            f"recorded brier_max={report.gate_thresholds.brier_max} weaker "
            f"than plan-locked {thresholds.brier_max}"
        )
    if report.gate_thresholds.log_loss_max > thresholds.log_loss_max:
        reasons.append(
            f"recorded log_loss_max={report.gate_thresholds.log_loss_max} weaker "
            f"than plan-locked {thresholds.log_loss_max}"
        )

    # Tamper detection: per-file SHA + chain SHA.
    try:
        live_model = _sha256_of_file(a / "model.json")
        live_manifest = _sha256_of_file(a / "manifest.json")
    except OSError as exc:
        logger.error("Promotion verify could not read artifact files under %s: %s", a, exc)
        return False, [f"artifact_files_unreadable: {exc}"]
    if live_model != report.model_json_sha256:
        reasons.append("model.json SHA mismatch (artifact tampered or stale)")
    if live_manifest != report.manifest_json_sha256:
        reasons.append("manifest.json SHA mismatch (artifact tampered or stale)")
    if _artifact_sha256(live_model, live_manifest) != report.artifact_sha256:
        reasons.append("chained artifact_sha256 mismatch")

    # OPTIONAL signature layer on top of the SHA chain.
    if verifier is not None:
        if report.signature == SIGNATURE_NONE:
            if require_signature:
                reasons.append(
                    "signature_required_but_absent: report is unsigned (signature='none')"
                )
        elif not verifier.verify(
            report.artifact_sha256.encode("utf-8"), report.signature
        ):
            reasons.append("signature_invalid: chain-SHA signature failed verification")

    if reasons:
        logger.warning("Promotion verify REFUSED %s: %s", a.name, "; ".join(reasons))
    else:
        logger.info("Promotion verify passed for %s", a.name)
    return (len(reasons) == 0), reasons


def promote_artifact(
    artifact_dir: str | Path,
    *,
    promoted_at: str | None = None,
    promoted_by: str | None = None,
    thresholds: PromotionThresholds = PromotionThresholds(),
    validation_seed: int = PROMOTION_VALIDATION_SEED,
    n_attempts: int = PROMOTION_VALIDATION_ATTEMPTS,
) -> PromotionReport:
    """
    One-shot evaluate -> build -> write -> return; always writes ``promotion.json``
    (even on fail) for an audit trail.
    """
    verdict, _metrics, validation = evaluate_for_promotion(
        artifact_dir,
        validation_seed=validation_seed,
        n_attempts=n_attempts,
        thresholds=thresholds,
    )
    report = build_promotion_report(
        artifact_dir,
        verdict,
        validation,
        promoted_at=promoted_at,
        promoted_by=promoted_by,
    )
    write_promotion_report(report, artifact_dir)
    return report


def lifecycle_phase_requires_promotion(lifecycle_phase: str) -> bool:
    """Return ``True`` iff this lifecycle phase requires a valid promotion"""
    return lifecycle_phase in _PROMOTED_LIFECYCLE_PHASES
