"""Production uncertainty calibration tool.

Given a JSONL replay of records carrying the seven typed signal channels plus a
``label`` in ``[0, 1]`` (1 ⇒ success), fits a deterministic, monotone-preserving
:class:`UncertaintyCalibration`: per-channel quantile bins -> mean label -> isotonic
(Pool-Adjacent-Violators) map on the ``[0, 1]`` domain the runtime fusion expects.
Channel weights are not learned here; they stay at the ``UncertaintyWeights()``
defaults. Fully deterministic no PRNG, stable sort order, pure-Python floats.

CLI:

    python -m src.robot.grasping.calibration.uncertainty_calibration \\
        --replay tests/data/uncertainty_replay.jsonl \\
        --out artifact.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from src.robot.grasping.constants import (
    UNCERTAINTY_CALIBRATION_LOG_FILE,
    create_grasping_logger,
)
from src.robot.grasping.uncertainty import (
    UncertaintyCalibration,
    UncertaintyChannel,
    UncertaintyMonotoneMap,
    UncertaintyWeights,
)

# Logging for this module.
logger = create_grasping_logger(
    "UncertaintyCalibration", UNCERTAINTY_CALIBRATION_LOG_FILE
)

__all__ = ["fit_uncertainty_calibration", "main"]


_NUM_BINS = 5


def _pav_non_decreasing(values: Sequence[float]) -> list[float]:
    """Pool-Adjacent-Violators: a non-decreasing sequence of the same length (deterministic)."""
    blocks: list[tuple[float, int]] = [(float(v), 1) for v in values]
    i = 1
    while i < len(blocks):
        if blocks[i][0] < blocks[i - 1][0]:
            (v0, n0), (v1, n1) = blocks[i - 1], blocks[i]
            blocks[i - 1] = ((v0 * n0 + v1 * n1) / (n0 + n1), n0 + n1)
            del blocks[i]
            i = max(1, i - 1)
        else:
            i += 1
    expanded: list[float] = []
    for value, count in blocks:
        expanded.extend([value] * count)
    return expanded


def _fit_channel_map(
    samples: list[tuple[float, float]],
) -> UncertaintyMonotoneMap:
    """Fit a monotone PWL map ``channel_value -> label_mean`` (quantile-bin + PAV)."""

    # Stable sort keeps ties deterministic.
    samples = sorted(samples, key=lambda p: p[0])
    n = len(samples)
    if n == 0:
        return UncertaintyMonotoneMap.identity()
    # Build bins of equal record count.
    bin_size = max(1, n // _NUM_BINS)
    bins: list[list[tuple[float, float]]] = []
    i = 0
    while i < n:
        bins.append(samples[i : i + bin_size])
        i += bin_size
    # If the last bin is small, merge into previous.
    if len(bins) > 1 and len(bins[-1]) < bin_size // 2:
        bins[-2].extend(bins[-1])
        bins.pop()
    # Per-bin breakpoint (mean x) and value (mean label).
    bp_raw = [
        sum(x for x, _ in b) / len(b) for b in bins
    ]
    vs_raw = [
        sum(y for _, y in b) / len(b) for b in bins
    ]
    vs_mono = _pav_non_decreasing(vs_raw)
    # Force breakpoints strictly increasing by collapsing duplicates.
    bp: list[float] = []
    vs: list[float] = []
    for x, y in zip(bp_raw, vs_mono):
        if bp and x <= bp[-1]:
            bp[-1] = (bp[-1] + x) / 2.0
            vs[-1] = max(vs[-1], y)
        else:
            bp.append(x)
            vs.append(y)
    # Anchor endpoints at 0 and 1 so the map covers ``[0, 1]``.
    if bp[0] > 0.0:
        bp.insert(0, 0.0)
        vs.insert(0, vs[0])
    if bp[-1] < 1.0:
        bp.append(1.0)
        vs.append(vs[-1])
    vs = [min(1.0, max(0.0, v)) for v in vs]
    if len(bp) < 2:
        return UncertaintyMonotoneMap.identity()
    return UncertaintyMonotoneMap(
        breakpoints=tuple(bp), values=tuple(vs),
    )


def fit_uncertainty_calibration(
    records: Iterable[Mapping[str, object]],
    *,
    calibration_id: str | None = None,
) -> UncertaintyCalibration:
    """Fit a monotone calibration from a deterministic record stream (records need a ``label`` in ``[0, 1]``; unlabelled ones are skipped)."""

    records = list(records)
    per_channel: dict[UncertaintyChannel, list[tuple[float, float]]] = {
        ch: [] for ch in UncertaintyChannel
    }
    for rec in records:
        label = rec.get("label")
        if label is None:
            continue
        try:
            lab = float(label)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if not 0.0 <= lab <= 1.0:
            continue
        for ch in UncertaintyChannel:
            raw = rec.get(ch.value)
            if raw is None:
                continue
            try:
                xv = float(raw)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            if not 0.0 <= xv <= 1.0:
                continue
            per_channel[ch].append((xv, lab))
    maps: dict[UncertaintyChannel, UncertaintyMonotoneMap] = {}
    for ch, samples in per_channel.items():
        maps[ch] = _fit_channel_map(samples)
    # One aggregated line, not one per channel inside the fit loop.
    logger.info(
        "Fitted uncertainty calibration id=%s from %d records; usable samples per channel: %s",
        calibration_id,
        len(records),
        ", ".join(f"{ch.value}={len(s)}" for ch, s in per_channel.items()),
    )
    empty = [ch.value for ch, s in per_channel.items() if not s]
    if empty:
        logger.warning(
            "Channels with no usable sample fell back to the identity map: %s",
            ", ".join(empty),
        )
    return UncertaintyCalibration(
        weights=UncertaintyWeights(),
        maps=maps,
        calibration_id=calibration_id,
    )


def _load_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.robot.grasping.calibration.uncertainty_calibration",
        description="Fit a monotone uncertainty calibration from a JSONL replay.",
    )
    parser.add_argument("--replay", required=True, help="Path to JSONL replay records.")
    parser.add_argument("--out", required=True, help="Path to write the JSON artifact.")
    parser.add_argument(
        "--calibration-id", default=None,
        help="Optional opaque calibration identifier embedded in the artifact.",
    )
    args = parser.parse_args(argv)
    replay_path = Path(args.replay)
    if not replay_path.exists():
        logger.error("Replay file not found: %s", replay_path)
        print(f"replay file not found: {replay_path}", file=sys.stderr)
        return 2
    records = _load_jsonl(replay_path)
    # If records carry no labels (e.g. the synthetic fixture), inject
    # a deterministic label rule so the CLI can still produce a
    # well-formed artifact.
    if not any("label" in r for r in records):
        logger.warning(
            "No record in %s carries a 'label'; falling back to the deterministic "
            "feasibility_margin>0.5 label rule (fixture path, not a real calibration)",
            replay_path,
        )
        for r in records:
            fm = r.get("feasibility_margin") or 0.0
            r["label"] = 1.0 if fm > 0.5 else 0.0
    cal = fit_uncertainty_calibration(records, calibration_id=args.calibration_id)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(cal.to_artifact(), indent=2, sort_keys=True)
    out_path.write_text(payload)
    logger.info("Wrote calibration artifact %s (%d bytes)", out_path, len(payload))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
