"""RT-DETR fine-tuning: train the closed-set detector on a COCO-format dataset.

Given an ``images/`` directory + a COCO ``annotations.json`` (the standard detection format:
``images`` + ``annotations`` (``bbox`` in ``[x,y,w,h]``, ``category_id``) + ``categories``), this
fine-tunes the HF RT-DETR model and exports a ``save_pretrained()`` checkpoint that
:class:`~src.models.detection.closed_set.detector.RtDetrObjectDetector` loads via
``model_path=<out> , local=True`` -- so a trained model drops straight into the perception stack.

The **class vocabulary comes from the dataset's ``categories``**: the model's classification head is
re-initialised for that class set (``ignore_mismatched_sizes``), so you can train arbitrary classes,
not just COCO.

CLI (mirrors ``success_model_calibration``)::

    python -m src.models.detection.closed_set.train train \
        --data-dir data/detect/v1 --output-dir assets/models/rtdetr/v1 --epochs 20
    python -m src.models.detection.closed_set.train eval  \
        --model-dir assets/models/rtdetr/v1 --data-dir data/detect/v1

"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timezone
import hashlib
import json
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.models.constants import MODELS_LOG_DIR, RTDETR_TRAIN_LOG_FILE
from src.utility.log_cfg import create_logger


_LOG = create_logger("RtDetrTrain", log_file=RTDETR_TRAIN_LOG_FILE, log_dir=MODELS_LOG_DIR)

DEFAULT_BASE_MODEL = "PekingU/rtdetr_r50vd"
DEFAULT_OUTPUT_DIR = "assets/models/rtdetr/v1"
DEFAULT_DATA_DIR = "data/detect/v1"
DEFAULT_SEED = 20260716
MANIFEST_SCHEMA = "willy.rtdetr.train_manifest/1"


# --------------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Fine-tuning hyper-parameters (deterministic defaults; override on the CLI)."""

    base_model_id: str = DEFAULT_BASE_MODEL
    epochs: int = 20
    learning_rate: float = 1e-4
    batch_size: int = 4
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.05
    image_size: int = 640
    seed: int = DEFAULT_SEED
    fp16: bool = True  # honoured only on CUDA; the trainer downgrades on CPU
    grad_accum_steps: int = 1
    num_workers: int = 2

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------------------------------
# COCO parsing
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CocoImageRecord:
    """One image + its annotations, in the shape the HF image processor's ``annotations`` arg wants."""

    image_id: int
    file_name: str
    coco_annotations: list[dict]  # each: {bbox:[x,y,w,h], category_id, area, iscrowd, id}


@dataclass(frozen=True, slots=True)
class CocoIndex:
    """A parsed COCO ``annotations.json``: category map + per-image records."""

    categories: dict[int, str]  # category_id -> name (sorted by id)
    records: list[CocoImageRecord]
    source_path: str
    num_annotations: int = field(default=0)

    def image_records(self) -> list[CocoImageRecord]:
        return self.records


def load_coco_index(annotations_path: str | Path) -> CocoIndex:
    """Parse a COCO detection ``annotations.json`` into a :class:`CocoIndex`."""
    p = Path(annotations_path)
    if not p.is_file():
        raise FileNotFoundError(f"COCO annotations.json not found: {p}")
    doc = json.loads(p.read_text(encoding="utf-8"))
    for key in ("images", "annotations", "categories"):
        if key not in doc:
            raise ValueError(f"COCO file {p} missing required key {key!r}")
    categories = {int(c["id"]): str(c["name"]) for c in doc["categories"]}
    if not categories:
        raise ValueError(f"COCO file {p} declares no categories")
    categories = dict(sorted(categories.items()))
    by_image: dict[int, list[dict]] = {int(im["id"]): [] for im in doc["images"]}
    for ann in doc["annotations"]:
        iid = int(ann["image_id"])
        if iid not in by_image:  # an annotation for an image the file doesn't declare -> skip, don't crash
            continue
        bbox = [float(v) for v in ann["bbox"]]
        by_image[iid].append({
            "bbox": bbox,
            "category_id": int(ann["category_id"]),
            "area": float(ann.get("area", bbox[2] * bbox[3])),
            "iscrowd": int(ann.get("iscrowd", 0)),
            "id": int(ann.get("id", 0)),
        })
    records = [
        CocoImageRecord(image_id=int(im["id"]), file_name=str(im["file_name"]),
                        coco_annotations=by_image.get(int(im["id"]), []))
        for im in doc["images"]
    ]
    return CocoIndex(categories=categories, records=records, source_path=str(p),
                     num_annotations=len(doc["annotations"]))


def build_id2label(index: CocoIndex) -> dict[int, str]:
    """The model ``id2label`` (CONTIGUOUS 0..N-1) + the category-id -> model-id remap it implies."""
    return {i: name for i, (_cid, name) in enumerate(index.categories.items())}


def _category_remap(index: CocoIndex) -> dict[int, int]:
    """COCO ``category_id`` -> contiguous model label id (0..N-1), by sorted category id."""
    return {cid: i for i, (cid, _name) in enumerate(index.categories.items())}


# --------------------------------------------------------------------------------------------------
# Manifest (canonical JSON + sha256, mirrors the calibration trainers)
# --------------------------------------------------------------------------------------------------
def _canonical_json_bytes(obj: object) -> bytes:
    return (json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def _env_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for mod in ("torch", "transformers", "accelerate"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 - a missing optional dep is reported as absent, not fatal
            versions[mod] = "absent"
    return versions


def build_manifest(
    *, config: TrainConfig, index: CocoIndex, output_dir: str, metrics: dict, trained_at: str,
) -> dict:
    """The training manifest: dataset fingerprint + class map + config + metrics + env versions."""
    return {
        "schema": MANIFEST_SCHEMA,
        "base_model_id": config.base_model_id,
        "output_dir": output_dir,
        "trained_at": trained_at,
        "dataset": {
            "annotations_path": index.source_path,
            "annotations_sha256": _sha256_file(index.source_path),
            "num_images": len(index.records),
            "num_annotations": index.num_annotations,
            "id2label": {str(k): v for k, v in build_id2label(index).items()},
        },
        "train_config": config.to_dict(),
        "metrics": metrics,
        "env": _env_versions(),
        "runtime_predictor": "src.models.detection.closed_set.detector.RtDetrObjectDetector",
    }


def write_manifest(manifest: dict, output_dir: str | Path) -> Path:
    out = Path(output_dir) / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json_bytes(manifest)
    out.write_bytes(payload)
    _LOG.info("manifest written: %s (%d bytes)", out, len(payload))
    return out


# --------------------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------------------
def _resolve_split(data_dir: str | Path, split: str) -> tuple[Path, CocoIndex] | None:
    """``<data_dir>/<split>/{images/, annotations.json}`` -> (images_dir, index); None if absent."""
    root = Path(data_dir) / split
    ann = root / "annotations.json"
    if not ann.is_file():
        return None
    return root / "images", load_coco_index(ann)


def train_rtdetr(*, data_dir: str, output_dir: str, config: TrainConfig) -> dict:
    """Fine-tune RT-DETR on ``<data_dir>/train`` (+ optional ``/val``), export to ``output_dir``.

    Returns the written manifest.
    """
    started = time.perf_counter()
    # Showing the config + the data split is the only way to know what was actually trained, so log it before
    # the training stack is imported (it is slow to import and may downgrade the log level).
    _LOG.info(
        "training RT-DETR from %s: data=%s out=%s epochs=%d batch=%d lr=%g image_size=%d seed=%d",
        config.base_model_id, data_dir, output_dir, config.epochs, config.batch_size,
        config.learning_rate, config.image_size, config.seed,
    )
    train_split = _resolve_split(data_dir, "train")
    if train_split is None:
        raise FileNotFoundError(f"no train split at {Path(data_dir) / 'train' / 'annotations.json'}")
    train_images, train_index = train_split
    _LOG.info(
        "train split: %d image(s), %d annotation(s), %d class(es) from %s",
        len(train_index.records), train_index.num_annotations, len(train_index.categories),
        train_index.source_path,
    )

    set_seed(config.seed)
    id2label = build_id2label(train_index)
    label2id = {v: k for k, v in id2label.items()}
    remap = _category_remap(train_index)

    processor = AutoImageProcessor.from_pretrained(
        config.base_model_id, do_resize=True, size={"height": config.image_size, "width": config.image_size},
    )
    model = AutoModelForObjectDetection.from_pretrained(
        config.base_model_id, id2label=id2label, label2id=label2id, num_labels=len(id2label),
        ignore_mismatched_sizes=True,  # re-init the class head for THIS dataset's classes
    )

    class _CocoDS(torch.utils.data.Dataset):
        def __init__(self, images_dir: Path, index: CocoIndex) -> None:
            self._images_dir = images_dir
            self._recs = index.image_records()

        def __len__(self) -> int:
            return len(self._recs)

        def __getitem__(self, i: int) -> dict:
            rec = self._recs[i]
            img = Image.open(self._images_dir / rec.file_name).convert("RGB")
            anns = [{**a, "category_id": remap[a["category_id"]]} for a in rec.coco_annotations
                    if a["category_id"] in remap]
            target = {"image_id": rec.image_id, "annotations": anns}
            enc = processor(images=img, annotations=target, return_tensors="pt")
            return {"pixel_values": enc["pixel_values"][0], "labels": enc["labels"][0]}

    def _collate(batch: list[dict]) -> dict:
        return {
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "labels": [b["labels"] for b in batch],
        }

    val_split = _resolve_split(data_dir, "val")
    train_ds = _CocoDS(train_images, train_index)
    eval_ds = _CocoDS(val_split[0], val_split[1]) if val_split is not None else None
    if val_split is None:
        _LOG.warning("no val split under %s training without evaluation", data_dir)
    else:
        _LOG.info("val split: %d image(s) from %s", len(val_split[1].records), val_split[1].source_path)
    if config.fp16 and not torch.cuda.is_available():
        # The downgrade is silent inside TrainingArguments
        _LOG.warning("fp16 requested but no CUDA device available: training in fp32")

    args = TrainingArguments(
        output_dir=str(Path(output_dir) / "_hf_trainer"),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        fp16=config.fp16 and torch.cuda.is_available(),
        dataloader_num_workers=config.num_workers,
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=10,
        seed=config.seed,
        remove_unused_columns=False,  # RT-DETR consumes 'labels' directly
        report_to=[],
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                      data_collator=_collate)
    train_out = trainer.train()
    _LOG.info(
        "training finished: final_train_loss=%.6f after %d epoch(s) in %.1f s",
        float(train_out.training_loss), config.epochs, time.perf_counter() - started,
    )

    # Export the fine-tuned model + processor as an HF checkpoint the detector loads verbatim.
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    _LOG.info("checkpoint exported to %s (model + processor)", Path(output_dir).resolve())

    metrics: dict = {"final_train_loss": float(train_out.training_loss)}
    if eval_ds is not None:
        metrics["final_eval_loss"] = float(trainer.evaluate().get("eval_loss", float("nan")))
        _LOG.info("final_eval_loss=%.6f", metrics["final_eval_loss"])
    manifest = build_manifest(
        config=config, index=train_index, output_dir=output_dir, metrics=metrics,
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    write_manifest(manifest, output_dir)
    return manifest


def evaluate_rtdetr(*, model_dir: str, data_dir: str, batch_size: int = 4) -> dict:
    """Evaluate a trained checkpoint on ``<data_dir>/val`` reports the eval loss.

    (COCO mAP would need ``pycocotools``/``torchmetrics`` -- kept optional so eval runs with just the
    training stack; wire it here when a labelled val set exists.)
    """

    started = time.perf_counter()
    val = _resolve_split(data_dir, "val")
    if val is None:
        # Falling back to the TRAIN split makes the number meaningless as a generalisation measure
        # it is data the checkpoint was fitted on.
        _LOG.warning("no val split under %s evaluating on the TRAIN split (not a held-out score)", data_dir)
        val = _resolve_split(data_dir, "train")
    if val is None:
        raise FileNotFoundError(f"no val (or train) split under {data_dir}")
    images_dir, index = val
    _LOG.info("evaluating %s on %d image(s) from %s", model_dir, len(index.records), index.source_path)
    remap = _category_remap(index)
    processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForObjectDetection.from_pretrained(model_dir, local_files_only=True)

    class _DS(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(index.records)

        def __getitem__(self, i: int) -> dict:
            rec = index.records[i]
            img = Image.open(images_dir / rec.file_name).convert("RGB")
            anns = [{**a, "category_id": remap[a["category_id"]]} for a in rec.coco_annotations
                    if a["category_id"] in remap]
            enc = processor(images=img, annotations={"image_id": rec.image_id, "annotations": anns},
                            return_tensors="pt")
            return {"pixel_values": enc["pixel_values"][0], "labels": enc["labels"][0]}

    def _collate(batch: list[dict]) -> dict:
        return {"pixel_values": torch.stack([b["pixel_values"] for b in batch]),
                "labels": [b["labels"] for b in batch]}

    args = TrainingArguments(output_dir=str(Path(model_dir) / "_eval"), per_device_eval_batch_size=batch_size,
                             remove_unused_columns=False, report_to=[])
    trainer = Trainer(model=model, args=args, data_collator=_collate)
    out = trainer.evaluate(eval_dataset=_DS())
    result = {"eval_loss": float(out.get("eval_loss", float("nan"))), "num_images": len(index.records)}
    _LOG.info(
        "eval finished: eval_loss=%.6f over %d image(s) in %.1f s",
        result["eval_loss"], int(result["num_images"]), time.perf_counter() - started,
    )
    return result


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------
def _cmd_train(args: argparse.Namespace) -> int:
    config = TrainConfig(
        base_model_id=args.base_model, epochs=args.epochs, learning_rate=args.lr,
        batch_size=args.batch_size, image_size=args.image_size, seed=args.seed,
    )
    try:
        manifest = train_rtdetr(data_dir=args.data_dir, output_dir=args.output_dir, config=config)
    except FileNotFoundError as exc:
        _LOG.error("train refused: %s", exc)
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - a training failure is exit 3 with the reason on stderr
        # ERROR, not `raise`: the CLI turns this into exit 3, so the traceback would otherwise be lost
        # and a run that died three hours in is exactly the one you come back to the log for.
        _LOG.exception("training failed: %s", exc)
        print(f"training failed: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(manifest, sort_keys=True, indent=2))
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    try:
        result = evaluate_rtdetr(model_dir=args.model_dir, data_dir=args.data_dir, batch_size=args.batch_size)
    except FileNotFoundError as exc:
        _LOG.error("eval refused: %s", exc)
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    """Parse a dataset split + report its class map / sizes no training stack needed."""
    split = _resolve_split(args.data_dir, args.split)
    if split is None:
        _LOG.error("inspect refused: no %s split under %s", args.split, args.data_dir)
        print(f"error: no {args.split} split under {args.data_dir}", file=sys.stderr)
        return 2
    _images, index = split
    report = {
        "split": args.split,
        "num_images": len(index.records),
        "num_annotations": index.num_annotations,
        "id2label": {str(k): v for k, v in build_id2label(index).items()},
    }
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.models.detection.closed_set.train",
        description="Fine-tune / evaluate the RT-DETR closed-set object detector on a COCO dataset.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    tr = sub.add_parser("train", help="Fine-tune RT-DETR + export an HF checkpoint + manifest.")
    tr.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="dir with train/ (+ optional val/) COCO splits")
    tr.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="where save_pretrained() + manifest.json go")
    tr.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    tr.add_argument("--epochs", type=int, default=20)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--batch-size", type=int, default=4)
    tr.add_argument("--image-size", type=int, default=640)
    tr.add_argument("--seed", type=int, default=DEFAULT_SEED)
    tr.set_defaults(func=_cmd_train)

    ev = sub.add_parser("eval", help="Evaluate a trained checkpoint's val loss.")
    ev.add_argument("--model-dir", default=DEFAULT_OUTPUT_DIR)
    ev.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ev.add_argument("--batch-size", type=int, default=4)
    ev.set_defaults(func=_cmd_eval)

    ins = sub.add_parser("inspect", help="Parse a split + print its class map / sizes (no training stack).")
    ins.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ins.add_argument("--split", default="train")
    ins.set_defaults(func=_cmd_inspect)

    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())