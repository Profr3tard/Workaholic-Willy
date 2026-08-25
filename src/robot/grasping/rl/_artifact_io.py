"""Shared artifact IO the single neutral source for writing and
hashing serialised policy/report artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def hash_artifact(path: str | Path) -> str:
    """Return the sha256 hex digest of a serialised policy artifact."""

    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_artifact_json(artifact: Mapping[str, Any], path: str | Path) -> str:
    """Write ``artifact`` as canonical JSON and return the sha256 hex of the bytes."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(dict(artifact), sort_keys=True, indent=2) + "\n").encode("utf-8")
    p.write_bytes(data)
    return hashlib.sha256(data).hexdigest()
