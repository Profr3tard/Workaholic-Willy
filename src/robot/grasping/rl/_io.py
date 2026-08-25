"""
The single fail-closed JSONL loader for the RL stack,
raises a typed ValueError on a malformed or non-object line.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dict records"""
    path = Path(path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: line {line_no} is not valid JSON: {exc}"
                ) from exc
            if not isinstance(obj, dict):
                raise ValueError(
                    f"{path}: line {line_no} is not a JSON object"
                )
            records.append(obj)
    return records
