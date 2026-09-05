"""Where did this config value come from: which file, which line, which profile layer?

The loader reads up to a dozen YAML files, deep-merges a chain of profile overlays onto each, and hands
back plain Python values. That merge is lossless about VALUES and total about ORIGIN: ``_deep_merge``
returns ``42`` with no memory of which of four files said ``42``. For a single-file config that costs
nothing. With ``WILLY_PROFILE=sim,ur3e,tiltcam`` it means the system cannot answer the two questions a
user actually has: *which layer set this?* and *why this number?* Neither can its error messages.

The measured symptom: a typo in ``robot.sim.yaml`` produced

    configuration failed schema validation under <data dir>:
    robot.sim.tcp_offset_typo  Extra inputs are not permitted

naming the *directory*, so the reader has to guess which of the layered files carries it, while
``loader.py``'s own module docstring promised "the message always names the offending file".

This module is that memory. It is a SIDE-CAR: a second, read-only walk over the same files the loader
walks, recording a line mark per leaf. It never participates in the merge, so a bug here cannot change a
single loaded value; the worst it can do is fail to explain one.

Deliberately stdlib-only (``yaml.compose()`` yields ``start_mark.line`` for every node, so no
``ruamel.yaml`` dependency at the bottom of the dependency stack), and it imports nothing from the robot
runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "Origin",
    "index_origins",
    "nearest_keys",
    "section_sources",
]


@dataclass(frozen=True, slots=True)
class Origin:
    """Where one leaf's winning value was written."""

    file: Path
    line: int          #: 1-based, ready to paste after a colon
    layer: str         #: "" for the base YAML, else the profile layer name
    key: str           #: the leaf's own name (not the dotted path)

    def location(self, relative_to: Path | None = None) -> str:
        """``path:line  [layer: x]``. Relative when a root is given, so the message stays readable."""
        path: Path | str = self.file
        if relative_to is not None:
            try:
                path = self.file.relative_to(relative_to)
            except ValueError:
                path = self.file
        suffix = f"  [layer: {self.layer}]" if self.layer else ""
        return f"{path}:{self.line}{suffix}"


def section_sources(root: Path) -> list[tuple[Path, str, str]]:
    """``(base_file, top_level_yaml_key, dotted AppConfig prefix)`` for every section the loader reads.

    This mirrors ``loader._load_cached`` on purpose: a file it does not read cannot explain anything, and
    a file it reads that is missing here would leave a key unexplained. Kept in one obvious table so the
    two can be compared by eye; a test asserts every indexed key resolves to a real loaded value.
    """
    return [
        (root / "camera" / "cam.yaml", "cameras", "camera.cameras"),
        (root / "camera" / "stereomatcher.yaml", "stereomatcher", "camera.stereomatcher"),
        (root / "camera" / "hand_eye.yaml", "hand_eye", "camera.hand_eye"),
        (root / "robot" / "robot.yaml", "robot", "robot"),
        (root / "app" / "runtime.yaml", "runtime", "runtime"),
    ]


def index_origins(root: Path, layers: tuple[str, ...] = ()) -> dict[str, Origin]:
    """Map each dotted config key to the file/line/layer whose value WINS.

    Layers are applied in the loader's order (base first, then each profile layer left to right), so a
    later write simply overwrites the earlier record: the same precedence, tracked instead of forgotten.
    Files that do not exist are skipped silently: an absent overlay is the normal case, not an error.
    """
    origins: dict[str, Origin] = {}
    for base, top_key, prefix in section_sources(root):
        for layer in ("", *layers):
            path = base if not layer else base.with_name(f"{base.stem}.{layer}{base.suffix}")
            _index_file(path, top_key, prefix, layer, origins)
    # models/*.yaml contribute their top-level keys directly under `models`, and a profile may add a
    # file with no base counterpart (the loader allows that), so the whole directory is walked.
    models_dir = root / "models"
    if models_dir.is_dir():
        # Base files FIRST, then each layer in chain order. Walking the directory alphabetically instead
        # would let `object.yaml` overwrite `object.sim.yaml` (s < y) and report the base as the winner
        # for a value the sim layer actually set, the exact class of wrong answer this module exists to
        # prevent, so the ordering is asserted by a test.
        for layer in ("", *layers):
            for path in sorted(models_dir.glob("*.yaml")):
                stem_parts = path.stem.split(".")
                file_layer = stem_parts[-1] if len(stem_parts) > 1 else ""
                if file_layer != layer:
                    continue
                _index_file(path, None, "models", layer, origins)
    return origins


def _index_file(
    path: Path, top_key: str | None, prefix: str, layer: str, out: dict[str, Origin]
) -> None:
    """Record a line mark for every leaf in ``path``. Unreadable/!mapping files are skipped, not raised.

    Skipping rather than raising is deliberate: this runs while the loader is ALREADY reporting an
    error, and an explainer that throws on the way to explaining is worse than one that says less.
    """
    try:
        node = yaml.compose(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 (explaining must never become a second failure)
        return
    if node is None:
        return
    if top_key is not None:
        node = _child(node, top_key)
        if node is None:
            return
    _walk(node, prefix, path, layer, out)


def _child(node: Any, key: str) -> Any:
    if not isinstance(node, yaml.MappingNode):
        return None
    for k, v in node.value:
        if getattr(k, "value", None) == key:
            return v
    return None


def _walk(node: Any, prefix: str, path: Path, layer: str, out: dict[str, Origin]) -> None:
    if isinstance(node, yaml.MappingNode):
        for k, v in node.value:
            name = str(getattr(k, "value", ""))
            dotted = f"{prefix}.{name}" if prefix else name
            # Record the KEY line for containers too: an `extra_forbidden` on a nested block should
            # point at the block, not vanish because the block is not a leaf.
            out[dotted] = Origin(path, int(k.start_mark.line) + 1, layer, name)
            _walk(v, dotted, path, layer, out)
    elif isinstance(node, yaml.SequenceNode):
        for i, item in enumerate(node.value):
            _walk(item, f"{prefix}[{i}]", path, layer, out)


def index_chains(root: Path, layers: tuple[str, ...] = ()) -> dict[str, list[Origin]]:
    """Every write of every key, in loader order, not just the one that won.

    The winner alone answers "where is this set"; the CHAIN answers "what did I override, and was my
    layer even reached", which is the question someone debugging a profile stack actually has. Same walk
    as :func:`index_origins`, appending instead of overwriting, so the last entry is always the winner.
    """
    chains: dict[str, list[Origin]] = {}
    single: dict[str, Origin] = {}
    for base, top_key, prefix in section_sources(root):
        for layer in ("", *layers):
            path = base if not layer else base.with_name(f"{base.stem}.{layer}{base.suffix}")
            single.clear()
            _index_file(path, top_key, prefix, layer, single)
            for key, origin in single.items():
                chains.setdefault(key, []).append(origin)
    models_dir = root / "models"
    if models_dir.is_dir():
        for layer in ("", *layers):
            for path in sorted(models_dir.glob("*.yaml")):
                stem_parts = path.stem.split(".")
                if (stem_parts[-1] if len(stem_parts) > 1 else "") != layer:
                    continue
                single.clear()
                _index_file(path, None, "models", layer, single)
                for key, origin in single.items():
                    chains.setdefault(key, []).append(origin)
    return chains


def comment_above(origin: Origin) -> str:
    """The contiguous ``#`` block written immediately above ``origin``'s line, dedented.

    This is the whole reason the explainer is worth building. The YAML comments are not decoration:
    they carry the measured WHY (*"MEASURED by sweeping the real planner: 6/6 up to 6 mm and 0/6 at
    10 mm"*), and the loader throws every one of them away at ``yaml.safe_load``. Harvesting the block
    at query time surfaces that evidence at the moment someone is confused about the value, and does it
    WITHOUT moving a single character: the comment stays where its author put it.
    """
    try:
        lines = origin.file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    out: list[str] = []
    i = origin.line - 2  # the line above the key (origin.line is 1-based)
    while i >= 0:
        stripped = lines[i].strip()
        if not stripped.startswith("#"):
            break
        out.append(stripped.lstrip("#").strip())
        i -= 1
    return "\n".join(reversed(out))


def read_value(origin: Origin) -> str:
    """The raw text written after the key on ``origin``'s line (``""`` for a block header)."""
    try:
        line = origin.file.read_text(encoding="utf-8").splitlines()[origin.line - 1]
    except (OSError, IndexError):
        return ""
    _, _, rest = line.partition(":")
    return rest.split("#")[0].strip()


def nearest_keys(name: str, candidates: list[str], *, limit: int = 3) -> list[str]:
    """Closest ``candidates`` to ``name`` for a did-you-mean hint (stdlib difflib, no dependency).

    Worth having because ``extra='forbid'`` turns every typo into a hard stop, and the most common typo
    is a near-miss of a real field: the reader knows what they meant and only needs the spelling.
    """
    import difflib

    return difflib.get_close_matches(name, candidates, n=limit, cutoff=0.6)
