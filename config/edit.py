"""Writing a measured value back into the config tree, without destroying the file it lands in.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from ._provenance import section_sources
from ._schema_index import schema_index

__all__ = [
    "WRITABLE",
    "WriteRefused",
    "WriteResult",
    "Writable",
    "set_key",
    "set_keys",
    "target_file",
    "writable",
]


class WriteRefused(StrEnum):
    """Why a write did not happen. Typed, because the console renders a different form for each."""

    #: The key is real but deliberately not writable from a tool. See the module docstring.
    NOT_WRITABLE = "not_writable"
    #: The schema does not accept the key at all.
    UNKNOWN_KEY = "unknown_key"
    #: The value was written, the loader rejected the result, the file was restored.
    INVALID_VALUE = "invalid_value"
    #: The file the value would land in does not exist, or the key's section is not file-backed.
    NO_TARGET = "no_target"
    #: The key decides WHICH machine moves, and something is currently connected to one.
    CELL_CONNECTED = "cell_connected"


@dataclass(frozen=True, slots=True)
class Writable:
    """One key an operator may set, with the sentence that tells them what they are measuring."""

    path: str
    #: What to call it in a form.
    label: str
    #: What the operator physically does to obtain the number. The form shows this, not a type.
    measure: str
    unit: str = ""
    #: Only offered for this ``robot.vendor``. Empty means every cell. Two entries exist purely
    #: because the same fact -- the controller's address -- lives under a different key per vendor, and
    #: showing an operator both would be showing them one field that does nothing.
    vendor: str = ""
    #: True for values that decide WHICH machine moves or how it is driven, as opposed to what the tool
    #: weighs. Writable, but only with nothing connected -- see :func:`set_keys`' ``connected`` guard.
    requires_disconnected: bool = False


#: The complete writable set. Adding to it is a deliberate act: the reason each of these is here is that
#: no API can answer it and the cell blocks until a human does.
WRITABLE: tuple[Writable, ...] = (
    Writable(
        path="robot.safety.payload.mass_kg",
        label="Payload mass",
        measure=(
            "Weigh the WHOLE assembly on a bench scale: gripper plus coupling/adapter plate, every "
            "cable and hose that rides on the wrist, and any workpiece the arm carries. Not the "
            "gripper's datasheet mass."
        ),
        unit="kg",
    ),
    Writable(
        path="robot.safety.payload.cog_mm",
        label="Payload centre of gravity",
        measure=(
            "From the flange face, in the same session as the mass. Leaving it at [0,0,0] declares a "
            "multi-kilogram tool to be a point mass at the flange at 3 kg and 132 mm that is "
            "3.88 N*m of wrist torque the controller does not model."
        ),
        unit="mm",
    ),
    Writable(
        path="robot.gripper.tool_frame.source",
        label="Tool frame owner",
        measure=(
            "'willy' if the controller runs a bare flange and this driver composes the offset; "
            "'polyscope' if an operator set the TCP on the pendant and the driver only verifies it. "
            "Either way connect() derives what the controller is actually running and refuses a "
            "mismatch."
        ),
    ),
    Writable(
        path="robot.gripper.tool_frame.offset_mm",
        label="Flange -> grasp centre offset",
        measure="Where the grasp centre sits relative to the flange face, measured on the real coupling.",
        unit="mm",
    ),
    Writable(
        path="robot.gripper.tool_frame.rotation_quat_xyzw",
        label="Flange -> grasp centre rotation",
        measure=(
            "Which flange axis the jaws close along. This half never crashes: ninety degrees out "
            "produces 10/10 logged successes with the jaws closing across the wrong object axis, and "
            "hand-eye calibration returns an excellent RMSE either way."
        ),
    ),
    Writable(
        path="camera.cameras.rigs[*].serial_number",
        label="Camera serial",
        measure=(
            "rs-enumerate-devices -s. With two identical D435s the serial is the only stable identity; "
            "device_index is the SDK's enumeration order and can swap between boots. Cameras that swap "
            "do not fail they hand back a complete, plausible scene with the views exchanged."
        ),
    ),
    Writable(
        path="robot.ur.ip",
        label="UR controller address",
        measure=(
            "The controller's address on your network, read it off the pendant under Settings > "
            "Network. It is a fact about the cell, like a camera serial, which is why it is writable "
            "here at all. It is also the one value that decides WHICH machine receives every motion, "
            "so it cannot be changed while anything is connected."
        ),
        vendor="ur",
        requires_disconnected=True,
    ),
    Writable(
        path="robot.kuka.controller_ip",
        label="KUKA controller address",
        measure=(
            "The controller's address on your network. As with UR: a fact about the cell, and the one "
            "value that decides which machine moves, so it is refused while anything is connected."
        ),
        vendor="kuka",
        requires_disconnected=True,
    ),
)

_INDEXED = re.compile(r"^(?P<head>.*?)\[(?P<index>\d+)\](?P<tail>.*)$")


def writable(key: str) -> Writable | None:
    """The :class:`Writable` for ``key``, or ``None``. Indexed paths match their ``[*]`` template."""
    generic = re.sub(r"\[\d+\]", "[*]", key)
    for entry in WRITABLE:
        if entry.path in (key, generic):
            return entry
    return None


def _known(key: str) -> bool:
    """Whether the schema accepts ``key``. ``rigs[0].x`` is indexed under the list's element shape."""
    index = schema_index()
    if key in index:
        return True
    stripped = re.sub(r"\[\d+\]", "", key)
    return stripped in index or f"{stripped}" in {re.sub(r"\[\d+\]", "", k) for k in index}


def target_file(key: str, root: Path, layers: tuple[str, ...]) -> Path | None:
    """The file a write to ``key`` must land in, given the profile chain being run.

    **The most specific layer wins, and that is where the measurement belongs.** A payload weighed on
    the UR3e is a fact about the UR3e; writing it into the shared base file would hand the same number
    to the UR5e and to the simulator.
    """
    for base, _top, prefix in section_sources(root):
        stripped = re.sub(r"\[\d+\]", "", key)
        if stripped == prefix or stripped.startswith(f"{prefix}."):
            if not layers:
                return base if base.exists() else None
            return base.with_name(f"{base.stem}.{layers[-1]}{base.suffix}")
    return None


@dataclass(frozen=True, slots=True)
class WriteResult:
    """What happened to the whole group. ``applied`` is the only thing a caller must branch on."""

    applied: bool
    keys: tuple[str, ...]
    #: Read back out of the reloaded tree, not echoed from the request -- the value that survived
    #: coercion and the validators is the one the cell will actually run with. Empty when refused.
    values: dict[str, Any] = field(default_factory=dict)
    files: tuple[Path, ...] = ()
    refused: WriteRefused | None = None
    #: Which key the refusal is about, when it is about one. Empty for a whole-tree validation failure.
    refused_key: str = ""
    #: Operator-readable. For ``INVALID_VALUE`` this is the loader's own message, unabridged -- it
    #: already names the file, the line and the validator's own sentence.
    message: str = ""


# --------------------------------------------------------------------------------------------------
# YAML leaf writing. Line-based on purpose: see the module docstring.
# --------------------------------------------------------------------------------------------------

def _emit(value: Any) -> str:
    """A scalar or flat list as this tree writes it: flow lists, quoted strings, ``null``."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_emit(item) for item in value) + "]"
    text = str(value)
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _block_end(lines: list[str], start: int, indent: int) -> int:
    """Index one past the last line belonging to the block opened at ``start`` (indent > ``indent``).

    Blank and comment lines inside the block are carried along; trailing ones are not, so an insert
    lands against the block's last real line rather than after the blank that separates the next one.
    """
    end = start + 1
    last_real = start + 1
    while end < len(lines):
        stripped = lines[end].strip()
        if stripped and not stripped.startswith("#") and _indent_of(lines[end]) <= indent:
            break
        if stripped and not stripped.startswith("#"):
            last_real = end + 1
        end += 1
    return last_real


def _find_child(lines: list[str], lo: int, hi: int, name: str, indent: int) -> int | None:
    """Line index of ``name:`` at exactly ``indent`` within ``[lo, hi)``."""
    pattern = re.compile(rf"^ {{{indent}}}{re.escape(name)}:(\s|$)")
    for i in range(lo, min(hi, len(lines))):
        if pattern.match(lines[i]):
            return i
    return None


def _find_item(lines: list[str], lo: int, hi: int, position: int, indent: int) -> tuple[int, int] | None:
    """``(line, item_indent)`` of the ``position``-th ``- `` entry at ``indent`` within ``[lo, hi)``."""
    pattern = re.compile(rf"^ {{{indent}}}-(\s|$)")
    seen = 0
    for i in range(lo, min(hi, len(lines))):
        if pattern.match(lines[i]):
            if seen == position:
                # A sequence item's own keys sit at the column after "- ", whether or not the first key
                # shares the dash's line -- which in this tree it always does.
                return i, indent + 2
            seen += 1
    return None


_SET_LINE = re.compile(r"^(?P<lead>\s*(?:-\s+)?)(?P<key>[A-Za-z_][\w.]*):(?P<gap>\s*)(?P<rest>.*)$")


def _rewrite(line: str, value: Any) -> str:
    """Replace the value on an existing ``key: value`` line, keeping indent, padding and any comment."""
    match = _SET_LINE.match(line)
    if match is None:  # pragma: no cover - callers only pass lines this matched already
        raise ValueError(f"not a settable line: {line!r}")
    rest = match["rest"]
    comment = ""
    # Only a comment that follows whitespace is one; a `#` inside a quoted value is not. The values this
    # module writes are numbers, short flow lists and bare identifiers, so a conservative split is safe.
    hit = re.search(r"(?<=\s)#.*$", rest)
    if hit:
        comment = "  " + hit.group(0).strip()
    gap = match["gap"] or " "
    return f"{match['lead']}{match['key']}:{gap}{_emit(value)}{comment}"


def _write_leaf(path: Path, top_key: str | None, dotted: str, value: Any) -> None:
    """Set ``dotted`` (relative to ``top_key``) in ``path``, creating the file or nesting if needed.
    """
    segments: list[str | int] = []
    for raw in dotted.split("."):
        match = _INDEXED.match(raw)
        if match is None:
            segments.append(raw)
            continue
        if match["head"]:
            segments.append(match["head"])
        segments.append(int(match["index"]))

    if not path.exists():
        lines: list[str] = []
        if top_key:
            lines.append(f"{top_key}:")
    else:
        lines = path.read_text(encoding="utf-8").splitlines()

    lo, hi, indent, parent = 0, len(lines), 0, -1
    if top_key:
        found = _find_child(lines, 0, len(lines), top_key, 0)
        if found is None:
            lines.append(f"{top_key}:")
            found = len(lines) - 1
            hi = len(lines)
        else:
            hi = _block_end(lines, found, 0)
        lo, indent, parent = found + 1, 2, found

    for position, segment in enumerate(segments):
        last = position == len(segments) - 1
        if isinstance(segment, int):
            item = _find_item(lines, lo, hi, segment, indent)
            if item is None:
                raise ValueError(f"{path.name}: no item [{segment}] under the block ending at line {hi}")
            parent, indent = item[0], item[1]
            lo, hi = parent, _block_end(lines, parent, indent - 2)
            continue

        found = _find_child(lines, lo, hi, segment, indent)
        if found is None:
            # Everything from here down is new. Insert it as a block at the parent's insertion point.
            insert = hi if parent >= 0 else len(lines)
            block: list[str] = []
            depth = indent
            for tail in segments[position:-1]:
                block.append(" " * depth + f"{tail}:")
                depth += 2
            block.append(" " * depth + f"{segments[-1]}: {_emit(value)}")
            lines[insert:insert] = block
            break
        if last:
            lines[found] = _rewrite(lines[found], value)
            break
        parent, lo = found, found + 1
        hi = _block_end(lines, found, indent)
        indent += 2

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def set_keys(
    items: Mapping[str, Any],
    *,
    root: Path,
    layers: tuple[str, ...] = (),
    profile: str | None = None,
    connected: bool = False,
) -> WriteResult:
    """Write a group of measured values as ONE transaction: all of them land, or none do.

    **The group is not a convenience, it is a correctness requirement.** The three ``tool_frame`` keys
    are a single measurement and the schema knows it: declaring ``source: "willy"`` while the offset is
    still identity is rejected, on the grounds that a grasp centre exactly at the flange face is the
    shape of an unmeasured cell rather than a mounted tool. Written one at a time there is no order that
    validates -- the first write is always half a tool frame. So every file is edited, the tree is
    reloaded once, and on any failure every file is restored to the bytes it had.

    ``profile`` is the chain string the reload runs under; ``layers`` is the same chain split, used to
    pick the target files. Both are passed in because the caller already holds both, and re-deriving one
    from the other here would be a second place for them to disagree.
    """
    if not items:
        return WriteResult(applied=True, keys=())

    plan: list[tuple[str, Any, Path, str | None, str]] = []
    for key, value in items.items():
        entry = writable(key)
        if entry is not None and entry.requires_disconnected and connected:
            # The controller address is the one writable value that does not describe the tool -- it
            # decides which machine every motion goes to. Changing it under a live connection leaves the
            # process driving one arm while the config names another, and nothing downstream would
            # notice: the driver holds the address it connected with.
            return WriteResult(
                applied=False, keys=tuple(items), refused=WriteRefused.CELL_CONNECTED, refused_key=key,
                message=(
                    f"{key} decides which machine receives every motion, so it cannot be changed while "
                    f"anything is connected. Disconnect the cell first."
                ),
            )
        if entry is None:
            if not _known(key):
                return WriteResult(
                    applied=False, keys=tuple(items), refused=WriteRefused.UNKNOWN_KEY,
                    refused_key=key,
                    message=(
                        f"{key} is not a config key. The schema rejects unknown keys at load, so "
                        f"writing it would break the tree rather than change anything."
                    ),
                )
            return WriteResult(
                applied=False, keys=tuple(items), refused=WriteRefused.NOT_WRITABLE, refused_key=key,
                message=(
                    f"{key} is real but not writable from a tool. Limits, thresholds and safety "
                    f"toggles live next to the comment that carries the evidence for their value; edit "
                    f"them in YAML, where that comment is readable. Writable here: "
                    + ", ".join(w.path for w in WRITABLE)
                ),
            )
        path = target_file(key, root, layers)
        section = next(
            (s for s in section_sources(root) if re.sub(r"\[\d+\]", "", key).startswith(s[2])), None
        )
        if path is None or section is None:
            return WriteResult(
                applied=False, keys=tuple(items), refused=WriteRefused.NO_TARGET, refused_key=key,
                message=f"no file backs {key} under the config root {root}.",
            )
        _base, top_key, prefix = section
        plan.append((key, value, path, top_key, key[len(prefix):].lstrip(".")))

    # Snapshot every file the group touches BEFORE editing any of them; `None` marks one that did not
    # exist, so a rollback deletes it instead of writing an empty file back.
    snapshot: dict[Path, str | None] = {
        path: (path.read_text(encoding="utf-8") if path.exists() else None)
        for _k, _v, path, _t, _d in plan
    }
    try:
        for _key, value, target, section_key, dotted in plan:
            _write_leaf(target, section_key, dotted, value)
        loaded = _reload(root, profile)
    except Exception as exc:  # noqa: BLE001
        for path, before in snapshot.items():
            if before is None:
                path.unlink(missing_ok=True)
            else:
                path.write_text(before, encoding="utf-8")
        _reload(root, profile)
        return WriteResult(
            applied=False, keys=tuple(items), files=tuple(snapshot), refused=WriteRefused.INVALID_VALUE,
            message=f"{exc}",
        )

    read_back = {key: _read_back(loaded, key) for key in items}
    absent = [key for key, value in read_back.items() if value is _MISSING]
    if absent:  # pragma: no cover - a live tree that validates always exposes what it validated
        return WriteResult(
            applied=False, keys=tuple(items), files=tuple(snapshot), refused=WriteRefused.NO_TARGET,
            refused_key=absent[0],
            message=f"wrote {absent[0]} and the reloaded tree does not expose it -- refusing to claim it applied.",
        )
    return WriteResult(applied=True, keys=tuple(items), files=tuple(snapshot), values=read_back)


def set_key(
    key: str,
    value: Any,
    *,
    root: Path,
    layers: tuple[str, ...] = (),
    profile: str | None = None,
    connected: bool = False,
) -> WriteResult:
    """One key, written as a group of one. See :func:`set_keys` for why groups are the primitive."""
    return set_keys(
        {key: value}, root=root, layers=layers, profile=profile, connected=connected,
    )


def _reload(root: Path, profile: str | None) -> Any:
    """Load the tree exactly as a runner would, under ``profile``. Raises on anything the loader hates."""
    from .loader import active_profile, load_config, reload_config, set_active_profile

    previous = active_profile()
    if profile != previous:
        set_active_profile(profile)
    try:
        reload_config()
        return load_config(root)
    finally:
        if profile != previous:
            set_active_profile(previous)
            reload_config()


#: Returned by :func:`_read_back` for a path that does not exist. Distinct from ``None``, which is a
#: perfectly good value here -- an unconfigured ``serial_number`` IS ``null``, and reporting that as
#: "missing" would tell an operator their write vanished when it landed exactly as asked.
_MISSING = object()


def _read_back(cfg: Any, key: str) -> Any:
    """The value the reloaded tree holds for ``key``, or :data:`_MISSING` if the path does not exist."""
    node: Any = cfg
    for raw in key.split("."):
        match = _INDEXED.match(raw)
        name = match["head"] if match else raw
        if name:
            node = node.get(name, _MISSING) if isinstance(node, dict) else getattr(node, name, _MISSING)
            if node is _MISSING:
                return _MISSING
        if match:
            index = int(match["index"])
            if not isinstance(node, (list, tuple)) or not 0 <= index < len(node):
                return _MISSING
            node = node[index]
    return node