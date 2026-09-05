"""Asking the config about itself: what is this key, where was it set, and why that value?

The config could always be read. It could not be asked. With a profile chain merging several files into
one section, "read it" means opening four files and reconstructing a merge in your head, and even then
the two questions that matter go unanswered:

  * **which layer set this?** ``_deep_merge`` returns the value and forgets the file.
  * **why this number?** The YAML comments carry the measured evidence (*"6/6 up to 6 mm and 0/6 at
    10 mm"*) and ``yaml.safe_load`` discards every one of them.

There is a third question the files cannot answer at all: **what am I allowed to set?** 107 schema fields
appear in no shipped YAML, including ``robot.ur.model``, so someone configuring real hardware by reading
``robot.yaml`` cannot discover the field that decides which robot they are configuring.

This module answers all three, read-only, from the tree as it stands. Nothing moves: the comment shown
is read from the file its author wrote it in, at query time.

Prior art for the shape of the answer: ``git config --show-origin --show-scope``, PostgreSQL's
``pg_settings.sourcefile/sourceline``, ``nixos-option`` (value + default + description + declared-by +
defined-by in one command), and ``ansible-config dump --only-changed``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .edit import MISSING, read_key
from ._provenance import comment_above, index_chains, nearest_keys, read_value
from ._tiers import TIERS, gate_state, tier_for
from ._schema_index import (
    alias_for,
    field_default,
    field_doc,
    model_doc,
    same_value,
    schema_index,
)

__all__ = ["KeyExplanation", "Layer", "decisions", "explain", "explain_in", "explain_key", "find_keys"]

_INDENT = "      "


def _yaml_path(path: str) -> str:
    """The spelling the YAML (and therefore the provenance index) uses for ``path``.

    The provenance index is keyed by what is written in the file, so an aliased field must be looked up
    under its alias or it is reported as "no YAML sets this" while the file plainly sets it,
    measured on all 7 camelCase stereomatcher keys.
    """
    return alias_for(path) or path


def _schema_lookup(index: dict[str, Any], path: str) -> Any:
    field = index.get(path)
    if field is not None:
        return field
    # The stereomatcher block keeps OpenCV's camelCase in YAML behind snake_case attributes, so a
    # question asked with the attribute name must still find the field the schema knows by its alias.
    aliased = alias_for(path)
    if aliased is not None and aliased in index:
        return index[aliased]
    # Try replacing each segment with `*` (left to right) to hit a dict-of-models entry.
    parts = path.split(".")
    for i in range(len(parts)):
        candidate = ".".join(parts[:i] + ["*"] + parts[i + 1:])
        field = index.get(candidate)
        if field is not None:
            return field
    return None


def _tier(path: str, index: dict[str, Any], values: dict[str, Any] | None, chain: list) -> str:
    """The tier of ``path``, using the loaded values when the caller has them."""
    field = _schema_lookup(index, path)
    if field is None:
        return "tuned"
    decided = bool(chain) and (
        values is None or not same_value(values.get(path), field_default(path, field.default))
    )
    return tier_for(
        field.path, required=field.required, decided=decided,
        gated_off=gate_state(field.path, index, values),
    )


@dataclass(frozen=True)
class Layer:
    """One file that sets a key, as it will be shown: where, and what it wrote there."""

    location: str
    raw: str
    winner: bool


@dataclass(frozen=True)
class KeyExplanation:
    """Everything known about one dotted key, decided but not yet rendered.

    Split out from the text so a second consumer cannot drift from the CLI. The operator console shows
    the same provenance in a browser, and re-deriving "which layer won" there would be a second
    implementation of the one question this module exists to answer. That is exactly how the browser
    and the terminal end up disagreeing about a cell. ``render()`` below produces the CLI text from this
    and nothing else, so the two are the same answer in two costumes.
    """

    path: str
    #: ``None`` when the caller has no loaded tree (``where`` must work while the config is broken).
    value: Any = None
    has_value: bool = False
    #: False for a key the schema does not accept; then only ``suggestions`` is populated.
    known: bool = False
    suggestions: tuple[str, ...] = ()
    tier: str = ""
    type_summary: str = ""
    default: Any = None
    required: bool = False
    doc: str = ""
    #: ``"means"`` for per-field prose, ``"block"`` when it fell back to the block's docstring. The
    #: distinction is the reader's: one describes this key, the other describes its neighbourhood.
    doc_scope: str = ""
    layers: tuple[Layer, ...] = ()
    #: The YAML comment above the winning line, where this project keeps the measured evidence.
    comment: str = ""

    @property
    def set_in(self) -> str:
        """The file:line in force, or the empty string when the schema default is."""
        return self.layers[-1].location if self.layers else ""

    def render(self) -> str:
        """The CLI text. The only renderer. See the class docstring."""
        out: list[str] = []
        out.append(f"{self.path} = {self.value!r}" if self.has_value else self.path)
        out.append("")

        if not self.known:
            out.append(f"{_INDENT}NOT A KNOWN KEY. The schema does not accept it, so setting it would be")
            out.append(f"{_INDENT}rejected at load (unknown keys are never ignored).")
            for suggestion in self.suggestions:
                out.append(f"{_INDENT}did you mean: {suggestion}?")
            return "\n".join(out)

        out.append(f"{_INDENT}tier      {self.tier}")
        out.append(f"{_INDENT}type      {self.type_summary}")
        out.append(f"{_INDENT}default   {self.default!r}" + ("  (REQUIRED)" if self.required else ""))
        if self.doc:
            first, *rest = self.doc.splitlines()
            out.append(f"{_INDENT}{self.doc_scope:9s} {first}")
            for line in rest:
                out.append(f"{_INDENT}          {line.strip()}")

        out.append("")
        if not self.layers:
            # Not written anywhere: the value in force is the schema default. Saying so is
            # the answer, and it is how the 107 never-written fields become discoverable at all.
            out.append(f"{_INDENT}set in    (no YAML sets this: the schema default is in force)")
            return "\n".join(out)

        out.append(f"{_INDENT}set in    {self.set_in}")
        if len(self.layers) > 1:
            out.append("")
            out.append(f"{_INDENT}layer chain (last wins)")
            for layer in self.layers:
                marker = "  <- winner" if layer.winner else ""
                shown = f" = {layer.raw}" if layer.raw else ""
                out.append(f"{_INDENT}  {layer.location}{shown}{marker}")

        if self.comment:
            out.append("")
            out.append(f"{_INDENT}why (comment above that line)")
            for line in self.comment.splitlines():
                out.append(f"{_INDENT}  {line}")
        return "\n".join(out)


def explain(
    path: str, root: Path, layers: tuple[str, ...], value: Any = None, *, has_value: bool | None = None
) -> KeyExplanation:
    """Everything known about one dotted config key, structured.

    ``has_value`` defaults to ``value is not None``, which is what the CLI wants (it passes the loaded
    value or nothing). A caller holding a key whose value legitimately IS ``None``
    (``serial_number`` on an unconfigured camera, say) passes ``has_value=True`` to have it shown as
    set rather than omitted.
    """
    index = schema_index()
    # A dict[str, Model] block (e.g. robot.sim.cameras.overhead) is indexed once under `*`, so a
    # concrete key finds the shared shape rather than reporting "unknown".
    field = _schema_lookup(index, path)
    chain = index_chains(root, layers).get(_yaml_path(path), [])
    shown = (value is not None) if has_value is None else has_value

    if field is None:
        return KeyExplanation(
            path=path, value=value, has_value=shown, known=False,
            suggestions=tuple(nearest_keys(path, sorted(index), limit=5)),
        )

    # `#:` comments in the schema source are how this project documents fields; Pydantic does not lift
    # them into the JSON schema, so the explanation existed and was invisible to every tool.
    meaning = field.description or field_doc(path) or field_doc(field.path)
    scope = "means"
    if not meaning:
        # No per-field prose: the block's own docstring is usually the better answer anyway, and it is
        # already written and already maintained. Labelled differently so the reader knows the scope.
        meaning, scope = model_doc(path) or model_doc(field.path), "block"

    winner = chain[-1] if chain else None
    return KeyExplanation(
        path=path, value=value, has_value=shown, known=True,
        tier=_tier(path, index, None, chain),
        type_summary=field.summary(), default=field.default, required=field.required,
        doc=meaning or "", doc_scope=scope if meaning else "",
        layers=tuple(
            Layer(location=o.location(root), raw=read_value(o), winner=o is winner) for o in chain
        ),
        comment=comment_above(winner) if winner is not None else "",
    )


def explain_in(cfg: Any, path: str, root: Path, layers: tuple[str, ...]) -> KeyExplanation:
    """Everything known about ``path`` IN ``cfg``: the read and the explanation, in one call.

    **This exists because the two callers disagreed, and the class below could not stop them.**
    :class:`KeyExplanation` was split out from its text (see its docstring) so the operator console
    and the terminal could not drift. They drifted anyway, one layer higher up: the drift was not in
    how the answer is rendered but in what each caller found out before asking.

    The console read the tree with ``edit.read_key``, which returns :data:`~src.config.edit.MISSING`
    for an absent path, and passed ``has_value`` explicitly. The CLI used its own walker, which
    returned ``None`` for both "absent" and "the value is None", so it let ``has_value`` default to
    ``value is not None``.

    Measured on the default tree: **45 of 468 keys hold ``None``**, and for every one of them::

        CLI      camera.cameras.active_rig_id
        console  camera.cameras.active_rig_id = None

    One reads as "nobody set this", the other as "this is set to null". Both were produced from the
    same function, from the same tree, about the same key.

    **So the fix is a function that cannot be called two ways.** Neither caller supplies ``value``
    or ``has_value`` any more, because neither is trusted to derive them. :func:`explain` keeps both
    parameters for a caller that genuinely holds a value the tree does not (a proposed write, a
    value from a form), which is a different question and deserves a different door.
    """
    value = read_key(cfg, path)
    found = value is not MISSING
    return explain(path, root, layers, None if not found else value, has_value=found)


def explain_key(
    path: str, root: Path, layers: tuple[str, ...], value: Any = None, *,
    has_value: bool | None = None,
) -> str:
    """Render everything known about one dotted config key.

    Prefer :func:`explain_in` when you hold the loaded tree: it does the read too, and therefore
    cannot be given a ``has_value`` that disagrees with another caller's. ``has_value`` is forwarded
    here for the same reason :func:`explain` accepts it.
    """
    return explain(path, root, layers, value, has_value=has_value).render()


def find_keys(needle: str, limit: int = 40, tier: str | None = None) -> str:
    """Search the schema (not the YAML) for keys matching ``needle``.

    Searching the schema is the point: the files only contain what someone chose, so grepping them for
    "gripper" returns five hits and misses the whole suction end-effector. The schema knows all 558.
    """
    index = schema_index()
    hits = sorted(path for path in index if needle.lower() in path.lower())
    if not hits:
        close = nearest_keys(needle, sorted(index), limit=5)
        body = "".join(f"\n  did you mean: {c}?" for c in close)
        return f"no config key matches {needle!r}.{body}"
    # No tree is loaded here on purpose (`where` must work when the config is broken), so the gate state
    # comes from the schema default: "advanced unless a cell enables it". That is true, and the
    # honest thing to say without a tree.
    by_tier: dict[str, list[str]] = {}
    for path in hits:
        field = index[path]
        name = tier_for(
            path, required=field.required, decided=False, gated_off=gate_state(path, index, None),
        )
        by_tier.setdefault(name, []).append(path)
    shown = [t for t in TIERS if t in by_tier and tier in (None, "all", t)]
    hidden = sum(len(v) for t, v in by_tier.items() if t not in shown)

    lines = [f"{len(hits)} key(s) match {needle!r}:"]
    for name in shown:
        lines.append(f"\n  [{name}]")
        for path in by_tier[name][:limit]:
            field = index[path]
            lines.append(f"    {path}\n        {field.summary()}, default {field.default!r}")
        if len(by_tier[name]) > limit:
            lines.append(f"    ... and {len(by_tier[name]) - limit} more")
    if hidden:
        lines.append(
            f"\n  {hidden} more in other tiers: `--tier all` shows every one. Nothing is hidden from "
            f"the CONFIG itself: every field, shown or not, stays settable."
        )
    return "\n".join(lines)


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """A validated config object as ``dotted.path -> leaf``. Lists are leaves (they are set whole)."""
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            out.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {prefix: value}


def decisions(
    cfg: Any, root: Path, layers: tuple[str, ...],
    section: str | None = None, tier: str | None = None,
) -> str:
    """Only the values that differ from their schema default: the decisions someone actually made.

    Measured: 3 of 310 robot leaves differ from the schema default, so a 448-line ``robot.yaml`` encodes
    three decisions and 307 restatements. ``--print`` dumps all 876 lines of the tree with no way to tell
    the two apart, which is why reading the config does not tell you what the cell was configured to do.

    Comparing against the schema index rather than against a default-constructed ``AppConfig`` is
    deliberate and necessary: ``AppConfig()`` is not constructible (``camera`` and ``models`` are
    required), so there is no all-defaults tree to diff against. The index has every default anyway.

    Prior art: ``ansible-config dump --only-changed``, ``helm get values`` (user-supplied) vs ``-a``
    (computed), ``sshd -T``, ``tsc --showConfig``.
    """
    index = schema_index()
    chains = index_chains(root, layers)
    values = _flatten(cfg)
    rows: list[tuple[str, Any, str, str]] = []
    for path, value in sorted(values.items()):
        if section and not path.startswith(section):
            continue
        field = _schema_lookup(index, path)
        if field is None:
            continue  # a dict key the schema does not describe (e.g. a user-named camera id)
        if same_value(value, field_default(path, field.default)):
            continue
        chain = chains.get(_yaml_path(path), [])
        where = chain[-1].location(root) if chain else "(not in any YAML: set by a validator or code)"
        # `decided=True` by construction: everything reaching here already differs from its default.
        # The gate state uses the loaded values, so a sim cell's own fields are not called "advanced"
        # merely because the block defaults to disabled.
        name = tier_for(
            field.path, required=field.required, decided=True,
            gated_off=gate_state(field.path, index, values),
        )
        if tier not in (None, "all", name):
            continue
        rows.append((path, value, where, name))

    scope = f" under {section}" if section else ""
    if not rows:
        return f"no value{scope} differs from its schema default."
    width = max(len(p) for p, _, _, _ in rows)
    lines = [f"{len(rows)} value(s){scope} differ from the schema default, i.e. the decisions:"]
    for group_name in TIERS:
        group = [row for row in rows if row[3] == group_name]
        if not group:
            continue
        lines.append(f"\n  [{group_name}]")
        for path, value, where, _ in group:
            lines.append(f"    {path:<{width}}  = {value!r}\n    {'':<{width}}    {where}")
    lines.append("\nEverything else is a schema default. `explain <key>` says why any one of these is set.")
    return "\n".join(lines)
