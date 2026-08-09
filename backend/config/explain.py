"""Asking the config about itself: what is this key, where was it set, and why that value?

The config could always be READ. It could not be ASKED. With a profile chain merging several files into
one section, "read it" means opening four files and reconstructing a merge in your head — and even then
the two questions that matter go unanswered:

  * **which layer set this?** ``_deep_merge`` returns the value and forgets the file.
  * **why THIS number?** The YAML comments carry the measured evidence (*"6/6 up to 6 mm and 0/6 at
    10 mm"*) and ``yaml.safe_load`` discards every one of them.

There is a third question the files cannot answer at all: **what am I allowed to set?** 107 schema fields
appear in no shipped YAML, including ``robot.ur.model`` — so someone configuring real hardware by reading
``robot.yaml`` cannot discover the field that decides which robot they are configuring.

This module answers all three, read-only, from the tree as it stands. Nothing moves: the comment shown
is read from the file its author wrote it in, at query time.

Prior art for the shape of the answer: ``git config --show-origin --show-scope``, PostgreSQL's
``pg_settings.sourcefile/sourceline``, ``nixos-option`` (value + default + description + declared-by +
defined-by in one command), and ``ansible-config dump --only-changed``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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

__all__ = ["decisions", "explain_key", "find_keys"]

_INDENT = "      "


def _yaml_path(path: str) -> str:
    """The spelling the YAML (and therefore the provenance index) uses for ``path``.

    The provenance index is keyed by what is WRITTEN in the file, so an aliased field must be looked up
    under its alias or it is reported as "no YAML sets this" while the file plainly sets it -- measured
    on all 7 camelCase stereomatcher keys.
    """
    return alias_for(path) or path


def _schema_lookup(index: dict[str, Any], path: str) -> Any:
    field = index.get(path)
    if field is not None:
        return field
    # The stereomatcher block keeps OpenCV's camelCase in YAML behind snake_case attributes, so a
    # question asked with the ATTRIBUTE name must still find the field the schema knows by its alias.
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


def explain_key(path: str, root: Path, layers: tuple[str, ...], value: Any = None) -> str:
    """Render everything known about one dotted config key."""
    index = schema_index()
    # A dict[str, Model] block (e.g. robot.sim.cameras.overhead) is indexed once under `*`, so a
    # concrete key finds the shared shape rather than reporting "unknown".
    field = _schema_lookup(index, path)
    chains = index_chains(root, layers)
    chain = chains.get(_yaml_path(path), [])

    out: list[str] = []
    head = f"{path}" if value is None else f"{path} = {value!r}"
    out.append(head)
    out.append("")

    if field is None:
        out.append(f"{_INDENT}NOT A KNOWN KEY. The schema does not accept it, so setting it would be")
        out.append(f"{_INDENT}rejected at load (unknown keys are never ignored).")
        for suggestion in nearest_keys(path, sorted(index), limit=5):
            out.append(f"{_INDENT}did you mean: {suggestion}?")
        return "\n".join(out)

    out.append(f"{_INDENT}tier      {_tier(path, index, None, chains.get(_yaml_path(path), []))}")
    out.append(f"{_INDENT}type      {field.summary()}")
    out.append(f"{_INDENT}default   {field.default!r}" + ("  (REQUIRED)" if field.required else ""))
    # `#:` comments in the schema source are how this project documents fields; Pydantic does not lift
    # them into the JSON schema, so the explanation existed and was invisible to every tool.
    meaning = field.description or field_doc(path) or field_doc(field.path)
    label = "means"
    if not meaning:
        # No per-field prose: the block's own docstring is usually the better answer anyway, and it is
        # already written and already maintained. Labelled differently so the reader knows the scope.
        meaning, label = model_doc(path) or model_doc(field.path), "block"
    if meaning:
        first, *rest = meaning.splitlines()
        out.append(f"{_INDENT}{label:9s} {first}")
        for line in rest:
            out.append(f"{_INDENT}          {line.strip()}")

    out.append("")
    if not chain:
        # Not written anywhere: the value in force is the schema default. Saying so is the answer --
        # and it is how the 107 never-written fields become discoverable at all.
        out.append(f"{_INDENT}set in    (no YAML sets this -- the schema default is in force)")
        return "\n".join(out)

    winner = chain[-1]
    out.append(f"{_INDENT}set in    {winner.location(root)}")
    if len(chain) > 1:
        out.append("")
        out.append(f"{_INDENT}layer chain (last wins)")
        for origin in chain:
            marker = "  <- winner" if origin is winner else ""
            raw = read_value(origin)
            shown = f" = {raw}" if raw else ""
            out.append(f"{_INDENT}  {origin.location(root)}{shown}{marker}")

    note = comment_above(winner)
    if note:
        out.append("")
        out.append(f"{_INDENT}why (comment above that line)")
        for line in note.splitlines():
            out.append(f"{_INDENT}  {line}")
    return "\n".join(out)


def find_keys(needle: str, limit: int = 40, tier: str | None = None) -> str:
    """Search the SCHEMA (not the YAML) for keys matching ``needle``.

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
    # comes from the schema default: "advanced unless a cell enables it" -- true, and the honest thing to
    # say without a tree.
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
            f"\n  {hidden} more in other tiers -- `--tier all` shows every one. Nothing is hidden from "
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
    """Only the values that DIFFER from their schema default -- the decisions someone actually made.

    Measured: 3 of 310 robot leaves differ from the schema default, so a 448-line ``robot.yaml`` encodes
    three decisions and 307 restatements. ``--print`` dumps all 876 lines of the tree with no way to tell
    the two apart, which is why reading the config does not tell you what the cell was configured to do.

    Comparing against the SCHEMA INDEX rather than against a default-constructed ``AppConfig`` is
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
        where = chain[-1].location(root) if chain else "(not in any YAML -- set by a validator or code)"
        # `decided=True` by construction: everything reaching here already differs from its default.
        # The gate state uses the LOADED values, so a sim cell's own fields are not called "advanced"
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
    lines = [f"{len(rows)} value(s){scope} differ from the schema default -- i.e. the decisions:"]
    for group_name in TIERS:
        group = [row for row in rows if row[3] == group_name]
        if not group:
            continue
        lines.append(f"\n  [{group_name}]")
        for path, value, where, _ in group:
            lines.append(f"    {path:<{width}}  = {value!r}\n    {'':<{width}}    {where}")
    lines.append("\nEverything else is a schema default. `explain <key>` says why any one of these is set.")
    return "\n".join(lines)
