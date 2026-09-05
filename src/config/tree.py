"""A config tree as one value: the directory, the chain, and the layers that chain splits into.

`explain_in`, `decisions`, `index_chains` and `set_keys` each take `root` and `layers` (and
`set_keys` also a `profile`) as independent arguments beside the already-loaded config. Nothing binds
those arguments to one another, so a config loaded under one chain can be walked for provenance under
another, and the answer that comes back is confidently wrong with no error raised:

  * on the read path the value comes from the config and the origin from `layers`, so the tool whose
    whole purpose is "which layer set this" can report `robot.sim.enabled = True` above
    `robot/robot.yaml:31`, a line that sets `false`.
  * on the write path `set_keys` picks the target file from `layers[-1]` and validates under
    `profile`, so `layers=("ur3e",)` with `profile=None` writes `mass_kg: 3.25` into
    `robot.ur3e.yaml` beneath the `max_mass_kg: 3.0` that forbids it, validates the base tree
    instead, reports `applied=True`, and leaves the tree unloadable under the layer it wrote.

`set_keys` in `edit.py` documents those two shapes of one chain and still takes both, so the
disagreement is constructible there.

So the ask methods hang off the tree and take only a key, and `layers` is derived from `profile`
rather than passed beside it. Carrying the three facts in one object is not on its own enough:
`api/cell.py`'s `Console` is a root+profile+layers carrier, and `api/routers/config.py:56` and `:138`
unpack it into four and five separate arguments by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from src.contracts import UNSET, Maybe, chosen

if TYPE_CHECKING:  # pragma: no cover, typing only
    from .edit import WriteResult
    from .explain import KeyExplanation

__all__ = ["ConfigTree", "LoadedTree", "default_data_dir"]


def default_data_dir() -> Path:
    """The directory `load_config()` reads when nobody names one.

    The public accessor for `loader.py:84`'s private `_DEFAULT_DATA_DIR`, a `parents[2]` walk that
    no other file can repeat. `explain`, `decisions` and `set_keys` all take `root` as a required
    argument, and a wrong root produces a silently wrong answer rather than an error, so a caller
    builds it here rather than deriving its own.
    """
    from .loader import _DEFAULT_DATA_DIR  # noqa: PLC0415

    return _DEFAULT_DATA_DIR


@dataclass(frozen=True, slots=True)
class LoadedTree:
    """A validated tree, and the three facts it was validated under.

    The ask methods live here rather than on `ConfigTree` for one reason: they need a loaded
    config and the root and the layers, and this is the only object that holds all three at once.
    A method that took the config as an argument would put the disagreement back.
    """

    tree: "ConfigTree"
    #: The validated `AppConfig`, or `None` when the tree did not load.
    config: Any = None
    #: The refusal, verbatim. Empty when it loaded.
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.config is not None

    @property
    def exit_code(self) -> int:
        """The process exit code: 0 when the tree loaded, 1 when it did not.

        The same 1 the CLI's `except ConfigError: return 1` returns, and the only source of a 1 in
        that entry point.
        """
        return 0 if self.ok else 1

    @property
    def chain(self) -> str:
        """The layers as a person reads them, or ``(no profile)``.

        The chain that was actually loaded, including whatever `WILLY_PROFILE` contributed, not the
        one that was asked for.
        """
        return " -> ".join(self.tree.layers) or "(no profile)"

    # --- the questions, each taking only what it is about --------------------------------------

    def explain(self, key: str) -> "KeyExplanation":
        """Everything known about one key: its value here, and which file decided it.

        `explain_in` reads the value out of a config and walks `root` + `layers` for the origin.
        All three come from this tree, so the value and the provenance cannot describe different
        loads.
        """
        from .explain import explain_in  # noqa: PLC0415

        return explain_in(self.config, key, self.tree.root, self.tree.layers)

    def decisions(
        self, *, section: "Maybe[str | None]" = UNSET, tier: "Maybe[str | None]" = UNSET
    ) -> str:
        """Only the values that differ from their schema default: what someone actually decided.

        Neither filter is defaulted here. `explain.decisions` declares `section=None` and
        `tier=None` already, and a copy of those in this signature would be a second declaration of
        one fact.
        """
        from .explain import decisions as _decisions  # noqa: PLC0415

        extra: dict[str, Any] = {}
        if chosen(section):
            extra["section"] = section
        if chosen(tier):
            extra["tier"] = tier
        return _decisions(self.config, self.tree.root, self.tree.layers, **extra)

    # --- the report halves ---------------------------------------------------------------------

    def render(self) -> str:
        """The verdict in one line. ASCII, no trailing newline, no arguments."""
        if not self.ok:
            return f"config error:\n{self.error}"
        named = str(self.tree.named_root) if self.tree.named_root is not None else "<default>"
        return f"OK: config under {named} validates.  layers: {self.chain}"

    def to_dict(self) -> dict[str, Any]:
        """Plain data. The config itself is not in here: it is a Pydantic model with its own
        `model_dump_json`, and a second serialisation of it would be a second answer.
        """
        return {
            "root": str(self.tree.root),
            "profile": self.tree.profile,
            "layers": list(self.tree.layers),
            "chain": self.chain,
            "ok": self.ok,
            "exit_code": self.exit_code,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class ConfigTree:
    """Where the YAML lives and which overlays are in force, as one value.

        from src.config import ConfigTree

        loaded = ConfigTree.from_directory(profile="sim").load()
        print(loaded.render())
        print(loaded.explain("robot.sim.enabled").render())

    `layers` is derived from `profile`, never passed. The two are the same fact in two shapes, and a
    caller allowed to supply both is a caller allowed to make them disagree.
    """

    root: Path
    #: The chain string. `None` means the base tree with no overlays, and it is a real value: a
    #: caller who has not chosen is `UNSET` at the factory and gets whatever `WILLY_PROFILE` says.
    profile: str | None
    layers: tuple[str, ...] = ()
    #: The root as the caller named it, or `None` when they named none. Only for `render()`, which
    #: echoes the operator's own words rather than the resolved path.
    named_root: str | None = field(default=None, compare=False)

    @classmethod
    def from_directory(
        cls,
        *,
        root: "Maybe[str | Path | None]" = UNSET,
        profile: "Maybe[str | None]" = UNSET,
    ) -> "ConfigTree":
        """The tree at ``root`` under ``profile``, with the layers derived from the chain.

        Three states for `profile`, and all three are meaningful. `UNSET` is "the caller did not
        choose", which lets `WILLY_PROFILE` decide; `None` is "the base tree, ignore the variable";
        a string is that chain. Collapsing the first two is how an exported variable gets silently
        disabled for an operator who set it deliberately.
        """
        from .loader import active_profile, profile_layers  # noqa: PLC0415

        named = None if not chosen(root) or root is None else str(root)
        resolved_root = Path(named).resolve() if named is not None else default_data_dir()
        chain = profile if chosen(profile) else active_profile()
        return cls(
            root=resolved_root,
            profile=chain,
            layers=profile_layers(chain),
            named_root=named,
        )

    def load(self) -> LoadedTree:
        """Read and validate. A tree that does not load is a verdict, not an exception.

        The profile travels as an argument rather than through the environment, which is what keeps
        the `source` that `_validated_chain` carries pointing at the thing the operator typed: a bad
        `--profile` is reported against the flag and not against `WILLY_PROFILE`.
        """
        from .loader import ConfigError, load_config  # noqa: PLC0415

        try:
            config = load_config(self.named_root, profile=self.profile)
        except ConfigError as exc:
            return LoadedTree(tree=self, error=str(exc))
        return LoadedTree(tree=self, config=config)

    def write(self, items: "Mapping[str, Any]", *, connected: bool) -> "WriteResult":
        """Write measured values into this tree as one transaction: all land, or none do.

        `connected` has no default. `set_keys` declares `connected: bool = False`, the permissive
        value, and a caller that leaves it out (`api/routers/config.py:138`, with
        `CellSession.connected` one attribute away) skips the `requires_disconnected` guard on the
        two controller-address keys (`robot.ur.ip`, `robot.kuka.controller_ip`): the machine that
        receives every motion is repointable from the browser while the cell is live, with nothing
        in the result saying the guard did not run.

        `root`, `layers` and `profile` come from the tree, so the write cannot land in a file that a
        different chain then validates. `set_keys` raises on that disagreement; here it is
        unconstructible.
        """
        from .edit import set_keys  # noqa: PLC0415

        return set_keys(
            items,
            root=self.root,
            layers=self.layers,
            profile=self.profile,
            connected=connected,
        )
