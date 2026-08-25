"""Tier-0 online learning side-state outside frozen policy artifacts.

Provides the mutable online-learning state for diagonal-LinUCB while keeping
committed policy artifacts immutable and the ShadowRouter strictly
shadow-only. Online observations accumulate as sufficient-statistic deltas
and are snapshotted into an immutable ``OnlineState`` carrying both an
order-sensitive update-chain hash and a content hash.

The effective policy is derived explicitly as ``frozen ⊕ online_state`` via
``apply_online_state``. Online-mutated state never becomes a promoted policy:
its artifact-root ``online_state`` stamp records ``online_mutated=True`` and
``still_shadow=True``, and the promotion gate refuses it until the state is
frozen into a fresh immutable artifact and re-evaluated.

``verify_online_state`` recomputes the content hash before application so
corrupted or manually edited side-state is detected fail-closed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

#: Runtime tiers accepted by :func:`make_online_accumulator` (mirrors ``robot.rl.runtime_tier``).
ONLINE_TIER_STDLIB: str = "stdlib"
ONLINE_TIER_NUMPY: str = "numpy"
ONLINE_TIER_TORCH: str = "torch"
ONLINE_TIERS: tuple[str, ...] = (ONLINE_TIER_STDLIB, ONLINE_TIER_NUMPY, ONLINE_TIER_TORCH)


class OnlineStateError(ValueError):
    """Raised on a dimension/action mismatch or a failed integrity check."""


# ---------------------------------------------------------------------------
# Immutable snapshot.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OnlineState:
    """Immutable snapshot of the accumulated online deltas + provenance.

    The deltas are *additive*: the effective policy adds them onto the frozen
    artifact's ``A_diag`` / ``b`` / ``feature_support`` vectors. ``update_chain_hash``
    is an order-sensitive fold over every observation (tamper / replay evidence);
    ``content_hash`` covers the whole snapshot and is re-checkable via
    :func:`verify_online_state`.
    """

    actions: tuple[str, ...]
    dim: int
    delta_A_diag_per_action: dict[str, tuple[float, ...]]
    delta_b_per_action: dict[str, tuple[float, ...]]
    delta_support_per_action: dict[str, tuple[int, ...]]
    n_updates: int
    high_water_mark: int
    update_chain_hash: str
    content_hash: str
    still_shadow: bool = True

    @property
    def online_mutated(self) -> bool:
        """``True`` once at least one observation has been folded in."""

        return self.n_updates > 0


def _canonical_payload(
    *,
    actions: Sequence[str],
    dim: int,
    delta_A: dict[str, tuple[float, ...]],
    delta_b: dict[str, tuple[float, ...]],
    delta_support: dict[str, tuple[int, ...]],
    n_updates: int,
    high_water_mark: int,
    update_chain_hash: str,
    still_shadow: bool,
) -> str:
    """Deterministic JSON serialisation used for the content hash."""

    return json.dumps(
        {
            "actions": list(actions),
            "dim": dim,
            "delta_A_diag_per_action": {a: list(delta_A[a]) for a in actions},
            "delta_b_per_action": {a: list(delta_b[a]) for a in actions},
            "delta_support_per_action": {a: list(delta_support[a]) for a in actions},
            "n_updates": n_updates,
            "high_water_mark": high_water_mark,
            "update_chain_hash": update_chain_hash,
            "still_shadow": still_shadow,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _content_hash(**kwargs: object) -> str:
    return hashlib.sha256(_canonical_payload(**kwargs).encode("utf-8")).hexdigest()  # type: ignore[arg-type]


def fold_update_chain(
    prev_chain: str, *, action: str, reward: float, onehot: Sequence[int]
) -> str:
    """Order-sensitive hash fold of one observation onto the running chain."""

    key = f"{prev_chain}|{action}|{float(reward)!r}|{tuple(int(x) for x in onehot)!r}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def verify_online_state(state: OnlineState) -> bool:
    """Recompute the content hash and compare — detects corruption / hand edits."""

    recomputed = _content_hash(
        actions=state.actions,
        dim=state.dim,
        delta_A=state.delta_A_diag_per_action,
        delta_b=state.delta_b_per_action,
        delta_support=state.delta_support_per_action,
        n_updates=state.n_updates,
        high_water_mark=state.high_water_mark,
        update_chain_hash=state.update_chain_hash,
        still_shadow=state.still_shadow,
    )
    return recomputed == state.content_hash


# ---------------------------------------------------------------------------
# Accumulator protocol + the stdlib (default) implementation.
# ---------------------------------------------------------------------------


class OnlineAccumulator(Protocol):
    """A mutable online side-state. Every tier implements this."""

    @property
    def n_updates(self) -> int: ...

    @property
    def high_water_mark(self) -> int: ...

    def observe(self, *, onehot: Sequence[int], action: str, reward: float) -> None: ...

    def snapshot(self) -> OnlineState: ...


class StdlibLinUCBOnlineAccumulator:
    """Pure-stdlib diagonal-LinUCB online accumulator (the default tier).

    Accumulates the standard closed-form update for a one-hot context ``x``:
    ``ΔA_diag[i] += x_i²``, ``Δb[i] += reward·x_i``, ``Δsupport[i] += x_i`` on the
    chosen action's row. Deltas start at zero (the frozen artifact is the base)."""

    tier: str = ONLINE_TIER_STDLIB

    def __init__(self, *, dim: int, actions: Sequence[str]) -> None:
        if dim <= 0:
            raise OnlineStateError(f"dim must be positive; got {dim}")
        self._dim = int(dim)
        self._actions = tuple(actions)
        if len(set(self._actions)) != len(self._actions):
            raise OnlineStateError("actions must be unique")
        self._delta_A: dict[str, list[float]] = {a: [0.0] * self._dim for a in self._actions}
        self._delta_b: dict[str, list[float]] = {a: [0.0] * self._dim for a in self._actions}
        self._delta_support: dict[str, list[int]] = {a: [0] * self._dim for a in self._actions}
        self._n_updates = 0
        self._high_water = 0
        self._chain = ""

    @property
    def n_updates(self) -> int:
        return self._n_updates

    @property
    def high_water_mark(self) -> int:
        return self._high_water

    def observe(self, *, onehot: Sequence[int], action: str, reward: float) -> None:
        if action not in self._delta_A:
            raise OnlineStateError(f"action={action!r} not in {self._actions!r}")
        if len(onehot) != self._dim:
            raise OnlineStateError(f"onehot length {len(onehot)} != dim {self._dim}")
        r = float(reward)
        dA = self._delta_A[action]
        db = self._delta_b[action]
        ds = self._delta_support[action]
        for i, x in enumerate(onehot):
            if x == 0:
                continue
            if x != 1:
                raise OnlineStateError("onehot entries must be 0 or 1")
            dA[i] += 1.0
            db[i] += r
            ds[i] += 1
        self._n_updates += 1
        self._high_water = max(self._high_water, self._n_updates)
        # order-sensitive fold: any reordering / replay changes the chain head.
        self._chain = fold_update_chain(
            self._chain, action=action, reward=r, onehot=onehot
        )

    def snapshot(self) -> OnlineState:
        dA = {a: tuple(self._delta_A[a]) for a in self._actions}
        db = {a: tuple(self._delta_b[a]) for a in self._actions}
        ds = {a: tuple(self._delta_support[a]) for a in self._actions}
        content = _content_hash(
            actions=self._actions,
            dim=self._dim,
            delta_A=dA,
            delta_b=db,
            delta_support=ds,
            n_updates=self._n_updates,
            high_water_mark=self._high_water,
            update_chain_hash=self._chain,
            still_shadow=True,
        )
        return OnlineState(
            actions=self._actions,
            dim=self._dim,
            delta_A_diag_per_action=dA,
            delta_b_per_action=db,
            delta_support_per_action=ds,
            n_updates=self._n_updates,
            high_water_mark=self._high_water,
            update_chain_hash=self._chain,
            content_hash=content,
            still_shadow=True,
        )


def make_online_accumulator(
    tier: str, *, dim: int, actions: Sequence[str]
) -> OnlineAccumulator:
    """Return the online accumulator for ``tier``."""

    if tier == ONLINE_TIER_STDLIB:
        return StdlibLinUCBOnlineAccumulator(dim=dim, actions=actions)
    if tier == ONLINE_TIER_NUMPY:
        from ._online_numpy import NumpyLinUCBOnlineAccumulator

        return NumpyLinUCBOnlineAccumulator(dim=dim, actions=actions)
    if tier == ONLINE_TIER_TORCH:
        raise OnlineStateError(
            "runtime_tier='torch' online learner is not built yet — it is the heavy "
            "exploration tier constructed with the deferred training run (needs the "
            "seeded corpus + GPU). Use 'stdlib' (default) or 'numpy' until then."
        )
    raise OnlineStateError(f"unknown runtime tier {tier!r}; expected one of {ONLINE_TIERS!r}")


# ---------------------------------------------------------------------------
# Apply + stamp.
# ---------------------------------------------------------------------------


def apply_online_state(
    *,
    A_diag_per_action: Mapping[str, Sequence[float]],
    b_per_action: Mapping[str, Sequence[float]],
    feature_support_per_action: Mapping[str, Sequence[int]],
    state: OnlineState,
) -> tuple[
    dict[str, tuple[float, ...]],
    dict[str, tuple[float, ...]],
    dict[str, tuple[int, ...]],
]:
    """Add the online deltas onto a frozen policy's ``(A_diag, b, support)`` triple."""

    if not verify_online_state(state):
        raise OnlineStateError("online_state content hash mismatch; refusing to apply")
    actions = set(A_diag_per_action)
    if set(state.actions) != actions:
        raise OnlineStateError("online_state actions do not match the policy actions")
    new_A: dict[str, tuple[float, ...]] = {}
    new_b: dict[str, tuple[float, ...]] = {}
    new_support: dict[str, tuple[int, ...]] = {}
    for a in A_diag_per_action:
        base_A = A_diag_per_action[a]
        base_b = b_per_action[a]
        base_s = feature_support_per_action[a]
        if not (len(base_A) == len(base_b) == len(base_s) == state.dim):
            raise OnlineStateError(f"dim mismatch on action {a!r}")
        dA = state.delta_A_diag_per_action[a]
        db = state.delta_b_per_action[a]
        ds = state.delta_support_per_action[a]
        new_A[a] = tuple(base_A[i] + dA[i] for i in range(state.dim))
        new_b[a] = tuple(base_b[i] + db[i] for i in range(state.dim))
        new_support[a] = tuple(base_s[i] + ds[i] for i in range(state.dim))
    return new_A, new_b, new_support


def build_online_state_stamp(state: OnlineState | None) -> dict[str, object]:
    """Artifact-root honesty stamp for an online-mutated policy (empty for a pristine one)."""

    if state is None or not state.online_mutated:
        return {"online_mutated": False, "n_updates": 0, "still_shadow": True}
    return {
        "online_mutated": True,
        "n_updates": state.n_updates,
        "high_water_mark": state.high_water_mark,
        "last_update_hash": state.update_chain_hash,
        "content_hash": state.content_hash,
        "still_shadow": True,
    }


def is_online_mutated_artifact(blob: dict[str, object]) -> bool:
    """True iff an artifact blob carries a root ``online_state`` stamp with ``online_mutated``."""

    stamp = blob.get("online_state")
    return isinstance(stamp, dict) and bool(stamp.get("online_mutated"))


__all__ = (
    "ONLINE_TIER_STDLIB",
    "ONLINE_TIER_NUMPY",
    "ONLINE_TIER_TORCH",
    "ONLINE_TIERS",
    "OnlineState",
    "OnlineStateError",
    "OnlineAccumulator",
    "StdlibLinUCBOnlineAccumulator",
    "apply_online_state",
    "build_online_state_stamp",
    "fold_update_chain",
    "is_online_mutated_artifact",
    "make_online_accumulator",
    "verify_online_state",
)
