"""Define the guarded RL optimisation extension and runtime mode contract.

Provides the public RL mode contract, typed rejection for unsupported
RL-active modes, and the runtime admission gate. RL optimisation is currently
a contract-only extension; no policy implementation or online learning is
performed.

The RL layer is subordinate to safety, hardware/runtime constraints,
deterministic geometry, and deterministic recovery. It must not bypass safety
or telemetry, mutate parameters or rewards in-process, widen action bounds, or
suppress fallbacks. Policy changes follow the offline-training to promotion,
shadow, canary, and active lifecycle.
"""


from __future__ import annotations

from config.schema.robot.robot_schema import (
    RL_ACTIVE_MODES,
    RL_MODE_GEOMETRY_ONLY,
    RL_MODE_HYBRID_ML,
    RL_MODE_RL_ACTIVE,
    RL_MODE_RL_EXPERIMENTAL,
    RL_MODE_RL_SHADOW,
    RL_MODE_VALUES,
)


#: The newest mode the runtime admits. It admits the deterministic modes plus this one; anything
#: past it (``rl_active`` / ``rl_experimental``) is offline / not-yet-shipped and is rejected by
#: :func:`assert_rl_mode_supported`.
RL_NEWEST_RUNTIME_MODE: str = RL_MODE_RL_SHADOW

#: Version of the RL offline tooling (dataset builder, trainers, OPE, the promotion gate, and the
#: soak + rollback harnesses) shipped in this package.
RL_TOOLING_VERSION: str = "1.0"

#: The deterministic-only modes: no RL policy object is ever constructed for these.
RL_SUPPORTED_MODES_DETERMINISTIC: frozenset[str] = frozenset(
    {RL_MODE_GEOMETRY_ONLY, RL_MODE_HYBRID_ML}
)

#: The deterministic modes plus ``rl_shadow`` (log-only shadow routing; the deterministic stack is
#: unchanged when it is selected). This is the full set the runtime admits.
RL_SUPPORTED_MODES_WITH_SHADOW: frozenset[str] = (
    RL_SUPPORTED_MODES_DETERMINISTIC | {RL_MODE_RL_SHADOW}
)

#: Canonical "currently supported" frozenset (equal to the with-shadow set).
RL_SUPPORTED_MODES: frozenset[str] = RL_SUPPORTED_MODES_WITH_SHADOW


class RLModeNotImplementedError(RuntimeError):
    """Raised when an RL-active mode is selected before its producer ships."""

    def __init__(self, mode: str, *, producer: str) -> None:
        self.mode = mode
        self.producer = producer
        super().__init__(
            f"robot.rl.mode={mode!r} is schema-valid but its producer ({producer}) has not "
            f"shipped; the runtime admits {sorted(RL_SUPPORTED_MODES)!r}."
        )


#: What each RL-active mode is waiting on, in plain language, for the actionable admission-error
#: message. Consumed by :func:`assert_rl_mode_supported`.
RL_MODE_PRODUCER: dict[str, str] = {
    RL_MODE_RL_SHADOW: "the log-only shadow router",
    RL_MODE_RL_ACTIVE: "bounded online->canary active control",
    RL_MODE_RL_EXPERIMENTAL: "the experimental policy lane",
}


def assert_rl_mode_supported(mode: str) -> None:
    """Validate that the configured ``robot.rl.mode`` is runtime-supported.

    Called at boot as the single typed admission gate for RL modes. Schema
    validation ensures the configured value belongs to the locked enum; this
    helper additionally rejects RL-active modes whose policy producer has not
    shipped, preventing silent no-op degradation.

    Raises ``RLModeNotImplementedError`` for unsupported RL-active modes and
    ``ValueError`` for invalid enum values as a defense-in-depth check.
    """

    if mode not in RL_MODE_VALUES:
        raise ValueError(
            f"unknown robot.rl.mode={mode!r}; "
            f"expected one of {sorted(RL_MODE_VALUES)!r}"
        )
    if mode in RL_SUPPORTED_MODES:
        return
    producer = RL_MODE_PRODUCER.get(mode, "an unshipped producer")
    raise RLModeNotImplementedError(mode, producer=producer)


__all__ = (
    "RL_ACTIVE_MODES",
    "RL_MODE_GEOMETRY_ONLY",
    "RL_MODE_HYBRID_ML",
    "RL_MODE_PRODUCER",
    "RL_MODE_RL_ACTIVE",
    "RL_MODE_RL_EXPERIMENTAL",
    "RL_MODE_RL_SHADOW",
    "RL_MODE_VALUES",
    "RL_NEWEST_RUNTIME_MODE",
    "RL_SUPPORTED_MODES",
    "RL_SUPPORTED_MODES_DETERMINISTIC",
    "RL_SUPPORTED_MODES_WITH_SHADOW",
    "RL_TOOLING_VERSION",
    "RLModeNotImplementedError",
    "assert_rl_mode_supported",
)
