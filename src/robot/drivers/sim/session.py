"""Lifecycle wrapper around the Isaac Sim application."""

from __future__ import annotations

import os
import sys
from typing import Any

from ...core import IsaacNotAvailableError
from .config import SimRobotConfig

__all__ = ["IsaacSimSession"]


class IsaacSimSession:
    """Manage the Isaac Sim application for one :class:`IsaacRobotArm`.

    Owns the config, the ``SimulationApp`` handle and the ``World``, and centralises the
    SDK import so :class:`IsaacRobotArm` never imports Isaac directly. ``SimulationApp`` is
    a process singleton, so there must be exactly one live session per process.
    """

    def __init__(self, config: SimRobotConfig) -> None:
        self._config = config
        self._app: Any | None = None
        self._world: Any | None = None
        self._running = False

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def config(self) -> SimRobotConfig:
        return self._config

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def app(self) -> Any | None:
        """The live ``SimulationApp`` handle (``None`` until :meth:`start`, or in mock mode)."""
        return self._app

    @property
    def world(self) -> Any | None:
        """The live Isaac ``World`` (``None`` until :meth:`start`, or in mock mode)."""
        return self._world

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Boot the Isaac application (idempotent).

        Behaviour depends on :attr:`SimRobotConfig.mock_mode`:

        * **mock_mode=True** (default for CI / macOS / any downstream project without an
          Isaac host): no SDK import is attempted; the session marks itself running and
          returns. Combined with :class:`IsaacRobotArm`'s mock motion path this is a
          deterministic pure-Python driver that satisfies the typed motion contract.
        * **mock_mode=False**: a real headless ``SimulationApp`` is booted, the configured
          scene (or a default empty stage) is opened, and a ``World`` is created + reset.

        Raises
        ------
        IsaacNotAvailableError
            When ``mock_mode=False`` and the Isaac SDK is not importable on this host.
        """
        if self._running:
            return  # idempotent

        if self._config.mock_mode:
            # Pure-Python path — deliberately never touch Isaac.
            self._running = True
            return

        # ``OMNI_KIT_ACCEPT_EULA`` must be set before the first isaacsim import or the
        # headless boot can block waiting for EULA acceptance.
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        try:
            # Lazy SDK import (Isaac Sim 5.1 namespace; ``omni.isaac.kit`` was removed in
            # 4.5). Kept inside the method so module import stays safe off-workstation.
            from isaacsim import SimulationApp  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - covered on non-Isaac hosts
            raise IsaacNotAvailableError(
                "Isaac Sim SDK is not importable on this host. The RobotVendor.SIM "
                "driver's non-mock path requires NVIDIA Isaac Sim 5.1 (run with its "
                "bundled python). Set SimRobotConfig(mock_mode=True) for the pure-Python "
                "mock kinematics path. See docs/ISAAC_VALIDATION_PLATFORM.md."
            ) from exc

        # ``SimulationApp`` forwards ``sys.argv`` to Omniverse Kit's own parser, which
        # rejects foreign flags (e.g. pytest's ``-m``) and then crashes. Strip argv to the
        # program name across the boot, then restore it.
        saved_argv, sys.argv = sys.argv, sys.argv[:1]
        try:
            self._app = SimulationApp({"headless": self._config.headless})
        finally:
            sys.argv = saved_argv

        # Deferred imports: only valid once the app has loaded its extensions.
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import create_new_stage, open_stage

        if self._config.scene:
            open_stage(self._config.scene)
        else:
            create_new_stage()

        self._world = World(
            stage_units_in_meters=1.0,
            physics_dt=self._config.step_dt_s,
            rendering_dt=self._config.step_dt_s,
        )
        self._world.reset()
        self._running = True

    def step(self, dt_s: float | None = None, *, render: bool = False) -> None:
        """Advance the simulation by one step.

        The physics / rendering ``dt`` is fixed at :attr:`SimRobotConfig.step_dt_s` (set on
        the ``World`` at :meth:`start`); ``dt_s`` is accepted for interface symmetry and
        currently ignored. No-op in mock mode or before :meth:`start`.
        """
        if not self._running or self._world is None:
            return
        self._world.step(render=render)

    def step_n(self, count: int, *, render: bool = False) -> None:
        """Drive a fixed number of :meth:`step` calls (most callers want this)."""
        for _ in range(max(0, count)):
            self.step(render=render)

    def stop(self) -> None:
        """Shut down the Isaac application if running. Safe to call repeatedly."""
        if self._app is not None:
            self._app.close()
        self._app = None
        self._world = None
        self._running = False
