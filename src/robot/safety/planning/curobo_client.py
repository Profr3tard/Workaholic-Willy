"""Client for the process-isolated cuRobo planning server (the py3.11 side of the seam)."""

from __future__ import annotations

import atexit
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from src.robot.constants import CUROBO_CLIENT_LOG_FILE, create_robot_logger

from ._curobo_margin import ENV_SELF_COLLISION_MARGIN_MM
from .environment import (
    ENV_CUROBO_STDERR,
    curobo_cuboid_cache,
    curobo_env_available,
    curobo_python_path,
    curobo_robot_config,
)

__all__ = ["CuroboPlanClient", "CuroboUnavailableError", "curobo_env_available"]

# The server file ships in this package but is run by the EXTERNAL cuRobo python, never imported here.
_SERVER_SCRIPT = str(Path(__file__).with_name("curobo_planner_server.py"))

# Logging config for curobo_client.py.
logger = create_robot_logger("CuroboPlanClient", CUROBO_CLIENT_LOG_FILE)

# Generous: the server JIT-warms cuRobo's kernels on first boot (cold ~25 s, kernel-cache-warm ~8 s).
_READY_TIMEOUT_S = 120.0
_PLAN_TIMEOUT_S = 30.0


class CuroboUnavailableError(RuntimeError):
    """The cuRobo planning server could not be started or became ready (env missing, JIT/load failure)."""


def _raise_if_call_failed(msg: dict) -> None:
    """Turn a FAILED CALL into an exception; leave a genuine "no solution" as ``None``.

    These two are not the same thing and must never look the same, which is the whole reason this
    function exists. "The planner searched and found nothing" is a verdict a caller can act on: fail
    safe, do not move. "The call raised" is a bug, and returning ``None`` for it dresses a bug up as a
    verdict about the robot.
    """
    if msg.get("planner_error"):
        raise CuroboUnavailableError(
            f"the cuRobo planning CALL failed (not a planning verdict): {msg.get('reason')}"
        )


class CuroboPlanClient:
    """Spawns + drives the warm cuRobo planning server. ``plan`` returns a joint trajectory or ``None``."""

    def __init__(
        self,
        *,
        python_path: str | None = None,
        server_script: str | None = None,
        robot_config: str | None = None,
        scene_config: str | None = None,
        stderr_log: str | None = None,
        self_collision_margin_mm: float = 0.0,
    ) -> None:
        # Every path/knob resolves through safety.planning.environment (the single anchor), so a caller
        # can override just what it needs and the rest tracks the env vars documented there.
        self._python = python_path or curobo_python_path()
        self._script = server_script or _SERVER_SCRIPT
        self._robot = robot_config or curobo_robot_config()
        self._scene = scene_config or curobo_cuboid_cache()
        # The clearance the SAFETY GUARD will demand of the plan's final configuration. Handed to the
        # sidecar so cuRobo plans WITH that margin instead of returning paths the guard then refuses
        # (measured: 9.44-9.47 mm plans against a 10.000 mm guard margin). 0.0 -> untouched config.
        self._self_collision_margin_mm = float(self_collision_margin_mm)
        # server stderr (cuRobo warmup/plan diagnostics) -> a log file when requested (param or
        # WILLY_CUROBO_STDERR), else discarded. Useful for on-box debugging of the isolated server.
        self._stderr_log = stderr_log or os.environ.get(ENV_CUROBO_STDERR)
        self._proc: subprocess.Popen[str] | None = None
        self._q: "queue.Queue[dict | None]" = queue.Queue()
        self._reader: threading.Thread | None = None
        self.joint_names: list[str] = []
        self.dt: float = 0.0

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        """Spawn the server and block until it reports ``ready`` (raises CuroboUnavailableError otherwise)."""
        if self._proc is not None:
            return
        if not Path(self._python).exists():
            raise CuroboUnavailableError(f"cuRobo env python not found: {self._python}")
        logger.info(
            "spawning the cuRobo sidecar: python=%s robot=%s cuboid_cache=%s self_collision_margin=%.3f mm",
            self._python, self._robot, self._scene, self._self_collision_margin_mm,
        )
        started = time.monotonic()
        stderr = open(self._stderr_log, "w") if self._stderr_log else subprocess.DEVNULL  # noqa: SIM115
        env = dict(os.environ)
        if self._self_collision_margin_mm > 0.0:
            env[ENV_SELF_COLLISION_MARGIN_MM] = repr(self._self_collision_margin_mm)
        self._proc = subprocess.Popen(
            [self._python, "-u", self._script, self._robot, self._scene],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, text=True, bufsize=1,
            env=env,
        )
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()
        atexit.register(self.close)
        msg = self._recv(_READY_TIMEOUT_S)
        if not msg or msg.get("status") != "ready":
            reason = (msg or {}).get("reason", "no ready signal (timeout or server died)")
            self.close()
            raise CuroboUnavailableError(f"cuRobo server did not become ready: {reason}")
        self.joint_names = list(msg["joint_names"])
        self.dt = float(msg.get("dt", 0.0))
        # The warm-up cost is worth recording: cold ~25 s vs kernel-cache-warm ~8 s is the difference
        # between "the planner is slow" and "the kernel cache was thrown away".
        logger.info(
            "cuRobo sidecar ready after %.1f s: %d joint(s), dt=%.4f s",
            time.monotonic() - started, len(self.joint_names), self.dt,
        )

    def _pump(self) -> None:
        """Reader thread: push each JSON line from the server's stdout onto the queue; ``None`` on EOF."""
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                line = line.strip()
                if line.startswith("{"):
                    try:
                        self._q.put(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        except (ValueError, OSError):  # stream closed under us during shutdown -> stop quietly
            pass
        self._q.put(None)  # EOF / server exit

    def _recv(self, timeout_s: float) -> dict | None:
        try:
            return self._q.get(timeout=timeout_s)
        except queue.Empty:
            return None

    def _send(self, obj: dict) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise CuroboUnavailableError("cuRobo server is not running")
        self._proc.stdin.write(json.dumps(obj) + "\n")
        self._proc.stdin.flush()

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                proc.stdin.flush()
            proc.wait(timeout=5)
            logger.info("cuRobo sidecar shut down cleanly")
        except Exception as exc:  # noqa: BLE001 - shutdown is best-effort
            logger.warning("cuRobo sidecar did not shut down cleanly (%s); killing it", exc)
            proc.kill()

    # -- planning ----------------------------------------------------------
    def plan(
        self,
        start_joints: list[float],
        goal_pos_m: list[float],
        goal_quat_wxyz: list[float],
    ) -> list[list[float]] | None:
        """Plan tool0 -> ``goal`` (metres, WXYZ, base frame) from ``start_joints`` (rad).

        Returns the interpolated joint trajectory ``[[6 rad], ...]`` in :attr:`joint_names` order, or ``None``
        if cuRobo found no collision-free solution.
        """
        if self._proc is None:
            self.start()
        started = time.monotonic()
        self._send({"start_joints": list(start_joints), "goal_pos_m": list(goal_pos_m),
                    "goal_quat_wxyz": list(goal_quat_wxyz)})
        msg = self._recv(_PLAN_TIMEOUT_S)
        if msg is None:
            raise CuroboUnavailableError("cuRobo server stopped responding (timeout / died)")
        if msg.get("success"):
            traj: list[list[float]] = msg["trajectory"]
            logger.debug(
                "planned to (%.3f, %.3f, %.3f) m in %.0f ms: %d waypoint(s)",
                goal_pos_m[0], goal_pos_m[1], goal_pos_m[2],
                (time.monotonic() - started) * 1000.0, len(traj),
            )
            return traj
        _raise_if_call_failed(msg)
        # A VERDICT, not a failure but the caller turns it into "no motion", so the reason a pick
        # stopped is only ever recoverable from here.
        logger.warning(
            "cuRobo found NO collision-free plan to (%.3f, %.3f, %.3f) m after %.0f ms: %s",
            goal_pos_m[0], goal_pos_m[1], goal_pos_m[2],
            (time.monotonic() - started) * 1000.0, msg.get("reason", "no reason given"),
        )
        return None

    def plan_joint(
        self, start_joints: list[float], goal_joints: list[float]
    ) -> list[list[float]] | None:
        """Plan ``start_joints`` -> ``goal_joints`` (rad) collision-free, or ``None`` if it cannot.

        The joint-space twin of :meth:`plan`, for goals that ARE joint configurations, park and home
        are stored that way.

        ``None`` means "no collision-free plan", and the caller must fail safe rather than move blindly
        which is the whole reason a caller would ask for a plan instead of interpolating.
        """
        if self._proc is None:
            self.start()
        started = time.monotonic()
        self._send({"cmd": "plan_js", "start_joints": list(start_joints),
                    "goal_joints": list(goal_joints)})
        msg = self._recv(_PLAN_TIMEOUT_S)
        if msg is None:
            raise CuroboUnavailableError("cuRobo server stopped responding (timeout / died)")
        if msg.get("success"):
            traj: list[list[float]] = msg["trajectory"]
            logger.debug(
                "planned a joint move in %.0f ms: %d waypoint(s)",
                (time.monotonic() - started) * 1000.0, len(traj),
            )
            return traj
        _raise_if_call_failed(msg)
        logger.warning(
            "cuRobo found NO collision-free joint plan after %.0f ms: %s",
            (time.monotonic() - started) * 1000.0, msg.get("reason", "no reason given"),
        )
        return None

    def set_world(self, cuboids: list[dict]) -> int:
        """Replace cuRobo's collision world with ``cuboids`` (the scene->planner world-model).

        Each cuboid is ``{"name": str, "dims_m": [x,y,z], "pose": [px,py,pz,qw,qx,qy,qz]}`` (base frame,
        metres, quaternion WXYZ). Returns the count registered, or 0 on failure.
        """
        if self._proc is None:
            self.start()
        self._send({"cmd": "set_world", "cuboids": list(cuboids)})
        msg = self._recv(_PLAN_TIMEOUT_S)
        if msg and msg.get("world_set") is not None:
            count = int(msg["world_set"])
            # An obstacle the planner never received is an obstacle it will happily route through, so
            # the count REGISTERED (not the count sent) is the number that matters.
            logger.info("collision world set: %d of %d cuboid(s) registered", count, len(cuboids))
            return count
        logger.error(
            "the cuRobo sidecar did not confirm the collision world (%d cuboid(s) sent); planning "
            "continues against the PREVIOUS world",
            len(cuboids),
        )
        return 0

    def fk(self, joints: list[float]) -> tuple[list[float], list[float]] | None:
        """Tool0 FK of ``joints`` (rad) -> (pos_m, quat_wxyz); ``None`` on failure. Used for validation."""
        if self._proc is None:
            self.start()
        self._send({"cmd": "fk", "joints": list(joints)})
        msg = self._recv(_PLAN_TIMEOUT_S)
        if msg and msg.get("fk_pos_m") is not None:
            return msg["fk_pos_m"], msg["fk_quat_wxyz"]
        return None

    def __enter__(self) -> "CuroboPlanClient":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
