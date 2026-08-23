"""Deep health check for the motion stack's external engines it LOADS them, it does not guess.

:mod:`.environment`'s ``probe_*`` helpers answer *"is it wired?"* from paths alone, cheaply, at build
time. That is the right question for the driver and the wrong one for an operator: a present
interpreter can still fail to import, and on Windows a present, correct, unmodified binary can be
refused outright by an OS application-control policy. This module answers the other question *"does
it actually load, right now, on this box?"* by importing Coal in-process and spawning the cuRobo
sidecar's own interpreter.

Exit-code contract of the CLI that wraps this (``--doctor``): ``0`` all green, ``1`` degraded
(something is missing, a documented fallback takes over), ``2`` an OS policy is blocking a binary —
deliberately distinct, because the operator response is completely different.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from src.robot.constants import PLANNING_DOCTOR_LOG_FILE, create_robot_logger

from .environment import (
    ENV_COAL_PREFIX,
    ENV_CUROBO_PYTHON,
    collision_mesh_bundle,
    curobo_python_path,
    curobo_robot_config,
    import_collision_engine,
)

__all__ = [
    "ProbeStatus",
    "Probe",
    "DoctorReport",
    "code_integrity_blocks",
    "looks_policy_blocked",
    "run_doctor",
]

# The verdicts are printed once and then scroll away, which is exactly wrong for this module: the
# BLOCKED verdict is a property of a cloud reputation service and CHANGES OVER TIME, so "when did this
# box last see Coal load" is a question only a dated file can answer. Hence a persisted trail
# alongside the printed report same facts, kept.
logger = create_robot_logger("PlanningDoctor", PLANNING_DOCTOR_LOG_FILE)

#: Localized fragments of the OS "an application control policy blocked this file" message. Matching
#: prose is unavoidably locale-bound and therefore INCOMPLETE by construction it is a fast path only.
#: :func:`code_integrity_blocks` is the locale-independent authority (it reads the event log's
#: structured fields, not its rendered text), and :func:`looks_policy_blocked` consults both.
_BLOCK_MARKERS = (
    "blocked by an application control policy",   # en-US
    "anwendungssteuerungsrichtlinie",             # de-DE
)

#: Windows CodeIntegrity events: 3033 = did not meet the signing level, 3077 = blocked by policy.
_CI_EVENT_IDS = (3033, 3077)

_CI_QUERY = """
$ErrorActionPreference='SilentlyContinue'
$since = (Get-Date).AddSeconds(-{seconds})
Get-WinEvent -FilterHashtable @{{LogName='Microsoft-Windows-CodeIntegrity/Operational'; Id={ids}; StartTime=$since}} |
  ForEach-Object {{
    ([xml]$_.ToXml()).Event.EventData.Data |
      Where-Object {{ $_.Name -eq 'FileNameBuffer' }} |
      ForEach-Object {{ $_.'#text' }}
  }}
"""


def code_integrity_blocks(within_seconds: int = 120) -> tuple[str, ...]:
    """Files an OS code-integrity policy refused in the last *within_seconds*, newest-last, deduplicated."""
    if sys.platform != "win32":
        return ()
    script = _CI_QUERY.format(seconds=int(within_seconds), ids=",".join(str(i) for i in _CI_EVENT_IDS))
    try:
        done = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    seen: dict[str, None] = {}
    for line in done.stdout.splitlines():
        name = line.strip()
        if name:
            seen.setdefault(name, None)
    return tuple(seen)


def looks_policy_blocked(text: str, *, blocks: tuple[str, ...] = ()) -> bool:
    """True if *text* is an OS application-control refusal."""
    low = text.lower()
    if any(marker in low for marker in _BLOCK_MARKERS):
        return True
    return any(Path(blocked).name.lower() in low for blocked in blocks)


class ProbeStatus(StrEnum):
    """Outcome of one engine probe. ``BLOCKED`` is deliberately not a flavour of ``BROKEN``."""

    OK = "ok"
    WARN = "warn"         #: works, but with reduced redundancy, nothing is degraded *yet*
    BLOCKED = "blocked"   #: present and intact, but an OS policy refused to load it
    MISSING = "missing"   #: not installed on this box (a documented fallback takes over)
    BROKEN = "broken"     #: present, and failed for some other reason


@dataclass(frozen=True, slots=True)
class Probe:
    """One engine, one verdict, and when it failed, what to do about it."""

    name: str
    status: ProbeStatus
    detail: str
    remedy: str = ""

    @property
    def ok(self) -> bool:
        """True when nothing the stack needs is degraded. ``WARN`` counts as ok. See :class:`ProbeStatus`."""
        return self.status in (ProbeStatus.OK, ProbeStatus.WARN)


@dataclass(frozen=True, slots=True)
class DoctorReport:
    """Every probe, plus the exit code the CLI owes its caller."""

    probes: tuple[Probe, ...]
    blocked_files: tuple[str, ...]

    @property
    def blocked(self) -> bool:
        return any(p.status is ProbeStatus.BLOCKED for p in self.probes)

    @property
    def healthy(self) -> bool:
        return all(p.ok for p in self.probes)

    @property
    def exit_code(self) -> int:
        if self.blocked:
            return 2
        return 0 if self.healthy else 1

    def report(self) -> str:
        symbol = {
            ProbeStatus.OK: "ok     ",
            ProbeStatus.WARN: "warn   ",
            ProbeStatus.BLOCKED: "BLOCKED",
            ProbeStatus.MISSING: "missing",
            ProbeStatus.BROKEN: "BROKEN ",
        }
        lines = ["Workaholic-Willy motion-stack doctor (loads every engine)"]
        for probe in self.probes:
            lines.append(f"  [{symbol[probe.status]}] {probe.name}: {probe.detail}")
            if probe.remedy and not probe.ok:
                lines.append(f"              -> {probe.remedy}")
        if self.blocked_files:
            lines.append("")
            lines.append("  an OS code-integrity policy refused these files:")
            lines.extend(f"    {name}" for name in self.blocked_files)
            lines.append("  see docs/code-integrity.md the fix is to PIN the dependency to a build")
            lines.append("  that has reputation, not to re-download or move the install.")
        return "\n".join(lines)


# --------------------------------------------------------------------------- individual probes
def _probe_coal(blocks: tuple[str, ...]) -> Probe:
    """Import the exact-mesh engine the way production does, then run one real distance query."""
    prefix = os.environ.get(ENV_COAL_PREFIX) or "ext_deps/coal_env (default)"
    try:
        engine, backend = import_collision_engine()
    except Exception as exc:  # noqa: BLE001 - a probe reports failures, it does not raise them
        text = f"{type(exc).__name__}: {exc}"
        if looks_policy_blocked(text, blocks=blocks):
            return Probe("exact-mesh collision engine", ProbeStatus.BLOCKED, text[:200],
                         "pin the Coal env to a build with reputation (docs/code-integrity.md)")
        return Probe("exact-mesh collision engine", ProbeStatus.BROKEN, text[:200], "see docs/coal-setup.md")
    if engine is None:
        return Probe(
            "exact-mesh collision engine", ProbeStatus.MISSING,
            f"neither Coal nor python-fcl importable (prefix: {prefix})",
            "install Coal docs/coal-setup.md. Until then the guard uses its capsule proxy.",
        )
    try:
        import numpy as np

        box, sphere = engine.Box(1.0, 1.0, 1.0), engine.Sphere(0.5)
        here, there = engine.Transform3s(), engine.Transform3s()
        there.setTranslation(np.array([3.0, 0.0, 0.0]))
        request, result = engine.DistanceRequest(), engine.DistanceResult()
        engine.distance(box, here, sphere, there, request, result)
        distance = float(result.min_distance)
    except Exception as exc:  # noqa: BLE001
        return Probe("exact-mesh collision engine", ProbeStatus.BROKEN,
                     f"{backend} imported but a distance query failed: {type(exc).__name__}: {exc}"[:200],
                     "see docs/coal-setup.md")
    version = getattr(engine, "__version__", "?")
    return Probe("exact-mesh collision engine", ProbeStatus.OK,
                 f"{backend} {version} (prefix: {prefix}) -- box<->sphere = {distance:.6f}")


def _probe_mesh_bundle(model: str) -> Probe:
    """The per-link collision meshes the guard and the cuRobo sphere fit both key on this robot's name."""
    bundle = collision_mesh_bundle(model)
    if bundle.is_file():
        return Probe(f"collision mesh bundle ({model})", ProbeStatus.OK, str(bundle))
    return Probe(
        f"collision mesh bundle ({model})", ProbeStatus.MISSING, f"absent: {bundle}",
        f"the guard falls back to capsules for {model}; build or commit the bundle",
    )


#: Run inside the sidecar's OWN interpreter. Reports what actually resolved there, including whether a
#: SECOND kernel backend exists a single backend is exactly the single point of failure that took the
#: planner down on 2026-08-10, so the doctor treats "only one" as worth saying out loud.
_CUROBO_PROBE = """
import json, os, sys
out = {"python": sys.executable}
try:
    import torch
    out["torch"] = torch.__version__
    out["cuda"] = bool(torch.cuda.is_available())
except Exception as exc:
    out["error"] = "torch: %s: %s" % (type(exc).__name__, exc)
    print(json.dumps(out)); raise SystemExit(0)
try:
    import curobo
    from curobo._src.curobolib.backends import get_backend_name
    out["curobo"] = getattr(curobo, "__version__", "?")
    out["backend"] = get_backend_name()
except Exception as exc:
    out["error"] = "curobo: %s: %s" % (type(exc).__name__, exc)
    print(json.dumps(out)); raise SystemExit(0)
backends, failures = [], {}
import curobo._src.runtime as rt
for name in ("cuda_core", "pybind"):
    try:
        rt.kernel_backend = name
        if get_backend_name() == name:
            backends.append(name)
        else:
            # cuRobo silently falls back rather than raising, so "asked for X, got Y" IS the failure.
            # Re-import the backing package to recover the REASON, which is what tells an operator
            # whether this is "not installed" or "the OS refused it".
            try:
                if name == "cuda_core":
                    import cuda.core.experimental  # noqa: F401
                else:
                    import curobo._src.curobolib.geom_cu  # noqa: F401
                failures[name] = "resolved to %s instead" % get_backend_name()
            except Exception as inner:
                failures[name] = "%s: %s" % (type(inner).__name__, inner)
    except Exception as exc:
        failures[name] = "%s: %s" % (type(exc).__name__, exc)
out["backends"] = backends
out["backend_failures"] = failures
try:
    from curobo.content import get_content_root
    descriptor = os.path.join(str(get_content_root()), "configs", "robot", os.environ["WILLY_DOCTOR_ROBOT"])
    out["descriptor"] = descriptor
    out["descriptor_present"] = os.path.isfile(descriptor)
except Exception as exc:
    out["descriptor_error"] = "%s: %s" % (type(exc).__name__, exc)
print(json.dumps(out))
"""


#: cuRobo's two kernel backends and what each one costs to obtain, for the remedy text.
_BACKEND_REMEDY = {
    "cuda_core": "install the JIT backend: pip install 'cuda-core[cu12]' into the cuRobo env "
                 "(docs/curobo-setup.md 2a)",
    "pybind": "build the compiled backend: docs/curobo/build_compiled_backend.bat "
              "(docs/curobo-setup.md 2b)",
}


def _kernel_backend_probe(payload: dict[str, object], blocks: tuple[str, ...]) -> Probe:
    """How much redundancy the planner's CUDA kernels actually have on this box."""
    # The payload crossed a subprocess boundary as JSON, so nothing about its shape is guaranteed --
    # narrow both fields by type rather than trusting them.
    raw_backends = payload.get("backends")
    backends = tuple(str(b) for b in raw_backends) if isinstance(raw_backends, list) else ()
    raw_failures = payload.get("backend_failures")
    failures = {str(k): str(v) for k, v in raw_failures.items()} if isinstance(raw_failures, dict) else {}

    if len(backends) >= 2:
        return Probe("cuRobo kernel backends", ProbeStatus.OK,
                     f"{len(backends)} independent backends: {', '.join(backends)}")
    if not backends:
        return Probe("cuRobo kernel backends", ProbeStatus.BROKEN,
                     f"none resolved ({'; '.join(f'{k}: {v}' for k, v in failures.items()) or 'no detail'})"[:200],
                     "; ".join(_BACKEND_REMEDY.values()))

    working = backends[0]
    absent = [name for name in ("cuda_core", "pybind") if name != working]
    missing = absent[0] if absent else "the other backend"
    reason = failures.get(missing, "not installed")
    if looks_policy_blocked(reason, blocks=blocks):
        return Probe(
            "cuRobo kernel backends", ProbeStatus.WARN,
            f"1 of 2 '{missing}' refused by an application-control policy; '{working}' is carrying "
            f"the planner. The redundancy is doing its job; you are now one verdict from an outage.",
            "docs/code-integrity.md pin that dependency to a build with reputation, and keep "
            f"'{working}' installed",
        )
    return Probe(
        "cuRobo kernel backends", ProbeStatus.WARN,
        f"1 of 2 only '{working}' resolves ({missing}: {reason})"[:200],
        _BACKEND_REMEDY.get(missing, "see docs/curobo-setup.md"),
    )


def _probe_curobo(blocks: tuple[str, ...], robot_config: str) -> tuple[Probe, ...]:
    """Spawn the sidecar's interpreter and report what it can actually import.

    This is the check ``--check`` explicitly cannot make: the descriptor lives *inside* a separate
    environment, so the only honest way to verify it is to ask that environment.
    """
    python = curobo_python_path()
    if not Path(python).exists():
        return (Probe(
            "cuRobo planner sidecar", ProbeStatus.MISSING, f"no interpreter at {python}",
            f"install it (docs/curobo-setup.md) or set {ENV_CUROBO_PYTHON}; "
            "until then the driver plans with blind IK",
        ),)
    env = dict(os.environ, WILLY_DOCTOR_ROBOT=robot_config, PYTHONIOENCODING="utf-8")
    # Importing torch + cuRobo in a cold interpreter costs tens of seconds; the timeout allows 300.
    logger.info("spawning the cuRobo sidecar interpreter %s (robot config %s, timeout 300s)", python, robot_config)
    try:
        done = subprocess.run([python, "-c", _CUROBO_PROBE], capture_output=True, text=True,
                              timeout=300, check=False, env=env)
    except (OSError, subprocess.SubprocessError) as exc:
        return (Probe("cuRobo planner sidecar", ProbeStatus.BROKEN,
                      f"could not run {python}: {type(exc).__name__}: {exc}"[:200],
                      "see docs/curobo-setup.md"),)

    # The sidecar prints one JSON object; cuRobo and torch print banners around it, so take the LAST
    # parseable line rather than assuming the payload is alone on stdout.
    payload: dict[str, object] = {}
    for line in done.stdout.splitlines():
        if line.startswith("{"):
            try:
                payload = json.loads(line)
            except ValueError:
                pass
    if not payload:
        text = (done.stderr or done.stdout or "no output").strip()
        status = (ProbeStatus.BLOCKED if looks_policy_blocked(text, blocks=blocks) else ProbeStatus.BROKEN)
        return (Probe("cuRobo planner sidecar", status, text[-200:],
                      "docs/code-integrity.md" if status is ProbeStatus.BLOCKED else "docs/curobo-setup.md"),)

    error = str(payload.get("error", ""))
    if error:
        status = ProbeStatus.BLOCKED if looks_policy_blocked(error, blocks=blocks) else ProbeStatus.BROKEN
        return (Probe("cuRobo planner sidecar", status, error[:200],
                      "docs/code-integrity.md" if status is ProbeStatus.BLOCKED else "docs/curobo-setup.md"),)

    detail = (f"curobo {payload.get('curobo', '?')} on torch {payload.get('torch', '?')} "
              f"(cuda={payload.get('cuda')}), kernel backend = {payload.get('backend', '?')}")
    probes = [Probe("cuRobo planner sidecar", ProbeStatus.OK, detail)]

    probes.append(_kernel_backend_probe(payload, blocks))

    if "descriptor_present" in payload:
        present = bool(payload["descriptor_present"])
        where = str(payload.get("descriptor", "?"))
        probes.append(Probe(
            f"cuRobo robot descriptor ({robot_config})",
            ProbeStatus.OK if present else ProbeStatus.MISSING,
            where,
            "" if present else "build it: docs/curobo-setup.md section 3 (it lives inside the clone, "
                               "so a fresh clone needs this again)",
        ))
    return tuple(probes)


def run_doctor(*, model: str = "ur5e", robot_config: str | None = None) -> DoctorReport:
    """Load every external engine and report what this box can actually do.

    The event log is read ONCE, before the probes, so a block that happens *during* a probe is still in
    the window when the probes classify their own failures.
    """
    blocks = code_integrity_blocks()
    logger.info("motion-stack doctor starting for model %r (robot config %s)", model, robot_config or "<default>")
    probes = (
        _probe_coal(blocks),
        _probe_mesh_bundle(model),
        *_probe_curobo(blocks, robot_config or curobo_robot_config()),
    )
    # Levels follow what the operator must DO: a BLOCKED/BROKEN engine is a returned failure (nothing
    # raises here), MISSING and WARN mean a documented fallback or lost redundancy is carrying the box.
    for probe in probes:
        if probe.status is ProbeStatus.OK:
            logger.info("%s: ok -- %s", probe.name, probe.detail)
        elif probe.status in (ProbeStatus.BLOCKED, ProbeStatus.BROKEN):
            logger.error("%s: %s -- %s (remedy: %s)", probe.name, probe.status.value, probe.detail,
                         probe.remedy or "none given")
        else:
            logger.warning("%s: %s -- %s (remedy: %s)", probe.name, probe.status.value, probe.detail,
                           probe.remedy or "none given")
    # Report only the blocked files a probe could plausibly be about. The log is machine-wide and
    # would otherwise drag in unrelated software the operator cannot act on.
    after = code_integrity_blocks()
    relevant = tuple(
        name for name in dict.fromkeys(blocks + after)
        if any(token in name.lower() for token in ("curobo", "coal", "aurora", "willy", "ext_deps", ".venv"))
    )
    report = DoctorReport(probes=probes, blocked_files=relevant)
    if relevant:
        logger.error("an OS code-integrity policy refused: %s", ", ".join(relevant))
    logger.info(
        "motion-stack doctor finished: %d/%d probes ok, exit code %d",
        sum(1 for p in probes if p.ok), len(probes), report.exit_code,
    )
    return report
