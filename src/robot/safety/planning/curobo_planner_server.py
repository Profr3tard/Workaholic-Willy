"""cuRobo planning SERVER a process-isolated sidecar.

WHY a separate process: cuRobo needs warp ~1.14, Isaac ships warp 1.8.2, and one Python process can hold only
one ``warp``, so cuRobo CANNOT run in the Isaac py3.11 process. The client
(:class:`src.robot.safety.planning.curobo_client.CuroboPlanClient`) spawns THIS script with the cuRobo
env's python and talks newline-delimited JSON over stdio. The planner + kinematics + JIT kernels load ONCE
(``warmup``), then the server serves many plan requests (a per-call subprocess would re-warm = far too slow).

This file is py3.10 / cuRobo code and is never imported by the py3.11 Willy package (its cuRobo imports are
``type: ignore``-d for the project mypy, which has no cuRobo). Run it via the client, or standalone:

    <cuRobo-env-python> -m ... curobo_planner_server.py [robot.yml] [scene.yml]

Protocol (one JSON object per line):
  startup -> {"status":"ready","joint_names":[...],"default_q":[...],"dt":float,"start_pos_m":[...],
              "start_quat_wxyz":[...]}   |   {"status":"error","reason":str}
  request <- {"start_joints":[6 rad],"goal_pos_m":[x,y,z],"goal_quat_wxyz":[w,x,y,z]}
             {"cmd":"fk","joints":[6 rad]}   |   {"cmd":"shutdown"}
  reply   -> {"success":bool,"trajectory":[[6 rad]...],"dt":float}  |  {"success":false,"reason":str}
             {"fk_pos_m":[...],"fk_quat_wxyz":[...]}

Goal pose is the TOOL0 (Lula EE) pose in METRES, base frame, quaternion WXYZ — the caller maps its grasp TCP
to tool0 + units/quaternion before sending. The trajectory is returned in ``joint_names`` order.
"""
from __future__ import annotations

import json
import os
import sys

ROBOT = sys.argv[1] if len(sys.argv) > 1 else "ur5e.yml"
# Reliability for CONSTRAINED queries (tight bins): cuRobo's plan_pose is stochastic (random IK/trajopt seeds) and
# default max_attempts=5 is FLAKY on a hard pick, a collision-free plan provably exists (measured: same query
# NONE then OK across runs) but the seeds sometimes miss it. MORE attempts (each a fresh IK/trajopt seed batch)
# reliably find it. Keep enable_graph_attempt=1 (cuRobo's default): the FIRST attempt is trajopt-only (the graph
# seeder can return no seed for a tight final approach and would otherwise skip every attempt). Env-overridable.
_PLAN_MAX_ATTEMPTS = int(os.environ.get("WILLY_CUROBO_MAX_ATTEMPTS", "16"))
_PLAN_GRAPH_FROM = int(os.environ.get("WILLY_CUROBO_GRAPH_FROM_ATTEMPT", "1"))
# cuRobo sizes its collision-world cache to the INITIAL scene's cuboid count, so we boot with this many
# far-away placeholder cuboids to RESERVE that many slots; `set_world` later replaces them with the real
# obstacles (the bin walls + neighbour bodies). argv[2] overrides the reserved count.
CUBOID_CACHE = int(sys.argv[2]) if len(sys.argv) > 2 else 16


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


try:
    import torch  # type: ignore[import-not-found]
    from curobo._src.geom.types import SceneCfg  # type: ignore[import-not-found]
    from curobo.kinematics import Kinematics, KinematicsCfg  # type: ignore[import-not-found]
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg  # type: ignore[import-not-found]
    from curobo.types import GoalToolPose, JointState  # type: ignore[import-not-found]

    # Boot world = a real TABLE (load-bearing: without a floor cuRobo plans a contorted park->pre-grasp path
    # that ends far from the goal) + (CUBOID_CACHE-1) far-away placeholder cuboids to RESERVE collision slots
    # that `set_world` can later fill with the scene's obstacles.
    _world = {"table": {"dims": [1.6, 1.6, 0.05], "pose": [0.0, 0.0, -0.026, 1, 0, 0, 0]}}
    _world.update({f"rsv{i}": {"dims": [0.01, 0.01, 0.01], "pose": [0.0, 0.0, -100.0 - i, 1, 0, 0, 0]}
                   for i in range(max(0, CUBOID_CACHE - 1))})
    # Teach the planner the clearance the SAFETY GUARD will demand, so it stops returning paths the
    # guard was always going to refuse.
    from _curobo_margin import (  # type: ignore[import-not-found]
        ENV_SELF_COLLISION_MARGIN_MM,
        derive_margin_config_file,
    )

    _ROBOT_IN_USE = ROBOT
    _margin_mm = float(os.environ.get(ENV_SELF_COLLISION_MARGIN_MM, "0") or 0.0)
    if _margin_mm > 0.0:
        import tempfile

        from curobo.content import get_content_root  # type: ignore[import-not-found]

        _src = ROBOT if os.path.isabs(ROBOT) else os.path.join(
            str(get_content_root()), "configs", "robot", ROBOT
        )
        _dst = os.path.join(tempfile.gettempdir(), f"willy_guard{_margin_mm:g}mm_{os.path.basename(_src)}")
        _n = derive_margin_config_file(_src, _dst, _margin_mm)
        print(f"[margin] +{_margin_mm:g} mm guard clearance on {_n} links -> {_dst}", file=sys.stderr, flush=True)
        if _n:
            _ROBOT_IN_USE = _dst
        else:
            # Loud, not silent: the planner is about to run WITHOUT the guard's margin, so it can still
            # hand back configurations the guard refuses.
            print(f"[margin] !! {_src} has no self_collision_buffer block: margin NOT applied",
                  file=sys.stderr, flush=True)
    _planner = MotionPlanner(MotionPlannerCfg.create(robot=_ROBOT_IN_USE, scene_model={"cuboid": _world}))
    _planner.warmup(enable_graph=True, num_warmup_iterations=5)
    _DT = float(_planner.trajopt_solver.config.interpolation_dt)
    _N = len(_planner.joint_names)
    _kin = Kinematics(KinematicsCfg.from_robot_yaml_file(_ROBOT_IN_USE))
    _default_q = _planner.default_joint_state.position.squeeze().cpu().tolist()

    def _fk(joints: list) -> tuple[list, list]:
        q = torch.tensor(joints, device="cuda", dtype=torch.float32).reshape(1, -1)
        st = _kin.compute_kinematics(JointState.from_position(q, joint_names=_kin.joint_names))
        p = st.tool_poses.get_link_pose(_kin.tool_frames[0])
        return p.position.squeeze().cpu().tolist(), p.quaternion.squeeze().cpu().tolist()

except Exception as exc:  # noqa: BLE001 - any import/load/JIT failure -> a typed error line, then exit
    _emit({"status": "error", "reason": f"{type(exc).__name__}: {exc}"})
    sys.exit(1)

_sp, _sq = _fk(_default_q)
_emit({"status": "ready", "joint_names": list(_planner.joint_names), "default_q": _default_q,
       "dt": _DT, "start_pos_m": _sp, "start_quat_wxyz": _sq})

for _line in sys.stdin:
    _line = _line.strip()
    if not _line:
        continue
    try:
        req = json.loads(_line)
    except Exception as exc:  # noqa: BLE001
        _emit({"success": False, "planner_error": True, "reason": f"bad json: {exc}"})
        continue
    cmd = req.get("cmd")
    if cmd == "shutdown":
        break
    if cmd == "fk":
        try:
            p, q = _fk(req["joints"])
            _emit({"fk_pos_m": p, "fk_quat_wxyz": q})
        except Exception as exc:  # noqa: BLE001
            _emit({"fk_pos_m": None, "reason": f"{type(exc).__name__}: {exc}"})
        continue
    if cmd == "plan_js":
        # Plan to a JOINT configuration, collision-free (cuRobo calls it plan_cspace).
        try:
            q0 = torch.tensor(req["start_joints"], device="cuda", dtype=torch.float32).unsqueeze(0)
            qg = torch.tensor(req["goal_joints"], device="cuda", dtype=torch.float32).unsqueeze(0)
            start = JointState.from_position(q0, joint_names=_planner.joint_names)
            goal = JointState.from_position(qg, joint_names=_planner.joint_names)
            # Graph seeding, explicitly, for the same reason the Cartesian branch uses it: a joint-space
            # goal can be most of a turn away (park is a pi shoulder swing from anywhere in the
            # workspace), and pure trajectory optimisation from a straight-line seed has no way
            # around an obstacle in the middle. The graph search does.
            result = _planner.plan_cspace(
                goal, start, max_attempts=_PLAN_MAX_ATTEMPTS, enable_graph_attempt=_PLAN_GRAPH_FROM,
            )
            ok = result is not None and bool(result.success.any())
            print(f"[plan_js] goal={[round(x, 3) for x in req['goal_joints']]} -> "
                  f"{'OK' if ok else 'FAIL'}", file=sys.stderr, flush=True)
            if not ok:
                _emit({"success": False, "planner_error": False,
                       "reason": "no collision-free joint-space plan"})
                continue
            # reshape(-1, N) like the Cartesian branch: position carries a leading BATCH dim, so
            # tolist() without it emits [[[q...], [q...]]] one "waypoint" that is itself the whole
            # path. The caller would then iterate once over a list of lists. Measured: home->park came
            # back as 1 waypoint for a pi shoulder rotation, which is what exposed it.
            traj = result.get_interpolated_plan().position.reshape(-1, _N).cpu().tolist()
            _emit({"success": True, "trajectory": traj, "dt": _DT})
        except Exception as exc:  # noqa: BLE001 - report, never take the sidecar down mid-session
            # planner_error=True: the CALL failed, the planner never rendered a verdict. The
            # distinction is not cosmetic, see the note on the client's plan_joint.
            print(f"[plan_cspace] CALL FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            _emit({"success": False, "planner_error": True, "reason": f"{type(exc).__name__}: {exc}"})
        continue
    if cmd == "set_world":
        # Replace cuRobo's collision world with the caller's obstacles (the sim's bin walls + neighbour
        # cuboids, base frame, metres). cuboids: [{"name","dims_m":[x,y,z],"pose":[px,py,pz,qw,qx,qy,qz]}].
        try:
            world = {c["name"]: {"dims": list(c["dims_m"]), "pose": list(c["pose"])} for c in req["cuboids"]}
            _planner.update_world(SceneCfg.create({"cuboid": world}))
            _emit({"world_set": len(world)})
        except Exception as exc:  # noqa: BLE001
            _emit({"world_set": None, "reason": f"{type(exc).__name__}: {exc}"})
        continue
    try:
        q0 = torch.tensor(req["start_joints"], device="cuda", dtype=torch.float32).unsqueeze(0)
        q_start = JointState.from_position(q0, joint_names=_planner.joint_names)
        goal = GoalToolPose(
            tool_frames=_planner.tool_frames,
            position=torch.tensor([[[[req["goal_pos_m"]]]]], device="cuda", dtype=torch.float32),
            quaternion=torch.tensor([[[[req["goal_quat_wxyz"]]]]], device="cuda", dtype=torch.float32),
        )
        result = _planner.plan_pose(
            goal, q_start, max_attempts=_PLAN_MAX_ATTEMPTS, enable_graph_attempt=_PLAN_GRAPH_FROM,
        )
        ok = result is not None and bool(result.success.any())
        print(f"[plan] goal={[round(x,3) for x in req['goal_pos_m']]} -> "
              f"{'OK' if ok else 'NONE'}", file=sys.stderr, flush=True)
        if ok:
            traj = result.get_interpolated_plan().position.reshape(-1, _N).cpu().tolist()
            _emit({"success": True, "trajectory": traj, "dt": _DT})
        else:
            # result is None when the query is invalid (unreachable IK / collision)
            _emit({"success": False, "planner_error": False,
                   "reason": "plan_pose returned no solution (unreachable / in-collision)"})
    except Exception as exc:  # noqa: BLE001
        print(f"[plan] CALL FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        _emit({"success": False, "planner_error": True, "reason": f"{type(exc).__name__}: {exc}"})
