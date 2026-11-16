#!/usr/bin/env python
"""WRC live dashboard on the mock stack (added by the hackathon setup; NOT upstream).

Same wiring as scripts/dashboard_runner.py (build_runtime + AgentOrchestrator +
the stdlib MJPEG dashboard with its chat box and STOP button), adapted for a
laptop demo:
  * mock camera + mock detector + mock arm, mock LLM (no keys, no network)
  * simulated grasp contact (arm.object_stop_frac = 0.5) so picks can succeed
  * binds to 127.0.0.1 only (the upstream default 0.0.0.0 would expose the
    robot's chat box to everyone on the hackathon Wi-Fi)
  * no 60 s wait for a "pink object" the mock scene does not contain
Open http://127.0.0.1:8090/ and type tasks into the dashboard's chat box.
Nothing physical moves.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_OFFLINE", "True")
os.chdir(REPO)

from wrc_demo.agent.llm import make_llm
from wrc_demo.agent.orchestrator import AgentOrchestrator
from wrc_demo.agent.reflex import ExperienceMemory, FastPlanner
from wrc_demo.apps.demo import build_runtime, shutdown_runtime
from wrc_demo.config import load_demo_config


def main() -> int:
    p = argparse.ArgumentParser(description="WRC mock dashboard")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--no-contact", action="store_true",
                   help="disable simulated grasp contact (stock mock behaviour: air grasps)")
    p.add_argument("--mirror-port", type=int, default=9870,
                   help="UDP port on 127.0.0.1 for the MuJoCo 3D mirror (local_mock/wrc_mujoco_mirror.py)")
    p.add_argument("--no-mirror", action="store_true", help="do not stream arm state to the 3D mirror")
    args = p.parse_args()

    cfg = load_demo_config(arm="mock", llm="mock")
    run_dir = REPO / "runs" / ("dashboard_" + time.strftime("%Y%m%d_%H%M%S"))
    cfg._data["stream"]["host"] = "127.0.0.1"
    cfg._data["stream"]["port"] = args.port
    cfg._data["grasp"]["memory_path"] = str(run_dir / "grasp_memory.json")

    llm = make_llm(cfg.llm)
    runtime, arm = build_runtime(cfg, run_dir, view=False, serve=True)
    if runtime.stream_server is None:
        print(f"ERROR: could not start the dashboard on port {args.port} (already in use?). "
              "A WRC dashboard is probably already running — use it at "
              f"http://127.0.0.1:{args.port}/ or stop it first (pkill -f local_mock/dashboard_mock.py).",
              file=sys.stderr, flush=True)
        shutdown_runtime(runtime, arm)
        return 2
    if not args.no_contact:
        arm.object_stop_frac = 0.5
    agent = AgentOrchestrator(
        llm, runtime, decompose=False, cfg=cfg,
        fast_planner=FastPlanner(ExperienceMemory(run_dir / "experience.json")),
    )

    def run_task(task: str):
        runtime.current_task = task
        try:
            r = agent.run_task(task)
            if r.path == "llm":
                runtime.memory.add(
                    "note",
                    f"{task!r} is not a built-in reflex command and the LLM tier is a MOCK "
                    "(no API key), so I cannot plan it. Try: look around / go home / "
                    "open the gripper / pick and place the red cube.")
            print(f"TASK {task!r}: success={r.success} path={r.path} {r.duration_s}s | {r.summary[:200]}",
                  flush=True)
            return r
        finally:
            runtime.current_task = None
            if runtime.arm.harness.estopped:
                runtime.arm.harness.reset_estop()
                runtime.memory.add("note", "stop cleared; ready for the next command")

    def cancel_task():
        runtime.memory.add("outcome", "USER pressed STOP -- aborting the current action")
        runtime.arm.harness.estop("chat cancel")

    runtime.stream_server.set_task_fn(run_task)
    runtime.stream_server.set_cancel_fn(cancel_task)
    mirror = None
    if not args.no_mirror:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from mirror_publisher import MirrorPublisher
        mirror = MirrorPublisher(runtime, arm, cfg, port=args.mirror_port).start()
        print(f"3D MIRROR: streaming arm state to udp://127.0.0.1:{args.mirror_port}", flush=True)
    print(f"DASHBOARD: http://127.0.0.1:{args.port}/  (traces -> {run_dir})", flush=True)
    print(f"simulated grasp contact: {not args.no_contact}. Ctrl+C to stop.", flush=True)
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        if mirror is not None:
            mirror.stop()
        shutdown_runtime(runtime, arm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
