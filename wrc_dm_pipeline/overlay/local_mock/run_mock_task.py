#!/usr/bin/env python
"""Local mock-mode driver for WRC (added by the hackathon setup; NOT upstream).

Runs a task through the SAME pipeline as ``wrc-demo`` (load_demo_config ->
build_runtime -> AgentOrchestrator with reflex / experience / LLM tiers) on
the fully mocked stack (mock camera + mock detector + mock arm), with two
opt-in, mock-only knobs that the stock CLI does not expose:

  --simulate-contact   set ``arm.object_stop_frac = 0.5`` so the mock gripper
                       stalls on the object like a real grasp. Without it the
                       CLI's mock arm always reports an "air grasp" (empty
                       gripper). This is the same fixture the repo's own
                       tests/test_orchestrator_e2e.py uses.
  --scripted-plan      replace the default 2-step mock LLM script with a
                       fixed, hand-written tool-call sequence for the
                       "survey / locate red block / pick / place next to the
                       box" task. The LLM DECISIONS are scripted (no model
                       runs); everything downstream of the decisions is the
                       real code path.

Nothing here talks to hardware, the network or any paid API. Persistent
learned state (grasp memory, tier-2 experience) is written into the run dir,
not into ~/.wrc_demo or runs/experience.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from wrc_demo.agent.llm import LLMResponse, MockLLM, ToolCall, make_llm
from wrc_demo.agent.orchestrator import AgentOrchestrator
from wrc_demo.apps.demo import _print_report, build_runtime, shutdown_runtime
from wrc_demo.config import PACKAGE_ROOT, load_demo_config

DEFAULT_TASK = ("Survey the workspace, identify the available objects, locate the "
                "red block, pick it up, and place it next to the box.")


def _scripted_plan() -> MockLLM:
    def call(name, args=None):
        return LLMResponse(tool_calls=[ToolCall(name, args or {})])

    return MockLLM([
        call("get_observation"),                                  # survey
        call("list_objects"),                                     # identify objects
        call("localize_object", {"label": "red cube"}),          # locate red block
        call("localize_object", {"label": "box"}),               # locate the box (absent in mock scene)
        # "next to the box": the mock scene has no box, so use the profile's
        # default drop zone (destination omitted) -- see grasp.drop_zone.
        call("pick_and_place", {"object": "red cube", "material": "rigid"}),
        call("list_objects"),                                     # verify final state
        call("task_done", {"success": True,
                           "summary": "scripted plan: surveyed, picked the red cube, "
                                      "placed it at the default drop zone (no box in mock scene)"}),
    ])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--simulate-contact", action="store_true")
    p.add_argument("--scripted-plan", action="store_true")
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--run-dir", default=None)
    args = p.parse_args(argv)

    cfg = load_demo_config(arm="mock", llm="mock")            # mock camera is implied by arm=mock
    run_dir = Path(args.run_dir) if args.run_dir else (
        PACKAGE_ROOT / "runs" / ("mock_" + time.strftime("%Y%m%d_%H%M%S")))
    cfg._data["grasp"]["memory_path"] = str(run_dir / "grasp_memory.json")

    llm = _scripted_plan() if args.scripted_plan else make_llm(cfg.llm)
    print(f"[mock-driver] camera={cfg.camera.name} arm={cfg.arm.type} llm={'scripted-plan' if args.scripted_plan else cfg.llm.type} "
          f"simulate_contact={args.simulate_contact}")
    print(f"[mock-driver] traces -> {run_dir}")

    runtime, arm = build_runtime(cfg, run_dir, view=False, serve=False)
    try:
        if args.simulate_contact:
            arm.object_stop_frac = 0.5
        # Wait for the WorldWatcher thread to make a detector pass.
        deadline = time.monotonic() + 10
        while runtime.beliefs.find("red cube") is None and time.monotonic() < deadline:
            time.sleep(0.05)
        from wrc_demo.agent.reflex import ExperienceMemory, FastPlanner
        agent = AgentOrchestrator(
            llm, runtime, advisor=None, max_steps=args.max_steps, decompose=False,
            fast_planner=None if args.scripted_plan else FastPlanner(ExperienceMemory(run_dir / "experience.json")),
            cfg=cfg,
        )
        runtime.current_task = args.task
        report = agent.run_task(args.task)
        runtime.current_task = None
        _print_report(report)
        print("\n=== tool log ===")
        for e in report.tool_log:
            r = e.get("result", {})
            print(f"- {e.get('tool')}({json.dumps(e.get('args', {}))}) -> ok={r.get('ok')} "
                  f"{str(r.get('error') or '')[:200]}")
        b = runtime.beliefs.find("red cube")
        print(f"\nred cube belief position (base frame, m): {None if b is None else [round(float(v), 3) for v in b.position]}")
        print(f"held object: {runtime.held_object}")
        print("NOTE: mock arm only — no physical motion occurred.")
        return 0 if report.success else 1
    finally:
        shutdown_runtime(runtime, arm)


if __name__ == "__main__":
    sys.exit(main())
