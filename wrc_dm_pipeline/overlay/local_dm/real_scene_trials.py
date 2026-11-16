#!/usr/bin/env python
"""Randomized trials in the team's real-workspace simulation (steel_team `simulation/`, rebot_sim).

Each trial: the can and the cup get new random places from the team's own randomizer
(rebot_sim spawn_region), a FRESH WRC process gets one task ("put X in bin N"), and the
result is judged on the simulator's ground truth -- is the object inside that bin
compartment? -- not on what the pipeline reports.

Started by  ./START_WRC.sh dm-trials  when the simulator runs the real-workspace scene.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import shlex

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
# detector label (what the keyword commands must say) and the words a person would use
OBJ = {"can": dict(label="red cup", names=["red can", "can", "red tin"]),
       "cup": dict(label="cup", names=["paper cup", "cup", "white cup"])}
BIN_WORDS = {"bin_0": ["the bin", "bin_0", "the bin compartment nearest to the robot"],
             "bin_1": ["bin_1", "the far compartment of the bin", "the second bin compartment"]}
PROMPTS = ["Put the {o} in {b}", "Throw the {o} into {b}", "Pick up the {o} and drop it in {b}",
           "Please tidy up: the {o} goes into {b}"]


def run(bridge, args, out: Path) -> int:
    rng = np.random.default_rng(args.seed)
    results = []
    env = dict(os.environ, PYTHONNOUSERSITE="1")
    print(f"{args.n} trials in the real-workspace scene, seed {args.seed}, planner: "
          f"{'keyword commands' if args.no_llm else 'Claude'}; results -> {out}\n", flush=True)
    for i in range(1, args.n + 1):
        target = ("can", "cup")[i % 2]
        goal = ("bin_0", "bin_1")[(i // 2) % 2]
        if args.no_llm:
            task = f"put the {OBJ[target]['label']} in {goal}"
        else:
            task = PROMPTS[int(rng.integers(len(PROMPTS)))].format(
                o=OBJ[target]["names"][int(rng.integers(3))], b=BIN_WORDS[goal][int(rng.integers(3))])
        bridge.call("reset")
        start = bridge.call("shuffle", seed=int(args.seed * 1000 + i))[0]["objects"]
        time.sleep(1.0)
        jpath, log = out / f"trial_{i:02d}.json", out / f"trial_{i:02d}.log"
        cmd = [sys.executable, "-s", str(HERE / "run_dm.py"), "--task", task, "--run-dir", str(out / f"trial_{i:02d}"),
               "--json-out", str(jpath), "--llm", "mock" if args.no_llm else "claude"]
        cmd += shlex.split(getattr(args, "extra", "") or "")
        t0, timed_out = time.time(), False
        with open(log, "w") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(REPO))
            try:
                proc.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                bridge.call("hold")
        time.sleep(1.5)
        st = bridge.call("state")[0]
        rep = json.loads(jpath.read_text()) if jpath.exists() else {}
        other = next(n for n in OBJ if n != target)
        placed = st["in_bin"].get(target) == goal
        moved_other = float(np.linalg.norm(np.array(st["objects"][other]) - np.array(start[other])))
        errors = [e["error"] for e in rep.get("tool_log", []) if e.get("error")]
        r = dict(trial=i, target=target, goal=goal, task=task, start_xy=[round(v, 3) for v in start[target][:2]],
                 placed=bool(placed), ended_in=st["in_bin"].get(target), other_moved_cm=round(100 * moved_other, 1),
                 reported_success=rep.get("success"), timed_out=timed_out, wall_s=round(time.time() - t0, 1),
                 claude_calls=len(rep.get("claude_calls", [])), tools=[e["tool"] for e in rep.get("tool_log", [])],
                 error=(errors[-1] if errors else ""), summary=rep.get("summary", "")[:300])
        results.append(r)
        agree = "" if bool(r["reported_success"]) == placed else "  <-- pipeline reported the opposite"
        print(f"{i:2d} {'OK  ' if placed else 'FAIL'} {target:3s} -> {goal} from ({r['start_xy'][0]:.2f},{r['start_xy'][1]:+.2f}) "
              f"{r['wall_s']:5.1f}s  \"{task}\"{agree}", flush=True)
        if not placed and (r["error"] or r["summary"]):
            print(f"        why: {(r['error'] or r['summary'])[:200]}", flush=True)
        (out / "results.json").write_text(json.dumps(results, indent=1))
    ok = sum(r["placed"] for r in results)
    lines = [f"# Real-workspace sim trials: {ok}/{len(results)} succeeded ({100 * ok / max(len(results), 1):.0f} %)", "",
             f"Seed {args.seed}, planner: {'keyword commands' if args.no_llm else 'Claude'}. Random start places from the "
             "team's rebot_sim randomizer. Judged on simulator ground truth: the object must lie inside the named bin compartment.", ""]
    for name in OBJ:
        sub = [r for r in results if r["target"] == name]
        if sub:
            lines.append(f"- {name}: {sum(r['placed'] for r in sub)}/{len(sub)}")
    lines += ["", "| # | object | goal | start x,y | result | s | task |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['trial']} | {r['target']} | {r['goal']} | {r['start_xy'][0]:.2f}, {r['start_xy'][1]:+.2f} | "
              f"{'ok' if r['placed'] else 'FAIL: ' + (r['error'] or r['summary'])[:80]} | {r['wall_s']} | {r['task']} |" for r in results]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"\n{ok}/{len(results)} succeeded. Details: {out / 'summary.md'}", flush=True)
    return 0 if ok == len(results) else 1
