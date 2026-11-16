#!/usr/bin/env python
"""Randomized pick-and-place trials on the DM physics sim -> a success rate.

Each trial: the three sim objects get new random positions (inside the area the arm
can reach and the camera can see), a FRESH WRC process gets one task, and the result
is judged on the simulator's ground truth -- not on what the pipeline reports.

  ./START_WRC.sh dm-trials                 # 20 trials, Claude plans (about 4 US cents each)
  ./START_WRC.sh dm-trials --n 5 --no-llm  # keyword commands only, no API calls
  ./START_WRC.sh dm-trials --seed 7        # a different random set

Results: runs/trials_<time>/results.json and summary.md
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
PORT = int(os.environ.get("WRC_DM_BRIDGE_PORT") or 9881)   # set it to use a private simulator
TABLE_Z = 0.03
BLUE_TOP_Z = TABLE_Z + 2 * 0.018
DROP_ZONE = (0.25, 0.14)            # run_dm.py sets grasp.drop_zone to this
# z: centre height when resting on the table. zone: where the object may lie when it is the one
# to pick (from the reach map in README.md; tall objects have a smaller zone).
OBJ = {
    "red_cube": dict(z=0.055, half_h=0.025, label="red block", names=["red block", "red cube"],
                     zone=((0.29, 0.39), (-0.18, 0.18))),
    "yellow_cylinder": dict(z=0.065, half_h=0.035, label="yellow ball", names=["yellow cylinder", "yellow thing"],
                            zone=((0.29, 0.33), (-0.08, 0.08))),
    "blue_block": dict(z=0.048, half_h=0.018, label="blue block", names=["blue block"],
                       zone=((0.29, 0.39), (-0.18, 0.18))),
}
OTHER_ZONE = ((0.29, 0.50), (-0.20, 0.20))      # where objects that are not picked may lie (in view)
STACK_BASE_ZONE = ((0.29, 0.46), (-0.18, 0.18))  # where the blue block may lie when something is stacked on it
FREE_PROMPTS = ["Move the {o} to the free spot", "Pick up the {o} and put it down in the empty area",
                "Please relocate the {o} to the drop zone", "Grab the {o} and place it in the free space"]
STACK_PROMPTS = ["Put the {o} on the blue block", "Stack the {o} on top of the blue one",
                 "Place the {o} onto the blue block"]
PLACE_TOL_XY, PLACE_TOL_Z, DISTURB_TOL = 0.03, 0.012, 0.03


def port_open() -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", PORT)) == 0


def sample_layout(rng, target: str, dest: str) -> dict:
    for _ in range(2000):
        lay = {}
        for name in OBJ:
            zone = OBJ[name]["zone"] if name == target else (
                STACK_BASE_ZONE if (name == "blue_block" and dest == "blue") else OTHER_ZONE)
            lay[name] = dict(pos=[float(rng.uniform(*zone[0])), float(rng.uniform(*zone[1])), OBJ[name]["z"]],
                             yaw=float(rng.uniform(0, np.pi)))
        pts = [np.array(v["pos"][:2]) for v in lay.values()]
        apart = all(np.linalg.norm(a - b) >= 0.11 for i, a in enumerate(pts) for b in pts[i + 1:])
        clear = all(np.linalg.norm(p - np.array(DROP_ZONE)) >= 0.12 for p in pts)
        if apart and clear:
            return lay
    raise RuntimeError("could not sample a layout")


def make_trials(n: int, seed: int, use_llm: bool) -> list[dict]:
    rng = np.random.default_rng(seed)
    targets = (["red_cube"] * 10 + ["yellow_cylinder"] * 5 + ["blue_block"] * 5)
    targets = [targets[i % len(targets)] for i in rng.permutation(max(n, len(targets)))][:n]
    trials, flip = [], 0
    for i, target in enumerate(targets):
        if target == "blue_block":
            dest = "free"
        else:
            dest = "blue" if flip % 2 else "free"
            flip += 1
        info = OBJ[target]
        if use_llm:
            name = info["names"][int(rng.integers(len(info["names"])))]
            tmpl = (STACK_PROMPTS if dest == "blue" else FREE_PROMPTS)
            task = tmpl[int(rng.integers(len(tmpl)))].format(o=name)
        else:
            task = (f"put the {info['label']} on the blue block" if dest == "blue"
                    else f"pick and place the {info['label']}")
        trials.append(dict(i=i + 1, target=target, dest=dest, task=task, layout=sample_layout(rng, target, dest)))
    return trials


MAX_TILT_DEG = 15.0   # an object left leaning more than this did not land properly


def judge(trial: dict, before: dict, after: dict, tilt_deg: float = 0.0) -> dict:
    tgt = np.array(after[trial["target"]])
    if trial["dest"] == "free":
        goal_xy, goal_z = np.array(DROP_ZONE), OBJ[trial["target"]]["z"]
    else:
        goal_xy, goal_z = np.array(after["blue_block"][:2]), BLUE_TOP_Z + OBJ[trial["target"]]["half_h"]
    dxy, dz = float(np.linalg.norm(tgt[:2] - goal_xy)), float(abs(tgt[2] - goal_z))
    disturbed = {n: float(np.linalg.norm(np.array(after[n][:2]) - np.array(before[n][:2])))
                 for n in OBJ if n != trial["target"]}
    worst = max(disturbed.values())
    return dict(placed=bool(dxy <= PLACE_TOL_XY and dz <= PLACE_TOL_Z and tilt_deg <= MAX_TILT_DEG),
                tilt_deg=round(float(tilt_deg), 1), err_xy_mm=round(1000 * dxy, 1),
                err_z_mm=round(1000 * dz, 1), moved_cm=round(100 * float(np.linalg.norm(tgt - np.array(before[trial['target']]))), 1),
                disturbed_others=bool(worst > DISTURB_TOL), worst_other_shift_mm=round(1000 * worst, 1))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--no-llm", action="store_true", help="keyword commands instead of Claude (no API calls)")
    p.add_argument("--timeout", type=float, default=240.0, help="per trial [s]")
    p.add_argument("--extra", default="", help="extra options passed to run_dm.py for every trial, e.g. "
                                               "\"--rgb-only --camera2 dm_sim_overhead\" (no depth, two cameras)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    from wrc_dm_plugin import BridgeClient
    out = Path(args.out) if args.out else REPO / "runs" / ("trials_" + time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    srv = None
    if not port_open():
        teleop = Path(os.environ.get("STEEL_TEAM_TELEOP_DIR", REPO.parent / "steel_team" / "teleoperation"))
        py = teleop / ".venv" / "bin" / "python"
        if not py.exists():
            print(f"teleoperation venv not found at {py}")
            return 2
        srv = subprocess.Popen([str(py), str(HERE / "dm_bridge_server.py"), "--headless", "--port", str(PORT),
                                "--teleop-dir", str(teleop)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 40
        while not port_open():
            if time.time() > deadline or srv.poll() is not None:
                print("the sim bridge did not start")
                return 2
            time.sleep(0.3)
    bridge = BridgeClient(port=PORT)
    info = bridge.call("info")[0]
    if info.get("robot") != "sim":
        print(f"the bridge on port {PORT} is not the simulator: refusing to run trials against a real arm")
        return 2
    if info.get("scene") == "real":          # the team's real-workspace scene has its own objects and goals
        import real_scene_trials
        try:
            return real_scene_trials.run(bridge, args, out)
        finally:
            if srv is not None:
                srv.terminate()

    trials = make_trials(args.n, args.seed, use_llm=not args.no_llm)
    results = []
    env = dict(os.environ, PYTHONNOUSERSITE="1")
    print(f"{len(trials)} trials, seed {args.seed}, planner: {'keyword commands' if args.no_llm else 'Claude'}; results -> {out}\n", flush=True)
    try:
        for t in trials:
            bridge.call("reset")
            bridge.call("set_objects", objects=t["layout"])
            time.sleep(1.0)                                   # let the objects settle
            before = bridge.call("state")[0]["objects"]
            jpath = out / f"trial_{t['i']:02d}.json"
            cmd = [sys.executable, "-s", str(HERE / "run_dm.py"), "--task", t["task"],
                   "--run-dir", str(out / f"trial_{t['i']:02d}"), "--json-out", str(jpath),
                   "--llm", "mock" if args.no_llm else "claude",
                   "--drop-zone", f"{DROP_ZONE[0]},{DROP_ZONE[1]}"]      # fixed goal so the judge can check it
            t0 = time.time()
            log = out / f"trial_{t['i']:02d}.log"
            timed_out = False
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
            time.sleep(1.5)                                   # let a released object come to rest
            st_after = bridge.call("state")[0]
            after = st_after["objects"]
            rep = json.loads(jpath.read_text()) if jpath.exists() else {}
            j = judge(t, before, after, st_after.get("object_tilt_deg", {}).get(t["target"], 0.0))
            errors = [e["error"] for e in rep.get("tool_log", []) if e.get("error")]
            r = dict(trial=t["i"], target=t["target"], dest=t["dest"], task=t["task"],
                     start_xy=[round(v, 3) for v in t["layout"][t["target"]]["pos"][:2]],
                     reported_success=rep.get("success"), timed_out=timed_out, wall_s=round(time.time() - t0, 1),
                     pipeline_s=rep.get("duration_s"), claude_calls=len(rep.get("claude_calls", [])),
                     tools=[e["tool"] for e in rep.get("tool_log", [])], error=(errors[-1] if errors else ""),
                     summary=rep.get("summary", "")[:300], **j)
            results.append(r)
            verdict = "OK  " if j["placed"] else "FAIL"
            agree = "" if (bool(r["reported_success"]) == j["placed"]) else "  <-- pipeline reported the opposite"
            print(f"{t['i']:2d} {verdict} {t['target']:15s} -> {t['dest']:4s} at ({r['start_xy'][0]:.2f},{r['start_xy'][1]:+.2f}) "
                  f"off {j['err_xy_mm']:5.1f} mm  {r['wall_s']:5.1f}s  \"{t['task']}\"{agree}", flush=True)
            if not j["placed"] and (r["error"] or r["summary"]):
                print(f"        why: {(r['error'] or r['summary'])[:170]}", flush=True)
            (out / "results.json").write_text(json.dumps(results, indent=1))
    finally:
        if srv is not None:
            srv.terminate()
            try:
                srv.wait(timeout=5)
            except subprocess.TimeoutExpired:
                srv.kill()

    n = len(results)
    ok = [r for r in results if r["placed"]]
    lines = [f"# DM sim trials: {len(ok)}/{n} succeeded ({100 * len(ok) / max(n, 1):.0f} %)", "",
             f"Seed {args.seed}, planner: {'keyword commands' if args.no_llm else 'Claude'}. Judged on simulator ground truth: "
             f"the object must end within {PLACE_TOL_XY * 100:.0f} cm of the goal, at the right height, and not leaning.", ""]
    for name in OBJ:
        sub = [r for r in results if r["target"] == name]
        if sub:
            lines.append(f"- {name}: {sum(r['placed'] for r in sub)}/{len(sub)}")
    for dest in ("free", "blue"):
        sub = [r for r in results if r["dest"] == dest]
        if sub:
            lines.append(f"- destination '{'free spot' if dest == 'free' else 'on the blue block'}': {sum(r['placed'] for r in sub)}/{len(sub)}")
    if ok:
        lines.append(f"- placement error of the successes: mean {np.mean([r['err_xy_mm'] for r in ok]):.1f} mm, "
                     f"max {np.max([r['err_xy_mm'] for r in ok]):.1f} mm")
        lines.append(f"- time per successful task: mean {np.mean([r['wall_s'] for r in ok]):.0f} s (includes ~8 s start-up of a fresh process)")
    wrong = [r for r in results if bool(r["reported_success"]) != r["placed"]]
    lines.append(f"- pipeline's own report disagreed with the ground truth in {len(wrong)} trial(s)"
                 + (": " + ", ".join(f"#{r['trial']} (reported {'success' if r['reported_success'] else 'failure'})" for r in wrong) if wrong else ""))
    lines.append(f"- other objects knocked more than 3 cm: {sum(r['disturbed_others'] for r in results)} trial(s)")
    if not args.no_llm:
        lines.append(f"- Claude calls: {sum(r['claude_calls'] for r in results)} in total")
    fails = [r for r in results if not r["placed"]]
    if fails:
        lines += ["", "## Failures", ""]
        for r in fails:
            lines.append(f"- #{r['trial']} {r['target']} -> {r['dest']} from ({r['start_xy'][0]:.2f}, {r['start_xy'][1]:+.2f}): "
                         f"{(r['error'] or r['summary'] or ('timed out' if r['timed_out'] else 'no report'))[:220]}")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
