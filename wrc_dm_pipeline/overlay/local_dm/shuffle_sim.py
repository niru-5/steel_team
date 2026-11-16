#!/usr/bin/env python
"""Move the objects of the running DM simulator to new random places (inside the area
the arm can reach), so you can check the robot really finds them with its camera.

  ./START_WRC.sh dm-shuffle          (while ./START_WRC.sh dm is running; not during a task)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_trials
from wrc_dm_plugin import BridgeClient


def main() -> int:
    try:
        b = BridgeClient()
        info = b.call("info")[0]
        if info.get("robot") != "sim":
            print("the bridge is not the simulator: nothing to shuffle")
            return 2
        if info.get("scene") == "real":       # the team's scene: its own randomizer decides where objects may go
            for n, pos in b.call("shuffle")[0]["objects"].items():
                print(f"{n:6s} -> x {pos[0]:.2f} m, y {pos[1]:+.2f} m")
            print("The camera needs a second or two to see the new scene. Some random places are out of the "
                  "arm's reach; shuffle again if the robot says so.")
            return 0
    except RuntimeError as e:
        print(e)
        return 2
    rng = np.random.default_rng()
    for _ in range(5000):
        lay = {n: dict(pos=[float(rng.uniform(*o["zone"][0])), float(rng.uniform(*o["zone"][1])), o["z"]],
                       yaw=float(rng.uniform(0, np.pi))) for n, o in sim_trials.OBJ.items()}
        pts = [np.array(v["pos"][:2]) for v in lay.values()]
        if (all(np.linalg.norm(a - c) >= 0.11 for i, a in enumerate(pts) for c in pts[i + 1:])
                and all(np.linalg.norm(p - np.array(sim_trials.DROP_ZONE)) >= 0.12 for p in pts)):
            break
    b.call("set_objects", objects=lay)
    for n, v in lay.items():
        print(f"{n:16s} -> x {v['pos'][0]:.2f} m, y {v['pos'][1]:+.2f} m, turned {np.degrees(v['yaw']):.0f} deg")
    print("The camera needs a second or two to see the new scene.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
