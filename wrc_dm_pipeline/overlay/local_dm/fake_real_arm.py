#!/usr/bin/env python
"""Rehearse the bridge's REAL-ARM code path without an arm.

Runs dm_bridge_server.py with --robot real, but replaces LeRobot's follower driver
with a stand-in that moves each joint toward its command at a limited speed. It
exercises the bridge's real-arm logic (joint units, measured gripper range, 50 Hz
streaming, lag guard). It proves nothing about the physical arm.

  <teleop>/.venv/bin/python local_dm/fake_real_arm.py --headless     (extra args go to the bridge)
"""
from __future__ import annotations

import os
import sys
import time
import types
from pathlib import Path

import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll"]
START_DEG = np.degrees([0.0, -1.51, -1.30, 1.36, 0.0, 0.0])      # the team's READY pose


class FakeConfig:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeFollower:
    def __init__(self, cfg: FakeConfig):
        self.cfg = cfg
        lo, hi = cfg.joint_limits["gripper"]
        self.g_lim = (lo, hi)
        self.pos = {f"{n}.pos": float(v) for n, v in zip(JOINTS, START_DEG)}
        self.pos["gripper.pos"] = float(lo)
        self.cmd_min = dict(self.pos)
        self.cmd_max = dict(self.pos)
        self.n = 0
        self._t = time.perf_counter()

    def connect(self):
        print(f"[fake-arm] connected (port {self.cfg.port}, control {self.cfg.control_mode}, "
              f"max step {self.cfg.max_relative_target} deg, {self.cfg.pos_vel_velocity} deg/s)", flush=True)

    def get_observation(self) -> dict:
        return dict(self.pos)

    def send_action(self, action: dict):
        now = time.perf_counter()
        dt, self._t = min(now - self._t, 0.1), now
        step = min(float(self.cfg.max_relative_target), float(self.cfg.pos_vel_velocity) * dt)
        if os.environ.get("FAKE_ARM_STUCK"):                    # rehearse a blocked arm (lag guard)
            step = 0.0
        for k, target in action.items():
            self.cmd_min[k], self.cmd_max[k] = min(self.cmd_min[k], target), max(self.cmd_max[k], target)
            if k == "gripper.pos":
                target = float(np.clip(target, *self.g_lim))
                gstep = 400.0 * dt                              # gripper motor: fast
                self.pos[k] += float(np.clip(target - self.pos[k], -gstep, gstep))
            else:
                self.pos[k] += float(np.clip(target - self.pos[k], -step, step))
        self.n += 1

    def disconnect(self):
        print(f"[fake-arm] disconnected after {self.n} commands. Commanded ranges [deg]:")
        for k in self.pos:
            print(f"   {k:18s} {self.cmd_min[k]:8.1f} .. {self.cmd_max[k]:8.1f}")
        sys.stdout.flush()


def main() -> int:
    mod = types.ModuleType("lerobot.robots.rebot_b601_follower")
    mod.RebotB601Follower, mod.RebotB601FollowerRobotConfig = FakeFollower, FakeConfig
    for name in ("lerobot", "lerobot.robots"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["lerobot.robots.rebot_b601_follower"] = mod
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import dm_bridge_server

    sys.argv = [sys.argv[0], "--robot", "real", "--confirm-real", "--serial-port", "FAKE"] + sys.argv[1:]
    return dm_bridge_server.main()


if __name__ == "__main__":
    sys.exit(main())
