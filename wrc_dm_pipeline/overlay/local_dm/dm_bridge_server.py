#!/usr/bin/env python
"""DM arm bridge server (hackathon DM port; NOT upstream WRC, NOT part of steel_team).

Exposes a Seeed reBot Arm B601-DM to WRC over a localhost TCP socket:

  --robot sim    MuJoCo PHYSICS simulation of Seeed's official B601-DM model
                 (servo-driven joints, contact-based grasping, RGB-D overhead camera).
                 This is the verified path.
  --robot real   The physical arm through LeRobot's rebot_b601_follower, wired like the team's
                 teleoperation/go_ready.py (pos_vel control, measured gripper range).
                 *** THIS BRIDGE PATH IS UNTESTED ON HARDWARE *** -- see local_dm/README.md for
                 the bring-up checklist. Needs --confirm-real.

Run it with the teleoperation venv (that env has mujoco; WRC's env does not):

  <teleop>/.venv/bin/mjpython local_dm/dm_bridge_server.py            # 3D window (macOS needs mjpython)
  <teleop>/.venv/bin/python   local_dm/dm_bridge_server.py --headless

Wire protocol (one JSON object per line; replies are a JSON line optionally followed
by raw binary blobs whose byte lengths are listed in reply["blobs"]):

  {"op": "info"}                      -> home_q, finger_max, robot kind, camera names, joint_lo / joint_hi (rad)
  {"op": "state"}                     -> q[6] rad, dq[6], finger m, finger_vel, t  (+ sim object poses)
  {"op": "target", "q": [6]}          -> set the joint position targets (rad)
  {"op": "gripper", "f": 0.03}        -> set the finger opening target (m, 0 = closed .. 0.05 = open)
  {"op": "hold"}                      -> freeze at the measured pose
  {"op": "frame", "cam": "overhead_rgb"} -> BGR uint8 HxWx3 + depth float32 HxW (m) + K + T_base_cam   (sim only)
  {"op": "reset"}                     -> sim only: objects + arm back to the start layout
  {"op": "set_objects", "objects": {name: {"pos": [x, y, z], "yaw": rad}}}  -> sim only, for tests
  {"op": "shuffle", "seed": int|null} -> real-workspace sim only: objects to random places (team's randomizer)
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_TELEOP = Path(os.environ.get("STEEL_TEAM_TELEOP_DIR",
                                     HERE.parents[1] / "steel_team" / "teleoperation"))

# WRC DM home pose: arm pulled back and raised so the open gripper does not hide the table from the overhead camera.
DM_HOME_Q = [0.0, -1.3, -2.0, 1.0, 0.0, 0.0]

# Start layout of the sim scene (base frame, metres). The table top is at z = 0.03.
SIM_LAYOUT = {
    "red_cube": [0.34, -0.13, 0.055],
    "blue_block": [0.40, 0.13, 0.048],
    "yellow_cylinder": [0.30, 0.02, 0.065],
}
# Firmer finger servo than teleop's default (kp 200): the real gripper pushes with
# a torque limit until the object stalls it; kp 1500 with the 20 N force limit gives
# a few newtons of squeeze on a 5 cm cube instead of ~1 N.
GRIP_KP, GRIP_KV = 1500.0, 40.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", choices=["sim", "real"], default="sim")
    p.add_argument("--teleop-dir", default=str(DEFAULT_TELEOP),
                   help="steel_team/teleoperation folder (provides teleop.rebot_model, rebot_real, assets)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=int(os.environ.get("WRC_DM_BRIDGE_PORT") or 9881))
    p.add_argument("--headless", action="store_true", help="no 3D window")
    p.add_argument("--duration", type=float, default=None, help="exit after N seconds (tests)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--scene", choices=["auto", "real", "demo"], default=os.environ.get("WRC_DM_SCENE", "auto"),
                   help="sim scene: 'real' = the team's measured workspace (steel_team/simulation, rebot_sim: "
                        "table, walls, foam bin, the real cameras, can + cup); 'demo' = Seeed's demo table with "
                        "three blocks; 'auto' = real when rebot_sim is installed, else demo")
    c = p.add_argument_group("calibration test scene (sim only)")
    c.add_argument("--calib-sheet", metavar="PNG",
                   help="lay the printed sheet from calibrate_camera.py --make-sheet on the sim table "
                        "(objects are moved to the far edge) to test the camera calibration")
    c.add_argument("--sheet-pose", default="0.40,0.0,0", help="sheet centre x,y [m] and yaw [deg]")
    r = p.add_argument_group("real arm (UNTESTED ON HARDWARE)")
    r.add_argument("--confirm-real", action="store_true",
                   help="required with --robot real: you have read local_dm/README.md 'Real arm bring-up'")
    r.add_argument("--serial-port", default="/dev/ttyACM0",
                   help="Damiao serial bridge, e.g. /dev/ttyACM0 or /dev/tty.usbmodemXXXX")
    r.add_argument("--max-step-deg", type=float, default=6.0,
                   help="LeRobot max_relative_target per command [deg] (team default 6)")
    r.add_argument("--vel-deg", type=float, default=60.0, help="pos_vel speed limit [deg/s] (team default 60)")
    return p.parse_args()


# ── backends ────────────────────────────────────────────────────────────────

class SimBackend:
    kind = "sim"

    def __init__(self, args):
        import mujoco
        from teleop.rebot_model import GRIPPER_MAX, RebotSim

        self.mj = mujoco
        self.finger_max = float(GRIPPER_MAX)
        self.layout = dict(SIM_LAYOUT)
        self.scene = args.scene
        if self.scene in ("auto", "real") and not args.calib_sheet:
            try:
                from rebot_sim import RealSetupEnv
                self.scene = "real"
            except ImportError as e:
                if self.scene == "real":
                    sys.exit(f"--scene real needs the team's rebot_sim package (steel_team/simulation on main; "
                             f"run 'uv sync' in teleoperation): {e}")
                self.scene = "demo"
        else:
            self.scene = "demo"
        self.bins: dict = {}
        self.table_z = 0.03
        if self.scene == "real":
            # The team's replica of the real workspace. Its own gripper tuning and objects are kept as they are.
            self.sim = RealSetupEnv()
            self.layout = {}
            self.table_z = float(self.sim.ws.table_z)
        elif args.calib_sheet:
            import teleop.rebot_model as rm
            x, y, yaw_deg = (float(v) for v in args.sheet_pose.split(","))
            yaw = np.radians(yaw_deg)
            orig_build = rm.build_spec

            def build_with_sheet(path=rm.DM_MJCF):
                spec = orig_build(path)
                spec.add_texture(name="calib_sheet", type=mujoco.mjtTexture.mjTEXTURE_2D,
                                 file=str(Path(args.calib_sheet).resolve()))
                spec.add_material(name="calib_sheet_mat", textures=["", "calib_sheet"], texrepeat=[1, 1])
                # A4 lying on the table top (z = 0.03): local +x = page right, local +y = page top.
                spec.worldbody.add_geom(
                    name="calib_sheet", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.105, 0.1485, 0.01],
                    pos=[x, y, 0.0306], quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)],
                    material="calib_sheet_mat", contype=0, conaffinity=0)
                return spec

            rm.build_spec = build_with_sheet
            try:
                self.sim = RebotSim()
            finally:
                rm.build_spec = orig_build
            self.layout = {"red_cube": [0.64, -0.18, 0.055], "blue_block": [0.64, 0.18, 0.048],
                           "yellow_cylinder": [0.67, 0.0, 0.065]}
            up = np.array([-np.sin(yaw), np.cos(yaw)])            # page-top direction in the base frame
            a, b = np.array([x, y]) + 0.1335 * up, np.array([x, y]) - 0.1335 * up
            print(f"[dm-bridge] calibration sheet on the table: cross A at ({a[0]:.4f}, {a[1]:.4f}), "
                  f"cross B at ({b[0]:.4f}, {b[1]:.4f}), table_z 0.03  <- what a ruler would measure", flush=True)
        else:
            self.sim = RebotSim()
        self.m, self.d = self.sim.model, self.sim.data
        m = self.m
        if self.scene == "real":
            self.object_names = list(self.sim.objects)
        else:
            self.object_names = list(SIM_LAYOUT)
            g = self.sim.grip_act
            m.actuator_gainprm[g, 0] = GRIP_KP
            m.actuator_biasprm[g, 1] = -GRIP_KP
            m.actuator_biasprm[g, 2] = -GRIP_KV
            # The MJCF's "blue_block" renders cyan on a tan mat; make it a clear blue.
            for gid in range(m.ngeom):
                if m.geom_bodyid[gid] == m.body("blue_block").id:
                    m.geom_matid[gid] = -1
                    m.geom_rgba[gid] = [0.08, 0.22, 0.90, 1.0]
        self.grip_dof = m.joint("finger_left").dofadr[0]
        self.w, self.h = args.width, args.height
        m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), self.w)     # offscreen buffer for the camera
        m.vis.global_.offheight = max(int(m.vis.global_.offheight), self.h)
        self.renderer = None
        self._frame_cache: dict[str, tuple[float, dict]] = {}
        self.reset()

    def reset(self):
        mj, m, d = self.mj, self.m, self.d
        t_keep = float(d.time)
        self.sim.reset(np.array(DM_HOME_Q, dtype=float), gripper=self.finger_max)
        d.time = max(t_keep, float(d.time))   # keep the clock running forward: a jump back is treated as an accident
        if self.scene == "real":
            self.bins = {b: self.sim.bin_pos(b).tolist() for b in self.sim.bins}
        d.mocap_pos[self.sim.target_mocap] = [0.0, 0.0, -5.0]   # hide teleop's IK target marker
        for name, pos in self.layout.items():
            adr = m.jnt_qposadr[m.body(name).jntadr[0]]
            d.qpos[adr:adr + 3] = pos
            d.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
            dadr = m.jnt_dofadr[m.body(name).jntadr[0]]
            d.qvel[dadr:dadr + 6] = 0
        mj.mj_forward(m, d)
        self._frame_cache.clear()

    def set_objects(self, objects: dict):
        """Test helper: put sim objects at given poses. {name: {"pos": [x, y, z], "yaw": rad}}"""
        mj, m, d = self.mj, self.m, self.d
        for name, spec in objects.items():
            if name not in self.object_names:
                raise ValueError(f"unknown sim object {name!r} (have: {self.object_names})")
            j = m.body(name).jntadr[0]
            adr, dadr = m.jnt_qposadr[j], m.jnt_dofadr[j]
            yaw = float(spec.get("yaw", 0.0))
            d.qpos[adr:adr + 3] = spec["pos"]
            d.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
            d.qvel[dadr:dadr + 6] = 0
        mj.mj_forward(m, d)
        self._frame_cache.clear()

    def shuffle(self, seed=None) -> dict:
        """Real-workspace scene: move the objects to random places with the team's own randomizer
        (rebot_sim spawn_region). The arm is not touched. Returns the new object positions."""
        if self.scene != "real":
            raise ValueError("shuffle is for the real-workspace scene; the demo scene uses set_objects")
        self.sim.randomize_objects(np.random.default_rng(seed))
        dofs = [self.m.jnt_dofadr[self.m.body(n).jntadr[0]] for n in self.object_names]
        for dadr in dofs:
            self.d.qvel[dadr:dadr + 6] = 0
        self.mj.mj_forward(self.m, self.d)
        self._frame_cache.clear()
        return {n: self.d.xpos[self.m.body(n).id].tolist() for n in self.object_names}

    def step_to(self, t_target: float, max_steps: int = 25):
        n = 0
        while self.d.time < t_target and n < max_steps:
            self.mj.mj_step(self.m, self.d)
            n += 1

    def state(self) -> dict:
        d, s = self.d, self.sim
        return {
            "q": s.q_arm.tolist(),
            "dq": d.qvel[s.arm_dadr].tolist(),
            "finger": s.gripper,
            "finger_vel": float(d.qvel[self.grip_dof]),
            "ctrl": d.ctrl[s.arm_act].tolist(),
            "t": float(d.time),
            "fault": None,
            # Ground truth for VERIFICATION ONLY (WRC never reads this; it uses the camera).
            "objects": {n: d.xpos[self.m.body(n).id].tolist() for n in self.object_names},
            # real scene only: which bin compartment each object lies in (None = not in a bin)
            "in_bin": ({n: next((b for b in self.sim.bins if self.sim.in_bin(n, b)), None)
                        for n in self.object_names} if self.scene == "real" else {}),
            # how far each object leans from upright, in degrees (0 = standing as it started, 90 = on its side)
            "object_tilt_deg": {n: float(np.degrees(np.arccos(np.clip(d.xmat[self.m.body(n).id][8], -1, 1))))
                                for n in self.object_names},
        }

    def set_target(self, q):
        lo, hi = self.m.actuator_ctrlrange[self.sim.arm_act].T
        self.d.ctrl[self.sim.arm_act] = np.clip(np.asarray(q, dtype=float), lo, hi)

    def set_gripper(self, f: float):
        self.d.ctrl[self.sim.grip_act] = float(np.clip(f, 0.0, self.finger_max))

    def hold(self):
        self.d.ctrl[self.sim.arm_act] = self.sim.q_arm

    def resume(self):
        pass

    def frame(self, cam: str) -> dict:
        mj, m, d = self.mj, self.m, self.d
        cached = self._frame_cache.get(cam)
        if cached and d.time - cached[0] < 0.04:
            return cached[1]
        if self.renderer is None:
            self.renderer = mj.Renderer(m, self.h, self.w)
        r = self.renderer
        r.update_scene(d, camera=cam)
        bgr = np.ascontiguousarray(r.render()[:, :, ::-1])
        r.enable_depth_rendering()
        r.update_scene(d, camera=cam)
        depth = np.ascontiguousarray(r.render(), dtype=np.float32)
        r.disable_depth_rendering()
        cid = m.camera(cam).id
        f = (self.h / 2) / np.tan(np.radians(m.cam_fovy[cid]) / 2)
        K = [[f, 0, self.w / 2], [0, f, self.h / 2], [0, 0, 1]]
        # MuJoCo camera axes: x right, y up, looks along -z. OpenCV: x right, y down, z forward.
        T = np.eye(4)
        T[:3, :3] = d.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
        T[:3, 3] = d.cam_xpos[cid]
        out = {"header": {"ok": True, "w": self.w, "h": self.h, "K": K, "T_base_cam": T.tolist(),
                          "t": float(d.time)},
               "blobs": [bgr.tobytes(), depth.tobytes()]}
        self._frame_cache[cam] = (float(d.time), out)
        return out

    def cameras(self):
        return [self.m.camera(i).name for i in range(self.m.ncam)]

    def scene_info(self) -> dict:
        """What a client needs to know about the scene (measured places, not object positions)."""
        info = {"scene": self.scene, "table_z": self.table_z, "object_names": self.object_names, "bins": {}}
        if self.scene == "real":
            cfg = self.sim.cfg["bin"]
            for b in self.sim.bins:
                c, half = self.sim.bin_pos(b), self.m.site(b).size
                info["bins"][b] = {"center": [float(c[0]), float(c[1])], "half": [float(half[0]), float(half[1])],
                                   "floor_z": float(c[2]), "rim_z": self.table_z + float(cfg["height"])}
        return info


class RealBackend:
    """Physical B601-DM through LeRobot, wired the way the team's own real-arm scripts do it
    (steel_team/teleoperation/go_ready.py): pos_vel control, the MEASURED gripper range,
    commands streamed at 50 Hz. The author of this bridge has NOT run it on hardware."""
    kind = "real"
    MAX_GAP_M = 0.091                    # jaw gap when fully open, from the MJCF -- verify on the arm
    GAP_OFFSET_M = 0.009                 # WRC's finger model: gap = 2 * finger - 0.009
    LAG_LIMIT_RAD = float(np.radians(20.0))
    LAG_TIME_S = 1.5

    def __init__(self, args):
        from lerobot.robots.rebot_b601_follower import RebotB601Follower, RebotB601FollowerRobotConfig
        from teleop import gripper_limits, rebot_real
        from teleop.rebot_model import GRIPPER_MAX, RebotSim

        limits = gripper_limits.load()
        if limits is None:
            sys.exit("Gripper range not measured. Run teleoperation/tools/measure_gripper.py --port "
                     f"{args.serial_port} first (the team's real-arm scripts need it too).")
        self.rr = rebot_real
        self.g_lo, self.g_hi = gripper_limits.bounds(limits)
        self.finger_max = float(GRIPPER_MAX)
        self.robot = RebotB601Follower(RebotB601FollowerRobotConfig(
            port=args.serial_port, joint_limits={"gripper": (self.g_lo, self.g_hi)},
            max_relative_target=args.max_step_deg, control_mode="pos_vel",
            pos_vel_velocity=args.vel_deg, disable_torque_on_disconnect=False))
        self.robot.connect()
        self.sim = RebotSim()                       # MuJoCo is only a mirror of the measured pose
        self.m, self.d = self.sim.model, self.sim.data
        obs = self.robot.get_observation()
        self.q = rebot_real.obs_to_q(obs)
        self.grip_deg = float(obs["gripper.pos"])
        self.q_cmd, self.grip_cmd = self.q.copy(), self.grip_deg     # start by holding the current pose
        self.dq = np.zeros(6)
        self.df = 0.0
        self.fault = None
        self._lag_since = None
        self._t_prev = time.perf_counter()

    # gripper: motor degrees <-> the bridge's "finger" value (assumes the jaw gap is linear in motor angle)
    def _closing(self, deg: float) -> float:
        span = self.rr.GRIPPER_CLOSED_DEG - self.rr.GRIPPER_OPEN_DEG
        return float(np.clip((deg - self.rr.GRIPPER_OPEN_DEG) / span, 0.0, 1.0))

    def _finger_from_deg(self, deg: float) -> float:
        return (self.MAX_GAP_M * (1.0 - self._closing(deg)) + self.GAP_OFFSET_M) / 2.0

    def _deg_from_finger(self, f: float) -> float:
        gap = float(np.clip(2.0 * f - self.GAP_OFFSET_M, 0.0, self.MAX_GAP_M))
        deg = self.rr.gripper_from_closing(1.0 - gap / self.MAX_GAP_M)
        return float(np.clip(deg, self.g_lo, self.g_hi))

    def tick(self):
        """One 50 Hz cycle: command, read back, mirror into the viewer, check the lag."""
        self.robot.send_action(self.rr.q_to_action(self.q_cmd, self.grip_cmd))
        obs = self.robot.get_observation()
        q, grip = self.rr.obs_to_q(obs), float(obs["gripper.pos"])
        now = time.perf_counter()
        dt = max(now - self._t_prev, 1e-3)
        f_new, f_old = self._finger_from_deg(grip), self._finger_from_deg(self.grip_deg)
        self.dq, self.df = (q - self.q) / dt, (f_new - f_old) / dt
        self.q, self.grip_deg, self._t_prev = q, grip, now
        self.sim.mirror(q, min(f_new, self.finger_max))
        # Same guard as the team's ready_move.py: a large, persistent lag means trouble -> hold.
        if float(np.max(np.abs(self.q_cmd - q))) > self.LAG_LIMIT_RAD:
            self._lag_since = self._lag_since or now
            if now - self._lag_since > self.LAG_TIME_S and self.fault is None:
                self.fault = "arm lags the commanded pose by more than 20 deg; holding"
                self.q_cmd = q.copy()
                print(f"[dm-bridge] FAULT: {self.fault}", flush=True)
        else:
            self._lag_since = None

    def state(self) -> dict:
        return {"q": self.q.tolist(), "dq": self.dq.tolist(), "finger": self._finger_from_deg(self.grip_deg),
                "finger_vel": float(self.df), "ctrl": self.q_cmd.tolist(), "t": time.time(),
                "gripper_deg": self.grip_deg, "fault": self.fault, "objects": {}}

    def set_target(self, q):
        if self.fault:
            raise RuntimeError(f"fault latched ({self.fault}); send 'resume' after checking the arm")
        self.q_cmd = np.asarray(q, dtype=float).copy()

    def set_gripper(self, f: float):
        self.grip_cmd = self._deg_from_finger(float(f))

    def hold(self):
        self.q_cmd = self.q.copy()

    def resume(self):
        self.fault, self._lag_since = None, None
        self.q_cmd = self.q.copy()

    def frame(self, cam: str) -> dict:
        raise RuntimeError("no simulated camera with --robot real: give WRC a real camera profile")

    def reset(self):
        raise RuntimeError("reset is sim-only")

    def set_objects(self, objects: dict):
        raise RuntimeError("set_objects is sim-only")

    def cameras(self):
        return []

    def scene_info(self) -> dict:
        return {"scene": "real-arm", "table_z": None, "object_names": [], "bins": {}}

    def close(self):
        self.hold()
        self.robot.disconnect()


# ── socket server (I/O threads) + command pump (runs in the robot/sim thread) ────

class Bridge:
    def __init__(self, backend, host: str, port: int):
        self.backend = backend
        self.q: queue.Queue = queue.Queue()
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.srv.bind((host, port))
        except OSError:
            sys.exit(f"ERROR: {host}:{port} is already in use -- a DM bridge server is probably already "
                     "running. Stop it first: pkill -f local_dm/dm_bridge_server.py")
        self.srv.listen(8)
        self.n_clients = 0
        threading.Thread(target=self._accept, daemon=True, name="bridge-accept").start()

    def _accept(self):
        while True:
            conn, _ = self.srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            threading.Thread(target=self._client, args=(conn,), daemon=True, name="bridge-client").start()

    def _client(self, conn: socket.socket):
        self.n_clients += 1
        try:
            for line in conn.makefile("rb"):
                box: dict = {}
                done = threading.Event()
                try:
                    req = json.loads(line)
                except json.JSONDecodeError:
                    box = {"header": {"ok": False, "error": "bad json"}}
                else:
                    self.q.put((req, box, done))
                    if not done.wait(5.0):
                        box = {"header": {"ok": False, "error": "server loop timeout"}}
                header = dict(box.get("header", {"ok": False, "error": "no reply"}))
                blobs = box.get("blobs", [])
                header["blobs"] = [len(b) for b in blobs]
                conn.sendall(json.dumps(header).encode() + b"\n" + b"".join(blobs))
        except (ConnectionError, OSError):
            pass
        finally:
            self.n_clients -= 1
            conn.close()

    def pump(self):
        """Execute queued requests. Call from the thread that owns the robot / MuJoCo data."""
        while True:
            try:
                req, box, done = self.q.get_nowait()
            except queue.Empty:
                return
            try:
                box.update(self._handle(req))
            except Exception as e:  # noqa: BLE001 -- report to the client, keep serving
                box["header"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            done.set()

    def _handle(self, req: dict) -> dict:
        b, op = self.backend, req.get("op")
        if op == "state":
            return {"header": {"ok": True, **b.state()}}
        if op == "target":
            b.set_target(req["q"])
            return {"header": {"ok": True}}
        if op == "gripper":
            b.set_gripper(float(req["f"]))
            return {"header": {"ok": True}}
        if op == "hold":
            b.hold()
            return {"header": {"ok": True}}
        if op == "resume":
            b.resume()
            return {"header": {"ok": True}}
        if op == "frame":
            return b.frame(str(req.get("cam", "overhead_rgb")))
        if op == "reset":
            b.reset()
            return {"header": {"ok": True}}
        if op == "test_viewer_reset" and b.kind == "sim":   # test hook: what Backspace in the 3D window does
            b.mj.mj_resetData(b.m, b.d)
            return {"header": {"ok": True}}
        if op == "set_objects":
            b.set_objects(req["objects"])
            return {"header": {"ok": True}}
        if op == "shuffle" and b.kind == "sim":
            return {"header": {"ok": True, "objects": b.shuffle(req.get("seed"))}}
        if op == "info":
            lim = b.m.actuator_ctrlrange[b.sim.arm_act]      # the ranges the arm accepts (real soft limits)
            return {"header": {"ok": True, "robot": b.kind, "home_q": DM_HOME_Q, "finger_max": b.finger_max,
                               "cameras": b.cameras(), "joint_lo": lim[:, 0].tolist(),
                               "joint_hi": lim[:, 1].tolist(), **b.scene_info()}}
        return {"header": {"ok": False, "error": f"unknown op {op!r}"}}


def main() -> int:
    args = parse_args()
    teleop = Path(args.teleop_dir).resolve()
    if not (teleop / "teleop" / "rebot_model.py").exists():
        sys.exit(f"teleoperation folder not found at {teleop} (set --teleop-dir or STEEL_TEAM_TELEOP_DIR)")
    sys.path.insert(0, str(teleop))
    if args.calib_sheet:
        args.calib_sheet = str(Path(args.calib_sheet).resolve())
    os.chdir(tempfile.gettempdir())       # MuJoCo drops MUJOCO_LOG.TXT into the CWD: keep it out of the repos

    if args.robot == "real":
        if not args.confirm_real:
            sys.exit("--robot real is UNTESTED ON HARDWARE. Read local_dm/README.md 'Real arm bring-up', "
                     "keep a hand on the e-stop / power switch, then add --confirm-real.")
        backend = RealBackend(args)
    else:
        backend = SimBackend(args)
    bridge = Bridge(backend, args.host, args.port)
    print(f"[dm-bridge] {backend.kind} B601-DM ready on tcp://{args.host}:{args.port}"
          + ("" if backend.kind == "sim" else "  (REAL ARM -- untested code path)"), flush=True)

    viewer = None
    if not args.headless:
        import mujoco.viewer as mjviewer
        viewer = mjviewer.launch_passive(backend.m, backend.d, show_left_ui=False, show_right_ui=False)
        viewer.cam.lookat[:] = [0.30, 0.0, 0.12]
        viewer.cam.distance = 1.15
        viewer.cam.azimuth = 215
        viewer.cam.elevation = -28

    def _terminate(_sig, _frm):            # `kill` must also end in the clean shutdown below
        raise KeyboardInterrupt
    try:
        signal.signal(signal.SIGTERM, _terminate)
    except ValueError:                      # not the main thread (mjpython): Ctrl+C / closing the window still work
        pass

    def _check_window(v):
        # Closing the 3D window must NOT stop the robot: keep serving without it (Ctrl+C stops the server).
        if v is not None and not v.is_running():
            print("[dm-bridge] 3D window closed; the simulator keeps running. Ctrl+C in this terminal stops it.",
                  flush=True)
            return None
        return v

    t_start = time.perf_counter()
    next_sync = 0.0
    try:
        if backend.kind == "sim":
            t0, sim0 = time.perf_counter(), backend.d.time
            snap = None
            while True:
                viewer = _check_window(viewer)
                now = time.perf_counter()
                if args.duration is not None and now - t_start > args.duration:
                    break
                d = backend.d
                bridge.pump()
                if snap is not None and d.time + 1e-9 < snap[0]:
                    # Time went backwards: a key pressed in the 3D window (Backspace = MuJoCo's own reset)
                    # wiped the scene and the arm targets. Put everything back as it was.
                    d.time = snap[0]
                    d.qpos[:], d.qvel[:], d.ctrl[:] = snap[1], snap[2], snap[3]
                    d.mocap_pos[:], d.mocap_quat[:] = snap[4], snap[5]
                    backend.mj.mj_forward(backend.m, d)
                    print("[dm-bridge] a key press in the 3D window reset the scene; restored it. "
                          "Type tasks in the web page, not in the 3D window.", flush=True)
                target = sim0 + (now - t0)
                if target - backend.d.time > 0.2:        # fell behind (e.g. laptop sleep): resync
                    t0, sim0 = now, backend.d.time
                    target = sim0
                backend.step_to(target)
                if snap is None or d.time >= snap[0]:   # never snapshot a wiped state
                    snap = (d.time, d.qpos.copy(), d.qvel.copy(), d.ctrl.copy(), d.mocap_pos.copy(), d.mocap_quat.copy())
                if viewer is not None and now >= next_sync:
                    viewer.sync()
                    next_sync = now + 1 / 50
                time.sleep(0.001)
        else:
            period = 1 / 50
            next_tick = time.perf_counter()
            while True:
                viewer = _check_window(viewer)
                now = time.perf_counter()
                if args.duration is not None and now - t_start > args.duration:
                    break
                bridge.pump()
                if now >= next_tick:
                    backend.tick()
                    next_tick += period
                    if viewer is not None:
                        viewer.sync()
                time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        if backend.kind == "real":
            print("[dm-bridge] disconnecting: the arm may go limp when commands stop -- SUPPORT IT", flush=True)
            backend.close()
        if viewer is not None:
            viewer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
