#!/usr/bin/env python
"""3D MuJoCo mirror of the WRC pipeline (added by the hackathon setup; NOT upstream).

Shows what WRC's (mock) arm is doing on Seeed's official reBot B601-RS MuJoCo
model. WRC streams its joint angles over UDP (local_mock/mirror_publisher.py,
started by local_mock/dashboard_mock.py); this viewer copies them 1:1 onto the
3D arm. The RS MJCF and WRC's RS URDF have identical kinematics (verified: same
TCP pose to ~4e-8 m for 10 random joint vectors), so no IK or remapping is done.

What you see:
  * the arm               = WRC's commanded/mock joint angles (kinematic playback, no physics)
  * solid red cube        = carried while WRC's jaws are closed on it, rests where the jaws open
  * translucent boxes     = WRC's world-model beliefs (red = visible now, grey = remembered)
  * green square          = WRC's default drop zone
Nothing here controls WRC or any hardware: it only listens on 127.0.0.1.

Run with the teleoperation venv (needs mujoco; macOS needs mjpython for the window):
  steel_team/teleoperation/.venv/bin/mjpython local_mock/wrc_mujoco_mirror.py
  .../.venv/bin/python local_mock/wrc_mujoco_mirror.py --headless --snapshot-dir DIR --duration 20
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = (HERE.parents[1] / "steel_team" / "teleoperation" / "assets" / "reBot-DevArm"
                 / "Rebot_Arm_description" / "RS" / "mjcf" / "rs_arm.xml")
WRC_HOME_Q = [0.0, 1.2, 1.2, 0.0, 0.75, 0.0]     # configs/arms/mock.yaml home_q
CUBE_HALF = np.array([0.025, 0.035, 0.025])        # mock camera: ~5 x 7 cm footprint, 5 cm tall
N_GHOSTS = 4
GRIP_STROKE = 0.05                                 # joint7: 0 = closed, 0.05 = fully open


def build_model(model_path: Path) -> mujoco.MjModel:
    if not model_path.exists():
        sys.exit(f"RS model not found at {model_path}. Run steel_team/teleoperation/scripts/setup_assets.sh")
    spec = mujoco.MjSpec.from_file(str(model_path))
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, rgb1=[0.82, 0.80, 0.76],
                     rgb2=[0.74, 0.72, 0.68], width=300, height=300)
    spec.add_material(name="table_mat", textures=["", "grid"], texrepeat=[8, 8], reflectance=0.05)
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, rgb1=[0.3, 0.45, 0.6],
                     rgb2=[0.05, 0.07, 0.1], width=256, height=1536)
    wb = spec.worldbody
    # WRC base frame: table surface is z = 0 (configs/demo.yaml safety.table_z).
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.8, 0.8, 0.02],
                material="table_mat", contype=0, conaffinity=0)
    directional = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    wb.add_light(pos=[0.3, -0.3, 1.5], dir=[-0.1, 0.2, -1], type=directional)
    wb.add_light(pos=[-0.3, 0.4, 1.2], dir=[0.2, -0.3, -1], type=directional, diffuse=[0.3, 0.3, 0.3])

    cube = wb.add_body(name="wrc_cube", mocap=True, pos=[0.29, 0.0, CUBE_HALF[2]])
    cube.add_geom(name="wrc_cube_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=CUBE_HALF,
                  rgba=[0.85, 0.12, 0.10, 1], contype=0, conaffinity=0)
    for i in range(N_GHOSTS):
        g = wb.add_body(name=f"ghost{i}", mocap=True, pos=[0, 0, -1])
        g.add_geom(name=f"ghost{i}_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=CUBE_HALF * 1.08,
                   rgba=[1, 0.3, 0.3, 0.0], contype=0, conaffinity=0)
    dz = wb.add_body(name="drop_zone", mocap=True, pos=[0.18, -0.17, 0.0005])
    for name, size, pos in (("n", [0.045, 0.003, 0.0005], [0, 0.045, 0]),
                            ("s", [0.045, 0.003, 0.0005], [0, -0.045, 0]),
                            ("e", [0.003, 0.045, 0.0005], [0.045, 0, 0]),
                            ("w", [0.003, 0.045, 0.0005], [-0.045, 0, 0])):
        dz.add_geom(name=f"dz_{name}", type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=pos,
                    rgba=[0.15, 0.75, 0.3, 1], contype=0, conaffinity=0)
    return spec.compile()


class Mirror:
    def __init__(self, m: mujoco.MjModel):
        self.m, self.d = m, mujoco.MjData(m)
        self.qadr = [m.joint(f"joint{i}").qposadr[0] for i in range(1, 7)]
        self.gadr = [m.joint(j).qposadr[0] for j in ("joint7", "joint_left", "joint_right")]
        self.tcp = m.site("tcp").id
        self.cube_mocap = m.body_mocapid[m.body("wrc_cube").id]
        self.dz_mocap = m.body_mocapid[m.body("drop_zone").id]
        self.ghosts = [(m.body_mocapid[m.body(f"ghost{i}").id], m.geom(f"ghost{i}_geom").id)
                       for i in range(N_GHOSTS)]
        self.cube_inited = False
        self.holding = None
        self.attached = False
        self.offset_p = np.zeros(3)
        self.offset_R = np.eye(3)
        self.cube_R = np.eye(3)
        self.status = "waiting for WRC (start ./START_WRC.sh dashboard)"
        self.last_msg_t = 0.0
        self.events: list[str] = []
        self.apply({"q": WRC_HOME_Q, "grip": 0.0, "holding": None})
        self.status = "waiting for WRC (start ./START_WRC.sh dashboard)"

    def _tcp_pose(self):
        return self.d.site_xpos[self.tcp].copy(), self.d.site_xmat[self.tcp].reshape(3, 3).copy()

    def _set_cube(self, p, R):
        self.d.mocap_pos[self.cube_mocap] = p
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, R.reshape(-1))
        self.d.mocap_quat[self.cube_mocap] = q
        self.cube_R = R

    def apply(self, msg: dict) -> None:
        d = self.d
        d.qpos[self.qadr] = msg["q"]
        g = float(np.clip(GRIP_STROKE * (1.0 - float(msg.get("grip", 0.0))), 0.0, GRIP_STROKE))
        d.qpos[self.gadr] = g
        mujoco.mj_kinematics(self.m, d)
        p_tcp, R_tcp = self._tcp_pose()

        if "drop_zone" in msg:
            self.d.mocap_pos[self.dz_mocap][:2] = msg["drop_zone"]
        beliefs = msg.get("beliefs")
        if beliefs is not None:
            if not self.cube_inited:
                for b in beliefs:
                    if "cube" in b["label"] or "block" in b["label"]:
                        x, y, z = b["position"]
                        self._set_cube(np.array([x, y, max(CUBE_HALF[2], z - CUBE_HALF[2])]), np.eye(3))
                        self.cube_inited = True
                        break
            for i, (mid, gid) in enumerate(self.ghosts):
                if i < len(beliefs):
                    x, y, z = beliefs[i]["position"]
                    d.mocap_pos[mid] = [x, y, max(CUBE_HALF[2], z - CUBE_HALF[2])]
                    visible = beliefs[i].get("state") == "visible"
                    self.m.geom_rgba[gid] = [1, 0.35, 0.35, 0.28] if visible else [0.7, 0.7, 0.7, 0.18]
                else:
                    d.mocap_pos[mid] = [0, 0, -1]
                    self.m.geom_rgba[gid] = [0, 0, 0, 0]

        # The cube is carried while the jaws are closed ON it. WRC itself only
        # sets `holding` after it lifts and verifies the grip (~1 s later), so
        # the jaws are the earlier, truthful signal. A fully closed gripper
        # (grip > 0.95) is WRC's "air grasp": nothing between the jaws.
        holding = msg.get("holding")
        grip = float(msg.get("grip", 0.0))
        cube_p = d.mocap_pos[self.cube_mocap].copy()
        jaws_on_object = 0.1 < grip < 0.95
        if not self.attached:
            near = np.linalg.norm(cube_p - p_tcp) < 0.05
            if jaws_on_object and (near or holding):
                if not near:          # WRC grasped where its (static) camera sees a cube
                    self.events.append("WRC grasped away from the drawn cube -> cube moved into the jaws")
                    cube_p = p_tcp.copy()
                self.offset_p = R_tcp.T @ (cube_p - p_tcp)
                self.offset_R = R_tcp.T @ self.cube_R
                self.attached = True
                self.events.append(f"jaws closed on the cube at ({p_tcp[0]:.2f}, {p_tcp[1]:.2f}, {p_tcp[2]:.3f})")
        if self.attached:
            if jaws_on_object:
                self._set_cube(p_tcp + R_tcp @ self.offset_p, R_tcp @ self.offset_R)
            else:                     # jaws opened (or closed fully = lost it): rest on the table
                self.attached = False
                yaw = np.arctan2(self.cube_R[1, 0], self.cube_R[0, 0])
                c, s = np.cos(yaw), np.sin(yaw)
                self._set_cube(np.array([cube_p[0], cube_p[1], CUBE_HALF[2]]),
                               np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]))
                self.events.append(f"released at ({cube_p[0]:.2f}, {cube_p[1]:.2f})")
        self.holding = holding
        mujoco.mj_forward(self.m, d)
        task = msg.get("task")
        self.status = (f"WRC task: {task}" if task else "WRC idle") + \
                      f" | holding: {holding or '-'} | tcp: ({p_tcp[0]:.2f}, {p_tcp[1]:.2f}, {p_tcp[2]:.2f})"
        self.last_msg_t = time.monotonic()


def open_socket(port: int) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind(("127.0.0.1", port))
    except OSError:
        sys.exit(f"ERROR: udp://127.0.0.1:{port} is already in use — another WRC 3D window is probably "
                 "open. Close it or run: pkill -f local_mock/wrc_mujoco_mirror.py")
    s.setblocking(False)
    return s


def drain(sock: socket.socket):
    latest, beliefs = None, None
    while True:
        try:
            data, _ = sock.recvfrom(65536)
        except BlockingIOError:
            break
        msg = json.loads(data)
        if "beliefs" in msg:
            beliefs = msg["beliefs"]
        latest = msg
    if latest is not None and beliefs is not None:
        latest["beliefs"] = beliefs
    return latest


def set_camera(cam):
    cam.lookat[:] = [0.22, -0.02, 0.12]
    cam.distance = 1.05
    cam.azimuth = 215
    cam.elevation = -28


def main() -> int:
    p = argparse.ArgumentParser(description="WRC -> MuJoCo 3D mirror")
    p.add_argument("--port", type=int, default=9870)
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--headless", action="store_true", help="no window; render snapshots instead")
    p.add_argument("--snapshot-dir", default=None)
    p.add_argument("--snapshot-every", type=float, default=1.0, help="[s] in headless mode")
    p.add_argument("--duration", type=float, default=None)
    args = p.parse_args()

    m = build_model(Path(args.model))
    mirror = Mirror(m)
    sock = open_socket(args.port)
    print(f"[mirror] listening on udp://127.0.0.1:{args.port}; model {Path(args.model).name}", flush=True)
    t0 = time.monotonic()

    def tick():
        msg = drain(sock)
        if msg is not None:
            mirror.apply(msg)
        while mirror.events:
            print(f"[mirror] {mirror.events.pop(0)}", flush=True)
        if time.monotonic() - mirror.last_msg_t > 2.0 and mirror.last_msg_t:
            mirror.status = "no data from WRC for >2 s (is the dashboard running?)"

    if args.headless:
        out = Path(args.snapshot_dir or "mirror_snapshots")
        out.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(m, height=480, width=640)
        cam = mujoco.MjvCamera()
        set_camera(cam)
        n, next_snap = 0, 0.0
        while args.duration is None or time.monotonic() - t0 < args.duration:
            tick()
            if time.monotonic() - t0 >= next_snap:
                renderer.update_scene(mirror.d, camera=cam)
                img = renderer.render()
                ppm = out / f"frame_{n:03d}.ppm"
                with open(ppm, "wb") as f:
                    f.write(b"P6 %d %d 255\n" % (img.shape[1], img.shape[0]) + img.tobytes())
                subprocess.run(["sips", "-s", "format", "png", str(ppm), "--out",
                                str(ppm.with_suffix(".png"))], capture_output=True)
                ppm.unlink(missing_ok=True)
                print(f"[mirror] {ppm.with_suffix('.png').name}: {mirror.status}", flush=True)
                n += 1
                next_snap += args.snapshot_every
            time.sleep(0.01)
        return 0

    import mujoco.viewer as mjviewer
    with mjviewer.launch_passive(m, mirror.d, show_left_ui=False, show_right_ui=False) as viewer:
        set_camera(viewer.cam)
        while viewer.is_running():
            if args.duration is not None and time.monotonic() - t0 > args.duration:
                break
            with viewer.lock():
                tick()
            try:
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                  "WRC 3D mirror", mirror.status))
            except Exception:
                pass
            viewer.sync()
            time.sleep(1 / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
