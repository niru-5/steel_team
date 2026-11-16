"""WRC backends for the reBot B601-DM bridge (hackathon DM port; NOT upstream WRC).

Adds, without editing any upstream WRC file:
  * arm type   ``rebot_dm_bridge``  -> DmBridgeArm   (configs/arms/rebot_dm.yaml)
  * camera type ``dm_bridge``       -> BridgeCamera  (configs/cameras/dm_sim_overhead.yaml)

Both talk to local_dm/dm_bridge_server.py (MuJoCo physics of the DM arm, or the real
arm). Call ``register()`` once before ``build_runtime``.

To upstream this properly: move the two classes into wrc_demo/control and
wrc_demo/perception and add a lazy-import branch to make_arm() / make_camera()
(CLAUDE.md "Adding a camera/arm backend").
"""
from __future__ import annotations

import dataclasses
import json
import os
import socket
import sys
import threading
import time

import numpy as np

from wrc_demo.control.mock_arm import MockArm, _min_jerk
from wrc_demo.perception.camera_base import CameraBase, CameraError
from wrc_demo.safety.harness import SafetyHarness, SafetyLimits
from wrc_demo.types import Frame, RobotState, SafetyViolation

# DM gripper geometry (Seeed MJCF): each finger slides f in [0, 0.05] m; the gap
# between the inner faces of the jaws is 2*f - 0.009 m (0.091 m fully open).
FINGER_MAX_M = 0.05
GAP_MAX_M = 0.091
GAP_OFFSET_M = 0.009


def bridge_port(profile_port=None) -> int:
    """$WRC_DM_BRIDGE_PORT wins over the profile, so tests can use a private simulator."""
    return int(os.environ.get("WRC_DM_BRIDGE_PORT") or profile_port or 9881)


class BridgeClient:
    """One blocking request/response connection to dm_bridge_server.py."""

    def __init__(self, host: str = "127.0.0.1", port: int | None = None, timeout_s: float = 5.0):
        self._addr = (host, bridge_port(port) if port is None else int(port))
        self._timeout = timeout_s
        self._lock = threading.Lock()
        self._sock = None
        self._file = None

    def _connect(self):
        self._sock = socket.create_connection(self._addr, timeout=self._timeout)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._file = self._sock.makefile("rb")

    def close(self):
        with self._lock:
            try:
                if self._sock is not None:
                    self._sock.close()
            finally:
                self._sock = self._file = None

    def call(self, op: str, **kw) -> tuple[dict, list[bytes]]:
        with self._lock:
            try:
                if self._sock is None:
                    self._connect()
                self._sock.sendall(json.dumps({"op": op, **kw}).encode() + b"\n")
                line = self._file.readline()
                if not line:
                    raise ConnectionError("bridge closed the connection")
                header = json.loads(line)
                blobs = [self._file.read(n) for n in header.get("blobs", [])]
            except (OSError, ValueError) as e:
                self._sock = self._file = None
                raise RuntimeError(
                    f"DM bridge at {self._addr[0]}:{self._addr[1]} not reachable ({e}). "
                    "Start it first: ./START_WRC.sh dm  (or local_dm/dm_bridge_server.py)") from e
        if not header.get("ok"):
            raise RuntimeError(f"DM bridge refused {op!r}: {header.get('error')}")
        return header, blobs


class DmBridgeArm(MockArm):
    """SafeArm-shaped driver for the B601-DM behind the bridge.

    Reuses MockArm's public surface (move_joints / move_joints_path / move_cartesian /
    close_gripper_two_stage / stop / resume ...) but, unlike the mock:
      * every 50 Hz waypoint is vetted by the REAL SafetyHarness built from
        cfg.safety (workspace box, table clearance, joint limits, velocity cap)
        and then sent to the arm in real time,
      * motion durations are stretched so the min-jerk peak stays under the
        velocity cap (what SafeArm does for the RS arm),
      * joint state and the gripper opening are MEASURED from the arm, so a
        grasp is "held" only if the jaws physically stall on an object.
    """

    def __init__(self, cfg, kinematics):
        super().__init__(cfg.arm, kinematics=kinematics)
        self.harness = SafetyHarness(SafetyLimits.from_config(cfg.safety), kinematics)
        b = cfg.arm.get("bridge") or {}
        self._client = BridgeClient(str(b.get("host", "127.0.0.1")), bridge_port(b.get("port")))
        self._vmax = 0.85 * float(cfg.safety.get("max_joint_vel", 1.2))
        self._dt = 0.02
        self._settle_tol = float(cfg.arm.get("settle_tol_rad", 0.02))
        # A motion that ends further than this from its target did not work (blocked, or the
        # arm clipped the command). Wide enough for contact compliance, far below a real miss.
        self._reach_tol = float(cfg.arm.get("reach_tol_rad", 0.10))
        self._arm_lo = self._arm_hi = None
        self.last_close_tcp_z = None
        self.last_close_pitch = None
        self.scene = {}                       # what the bridge says about its scene (bins, table height, ...)
        self.robot_kind = "?"

    # ── lifecycle / state ────────────────────────────────────────────────
    def connect(self) -> None:
        info, _ = self._client.call("info")
        self.robot_kind = str(info.get("robot", "?"))
        self.scene = {k: info.get(k) for k in ("scene", "table_z", "bins", "object_names")}
        lo_b, hi_b = np.asarray(info.get("joint_lo", []), float), np.asarray(info.get("joint_hi", []), float)
        if lo_b.size == self.n_joints and self.kin is not None:
            # WRC must not plan outside what the arm accepts: the arm clips such commands and
            # the gripper then arrives in the right place but turned wrong.
            lo_u, hi_u = self.kin.joint_limits
            bad = [f"joint{i + 1}" for i in range(self.n_joints)
                   if lo_u[i] < lo_b[i] - 0.01 or hi_u[i] > hi_b[i] + 0.01]
            if bad:
                raise RuntimeError(
                    f"WRC's arm model lets {', '.join(bad)} move further than the arm accepts. "
                    "Regenerate it: python local_dm/make_dm_urdf.py")
            self._arm_lo, self._arm_hi = lo_b, hi_b
        super().connect()
        self._q = np.asarray(self._client.call("state")[0]["q"], dtype=float)

    def disconnect(self) -> None:
        try:
            self._client.call("hold")
        except RuntimeError:
            pass
        self._client.close()
        super().disconnect()

    def _closed_frac(self, finger_m: float) -> float:
        gap = 2.0 * float(finger_m) - GAP_OFFSET_M
        return float(np.clip(1.0 - gap / GAP_MAX_M, 0.0, 1.0))

    def get_state(self) -> RobotState:
        st = self._client.call("state")[0]
        closed = self._closed_frac(st["finger"])
        self._gripper = self._grip_open + (self._grip_closed - self._grip_open) * closed
        return RobotState(
            q=np.asarray(st["q"], dtype=float),
            dq=np.asarray(st["dq"], dtype=float),
            tau=np.zeros(self.n_joints),
            gripper_pos=self._gripper,
            t=time.monotonic(),
        )

    # ── motion ───────────────────────────────────────────────────────────
    def move_joints(self, q_target, duration_s: float = 2.0) -> bool:
        q_target = np.asarray(q_target, dtype=float).reshape(-1)[: self.n_joints]
        self.harness.begin_motion()          # raises SafetyViolation if e-stopped / perception stale
        ok = self._stream([self.get_state().q, q_target], duration_s)
        return ok and not self.harness.estopped

    def _abort(self, reason: str) -> bool:
        self._stopped = True
        self.harness.estop(reason=reason)
        try:
            self._client.call("hold")
        except RuntimeError:
            pass
        return False

    def _stream(self, q_pts: list, duration_s: float) -> bool:
        if not q_pts:
            return True
        if self._stopped:
            return False
        h = self.harness
        try:
            h.begin_motion()
        except SafetyViolation:
            return False
        q_meas = self.get_state().q
        tail = [np.asarray(p, dtype=float).reshape(-1)[: self.n_joints] for p in q_pts]
        pts = [q_meas] + (tail[1:] if len(tail) > 1 else tail)
        seg_T = max(duration_s, 0.05) / max(len(pts) - 1, 1)
        prev = q_meas.copy()
        next_tick = time.monotonic()
        for a, b in zip(pts[:-1], pts[1:]):
            # min-jerk peak velocity = 1.875 * dq / T  -> stretch T to respect the cap
            T = max(seg_T, 1.875 * float(np.abs(b - a).max()) / self._vmax, self._dt)
            n = max(1, int(np.ceil(T / self._dt)))
            for k in range(1, n + 1):
                if self._stop_send.is_set() or self._stopped or h.estopped:
                    self._client.call("hold")
                    return False
                q = a + (b - a) * _min_jerk(k / n)
                if self._arm_lo is not None and (np.any(q < self._arm_lo - 1e-3) or np.any(q > self._arm_hi + 1e-3)):
                    return self._abort("dm arm: target outside the arm's joint range")
                try:
                    approved = bool(h.approve(prev, q, dt=self._dt))
                    why = "harness veto"
                except SafetyViolation as e:
                    approved, why = False, str(e)
                if not approved:
                    return self._abort(f"dm arm: {why}")
                try:
                    self._client.call("target", q=q.tolist())
                except RuntimeError as e:          # bridge refused (e.g. a latched lag fault on the real arm)
                    return self._abort(f"dm arm: {e}")
                prev = q
                self._q = q.copy()
                self.commands.append(q.copy())
                next_tick += self._dt
                time.sleep(max(0.0, next_tick - time.monotonic()))
        deadline = time.monotonic() + 2.5      # let the servos settle on the last waypoint
        err = float("inf")
        while time.monotonic() < deadline:
            err = float(np.abs(self.get_state().q - pts[-1]).max())
            if err < self._settle_tol:
                break
            time.sleep(self._dt)
        if err > self._reach_tol:               # never report a pose the arm did not reach
            print(f"[dm-arm] the arm stopped {np.degrees(err):.0f} deg short of the commanded pose",
                  file=sys.stderr, flush=True)
            return False
        return True

    # ── gripper ──────────────────────────────────────────────────────────
    def set_gripper(self, pos: float, effort: float = 0.0) -> None:
        """``pos`` in the profile's convention (open_pos .. closed_pos). Blocks until the
        jaws stop moving -- at the target, or stalled on an object."""
        if self._stopped or self.harness.estopped:
            return
        span = self._grip_closed - self._grip_open
        closed = float(np.clip((float(pos) - self._grip_open) / span, 0.0, 1.0)) if abs(span) > 1e-9 else 0.0
        finger = float(np.clip((GAP_MAX_M * (1.0 - closed) + GAP_OFFSET_M) / 2.0, 0.0, FINGER_MAX_M))
        if closed > 0.3 and self.kin is not None:
            # Height of the tool when the jaws close: with the object standing on the table this tells
            # how far it hangs below the tool, which the put-down needs for tall objects.
            T_close = self.kin.fk(np.asarray(self._client.call("state")[0]["q"], float))
            self.last_close_tcp_z = float(T_close[2, 3])
            # ... and how steeply the tool pointed down (pi/2 = straight down). Putting the object
            # down at the same angle sets it down the way it stood.
            fwd = T_close[:3, 0]
            self.last_close_pitch = float(np.arctan2(-fwd[2], np.hypot(fwd[0], fwd[1])))
        self._client.call("gripper", f=finger)
        self._gripper_effort = float(effort)
        deadline = time.monotonic() + 2.0
        still = 0
        time.sleep(0.1)
        while time.monotonic() < deadline and still < 4:
            st = self._client.call("state")[0]
            reached = abs(st["finger"] - finger) < 5e-4
            still = still + 1 if (reached or abs(st["finger_vel"]) < 2e-3) else 0
            time.sleep(0.03)
        self.get_state()                      # refresh self._gripper from the measurement

    # ── soft stop ────────────────────────────────────────────────────────
    def stop(self) -> None:
        super().stop()
        try:
            self._client.call("hold")
        except RuntimeError:
            pass

    def resume(self) -> None:
        try:
            self._client.call("resume")           # clears a latched fault in the bridge (real arm)
        except RuntimeError:
            pass
        super().resume()

    def wait_settled(self, timeout_s=2.0, tol=None) -> bool:
        tol = self._settle_tol if tol is None else float(tol)
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            st = self._client.call("state")[0]
            if float(np.abs(np.asarray(st["dq"])).max()) < 0.05:
                return True
            time.sleep(self._dt)
        return False

    # verification helper (sim only): ground-truth object positions, never used by WRC itself
    def sim_objects(self) -> dict:
        return self._client.call("state")[0].get("objects", {})


class BridgeCamera(CameraBase):
    """RGB-D camera rendered by the DM bridge's physics sim (type: dm_bridge)."""

    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        b = cfg.get("bridge") or {}
        self._client = BridgeClient(str(b.get("host", "127.0.0.1")), bridge_port(b.get("port")))
        self._cam = str(cfg.get("camera", "overhead_rgb"))
        # rgb_only: true drops the rendered depth, to rehearse a plain RGB table camera
        # (WRC then casts pixels onto the table plane, `table_plane_cam` in the profile).
        self._rgb_only = bool(cfg.get("rgb_only", False))
        self._opened = False

    @property
    def has_depth(self) -> bool:
        return not self._rgb_only

    def open(self) -> None:
        self._client.call("info")
        self._opened = True

    def close(self) -> None:
        self._opened = False
        self._client.close()

    def _grab(self) -> Frame:
        if not self._opened:
            raise CameraError("camera not opened")
        try:
            hdr, blobs = self._client.call("frame", cam=self._cam)
        except RuntimeError as e:
            raise CameraError(str(e)) from e
        h, w = int(hdr["h"]), int(hdr["w"])
        bgr = np.frombuffer(blobs[0], dtype=np.uint8).reshape(h, w, 3).copy()
        K = np.asarray(hdr["K"], dtype=np.float64)
        if self._rgb_only:
            return Frame(rgb=bgr, depth_m=None, K=K, depth_source="none")
        depth = np.frombuffer(blobs[1], dtype=np.float32).reshape(h, w).copy()
        return Frame(rgb=bgr, depth_m=depth, K=K, depth_source="sensor")


_registered = False


def register() -> None:
    """Teach WRC's composition root (apps/demo.py) the two DM bridge backends."""
    global _registered
    if _registered:
        return
    import wrc_demo.apps.demo as demo

    orig_make_arm, orig_make_camera = demo.make_arm, demo.make_camera

    def make_arm(cfg, kinematics=None):
        if str(cfg.arm.get("type", "mock")) == "rebot_dm_bridge":
            arm = DmBridgeArm(cfg, kinematics)
            arm.connect()
            arm.grasp_bridge = None
            return arm
        return orig_make_arm(cfg, kinematics=kinematics)

    def make_camera(ccfg):
        if str(ccfg.get("type", "")) == "dm_bridge":
            return BridgeCamera(ccfg)
        return orig_make_camera(ccfg)

    demo.make_arm, demo.make_camera = make_arm, make_camera

    # A parallel gripper turned half a turn about its own axis is the same grasp (WRC says so
    # itself in camera_grasp.py). WRC plans only one of the two; with the real arm's wrist roll
    # limited to +-90 deg, that one is often out of range while its twin is not. Offer both.
    import wrc_demo.skills.runtime as rt

    orig_select = rt.select_grasp
    half_turn = np.diag([1.0, -1.0, -1.0])

    def select_grasp_with_twins(grasps, kin, q_current, **kw):
        both = []
        for g in grasps:
            both.append(g)
            both.append(dataclasses.replace(g, rotation=np.asarray(g.rotation, dtype=float) @ half_turn))
        return orig_select(both, kin, q_current, **kw)

    rt.select_grasp = select_grasp_with_twins
    _registered = True
