"""Streams WRC's live (mock) arm state to the MuJoCo 3D mirror viewer.

Added by the hackathon setup; NOT upstream. Read-only with respect to the
robot: it only calls arm.get_state() and reads the belief store, so it cannot
command motion. Sends small JSON datagrams over UDP to localhost.

Packet: {"t", "q": [6 rad, asset frame], "grip": 0=open..1=closed,
         "holding": label|None, "task": str|None, "beliefs": [...] (every 0.5 s),
         "drop_zone": [x, y]}
"""
from __future__ import annotations

import json
import socket
import threading
import time


class MirrorPublisher:
    def __init__(self, runtime, arm, cfg, host: str = "127.0.0.1", port: int = 9870,
                 rate_hz: float = 50.0):
        self.runtime, self.arm = runtime, arm
        self.addr = (host, port)
        self.period = 1.0 / rate_hz
        self.drop_zone = list(cfg.grasp.get("drop_zone", [0.18, -0.17]))
        g = cfg.arm.get("gripper") or {}
        self.g_open = float(g.get("open_pos", 0.0))
        self.g_closed = float(g.get("closed_pos", 1.0))
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="mirror-pub")

    def start(self) -> "MirrorPublisher":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _beliefs(self):
        try:
            return [{"label": b["label"], "position": [float(v) for v in b["position"]],
                     "state": b.get("state")} for b in self.runtime.beliefs.summary()]
        except Exception:
            return []

    def _loop(self) -> None:
        last_beliefs = 0.0
        span = (self.g_closed - self.g_open) or 1.0
        while not self._stop.is_set():
            try:
                st = self.arm.get_state()
                msg = {
                    "t": time.time(),
                    "q": [float(v) for v in st.q[:6]],
                    "grip": float((st.gripper_pos - self.g_open) / span),
                    "holding": getattr(self.runtime, "held_object", None),
                    "task": getattr(self.runtime, "current_task", None),
                    "drop_zone": self.drop_zone,
                }
                now = time.monotonic()
                if now - last_beliefs > 0.5:
                    msg["beliefs"] = self._beliefs()
                    last_beliefs = now
                self._sock.sendto(json.dumps(msg).encode(), self.addr)
            except Exception:
                pass  # viewer not running / transient: never disturb the robot loop
            time.sleep(self.period)
