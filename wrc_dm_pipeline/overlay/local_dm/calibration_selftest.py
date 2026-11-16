#!/usr/bin/env python
"""Self-test of calibrate_camera.py against the DM physics sim, where the true camera
pose is known. Lays the printed sheet on the sim table (rotated, off-centre), calibrates
from the rendered image and compares with the truth.  Run:  ./START_WRC.sh dm-calib-test

This checks the maths and the frame conventions. It cannot check your ruler work or
your real camera's lens.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
PORT = int(os.environ.get("WRC_DM_BRIDGE_PORT") or 9881)   # set it to use a private simulator
POSE = (0.39, 0.04, 35.0)          # sheet centre x, y [m], yaw [deg]
TABLE_Z = 0.03


def port_open() -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", PORT)) == 0


def main() -> int:
    import calibrate_camera as cc
    from wrc_dm_plugin import BridgeClient

    if port_open():
        print(f"A DM bridge is already running on port {PORT} (it has no calibration sheet). Stop it, or use "
              "a private port:  WRC_DM_BRIDGE_PORT=9891 ./START_WRC.sh dm-calib-test")
        return 2
    teleop = Path(os.environ.get("STEEL_TEAM_TELEOP_DIR", REPO.parent / "steel_team" / "teleoperation"))
    py = teleop / ".venv" / "bin" / "python"
    if not py.exists():
        print(f"teleoperation venv not found at {py}")
        return 2
    tmp = Path(tempfile.mkdtemp(prefix="wrc_calib_"))
    sheet = tmp / "calib_sheet.png"
    cc.make_sheet(sheet)
    srv = subprocess.Popen([str(py), str(HERE / "dm_bridge_server.py"), "--headless", "--scene", "demo", "--calib-sheet", str(sheet),
                            "--sheet-pose", ",".join(str(v) for v in POSE), "--port", str(PORT)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 40
        while not port_open():
            if time.time() > deadline or srv.poll() is not None:
                print("the sim bridge did not start")
                return 2
            time.sleep(0.3)
        yaw = np.radians(POSE[2])
        up = np.array([-np.sin(yaw), np.cos(yaw)])
        centre = np.array(POSE[:2])
        cfg = {"table_z": TABLE_Z, "sheet": {"A": (centre + 0.1335 * up).tolist(), "B": (centre - 0.1335 * up).tolist()}}

        bgr, K, _ = cc.grab_from_profile("dm_sim_overhead")
        found = cc.detect(bgr)
        res = cc.solve(found, cfg, K)
        est = res["T"]
        true = np.array(BridgeClient(port=PORT).call("frame", cam="overhead_rgb")[0]["T_base_cam"])

        dR = est[:3, :3].T @ true[:3, :3]
        ang = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
        Kinv = np.linalg.inv(K)
        h, w = bgr.shape[:2]
        errs = []
        for u in np.linspace(0.15 * w, 0.85 * w, 5):
            for v in np.linspace(0.15 * h, 0.85 * h, 4):
                ray = Kinv @ np.array([u, v, 1.0])
                pts = []
                for T in (true, est):
                    r = T[:3, :3] @ ray
                    pts.append(T[:3, 3] + (TABLE_Z - T[2, 3]) / r[2] * r)
                errs.append(1000 * float(np.linalg.norm(pts[0] - pts[1])))
        print(f"markers detected: {sorted(found)} of [0..5]; fit {res['rmse_px']:.2f} px RMSE")
        print(f"camera pose vs truth: {1000 * np.linalg.norm(est[:3, 3] - true[:3, 3]):.1f} mm, {ang:.2f} deg")
        print(f"table points across the image vs truth: mean {np.mean(errs):.1f} mm, max {np.max(errs):.1f} mm")
        ok = len(found) == 6 and max(errs) < 5.0
        print("PASS" if ok else "FAIL (expected all 6 markers and < 5 mm on the table)")
        return 0 if ok else 1
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=5)
        except subprocess.TimeoutExpired:
            srv.kill()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
