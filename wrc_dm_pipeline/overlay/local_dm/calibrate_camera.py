#!/usr/bin/env python
"""Camera-to-base calibration for a fixed table camera, from a printed marker sheet.

WRC needs to know where the camera is relative to the robot base (``extrinsics.T``).
WRC's own calibration tools drive the RS arm's SDK, so this is the DM-friendly way:
no arm motion, only a printed sheet and a ruler.

  1. python local_dm/calibrate_camera.py --make-sheet calib_sheet.png
     Print it on A4 at 100 % and lay it flat on the table, printed side up, inside
     the camera view (anywhere, any rotation).
  2. Measure where the two crosses "A" and "B" are in the ROBOT BASE FRAME, in metres:
       origin = joint-1 rotation axis at the base mounting plane
       +x = straight ahead of the arm, +y = to the arm's left, +z = up
     and write them into a small YAML file (see --example-config).
  3. python local_dm/calibrate_camera.py --camera <profile> --config my_calib.yaml \\
            --write-profile my_camera
     -> writes configs/cameras/my_camera.yaml (a copy of <profile> with the measured
        extrinsics and table plane). Use it with:  run_dm.py --camera my_camera

The result is only as good as the two cross measurements: 5 mm off there is 5 mm off
in every grasp. The printed scale does not matter (it is derived from the A-B distance).

Offline use on a saved image: --image shot.png --fov-deg 70   (or --intrinsics fx,fy,cx,cy)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# ── the printed sheet (A4 portrait, millimetres, origin top-left, u right, v down) ──
SHEET_W_MM, SHEET_H_MM = 210.0, 297.0
MARKER_MM = 60.0
SHEET_MARKERS = {0: (55.0, 58.5), 1: (155.0, 58.5), 2: (55.0, 148.5),
                 3: (155.0, 148.5), 4: (55.0, 238.5), 5: (155.0, 238.5)}   # id -> centre
CROSS_A, CROSS_B = (105.0, 15.0), (105.0, 282.0)
AB_MM = CROSS_B[1] - CROSS_A[1]
DICT = cv2.aruco.DICT_4X4_50

EXAMPLE_CONFIG = """\
# Camera calibration input for local_dm/calibrate_camera.py. All positions are in the
# ROBOT BASE FRAME, in metres: origin = joint-1 axis at the base mounting plane,
# +x straight ahead of the arm, +y to the arm's left, +z up.

table_z: 0.0          # height of the table surface (where the sheet lies) above the base plane

sheet:                # the printed A4 sheet: where its two crosses are
  A: [0.400, 0.134]   # cross "A" (top of the page)
  B: [0.400, -0.133]  # cross "B" (bottom of the page)

# Optional extra markers (same dictionary, DICT_4X4_50, ids 6..49) placed anywhere on the
# table; give the position of each marker's CENTRE. A wider spread improves accuracy.
# markers:
#   6: [0.20, -0.20]
#   7: [0.20, 0.20]
"""


def make_sheet(path: Path, dpi: int = 300) -> None:
    px = lambda mm: int(round(mm / 25.4 * dpi))  # noqa: E731
    img = np.full((px(SHEET_H_MM), px(SHEET_W_MM)), 255, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(DICT)
    side = px(MARKER_MM)
    for mid, (u, v) in SHEET_MARKERS.items():
        m = cv2.aruco.generateImageMarker(d, mid, side)
        x0, y0 = px(u) - side // 2, px(v) - side // 2
        img[y0:y0 + side, x0:x0 + side] = m
        cv2.putText(img, f"id {mid}", (x0, y0 + side + px(5)), cv2.FONT_HERSHEY_SIMPLEX,
                    dpi / 300 * 0.9, 0, max(1, dpi // 150), cv2.LINE_AA)
    for name, (u, v) in (("A", CROSS_A), ("B", CROSS_B)):
        c, arm, t = (px(u), px(v)), px(6), max(1, px(0.35))
        cv2.line(img, (c[0] - arm, c[1]), (c[0] + arm, c[1]), 0, t)
        cv2.line(img, (c[0], c[1] - arm), (c[0], c[1] + arm), 0, t)
        cv2.putText(img, name, (c[0] + px(8), c[1] + px(3)), cv2.FONT_HERSHEY_SIMPLEX,
                    dpi / 300 * 1.6, 0, max(2, dpi // 100), cv2.LINE_AA)
    cv2.putText(img, "WRC camera calibration sheet - print at 100 % - measure crosses A and B",
                (px(40), px(292)), cv2.FONT_HERSHEY_SIMPLEX, dpi / 300 * 0.7, 0, max(1, dpi // 150), cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise SystemExit(f"could not write {path}")
    print(f"wrote {path}  ({img.shape[1]}x{img.shape[0]} px = A4 at {dpi} dpi). "
          f"Crosses A and B are {AB_MM:.0f} mm apart when printed at 100 %.")


# ── geometry ────────────────────────────────────────────────────────────────

def sheet_points(A, B) -> tuple[dict[int, np.ndarray], float]:
    """Base-frame xy of the 4 corners (TL, TR, BR, BL) of every sheet marker, from the two
    measured crosses. Returns ({id: (4, 2)}, print_scale)."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    L = float(np.linalg.norm(B - A))
    if L < 0.05:
        raise SystemExit("sheet.A and sheet.B are (almost) the same point")
    scale = L / (AB_MM / 1000.0)
    t_axis = -(B - A) / L                          # page "up" in the base frame
    s_axis = np.array([t_axis[1], -t_axis[0]])     # page "right" (printed side up, seen from above)

    def to_base(u, v):
        return A + scale * ((u - CROSS_A[0]) / 1000.0 * s_axis - (v - CROSS_A[1]) / 1000.0 * t_axis)

    h = MARKER_MM / 2
    out = {mid: np.array([to_base(u - h, v - h), to_base(u + h, v - h),
                          to_base(u + h, v + h), to_base(u - h, v + h)])
           for mid, (u, v) in SHEET_MARKERS.items()}
    return out, scale


def marker_centre_px(c: np.ndarray) -> np.ndarray:
    """Projective centre of a detected marker = intersection of its diagonals."""
    p1, p2, p3, p4 = c[0], c[2], c[1], c[3]
    d1, d2 = p2 - p1, p4 - p3
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-9:
        return c.mean(axis=0)
    s = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / den
    return p1 + s * d1


def detect(bgr: np.ndarray):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, _ = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT), params).detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).astype(np.float64) for i, c in zip(ids.flatten(), corners)}


def solve(found: dict, cfg: dict, K: np.ndarray) -> dict:
    """PnP on the table plane. Returns T_base_cam (OpenCV camera frame) and quality numbers."""
    table_z = float(cfg.get("table_z", 0.0))
    obj, img, labels = [], [], []
    scale = None
    if cfg.get("sheet"):
        pts, scale = sheet_points(cfg["sheet"]["A"], cfg["sheet"]["B"])
        for mid, xy in pts.items():
            if mid in found:
                for k in range(4):
                    obj.append(xy[k]); img.append(found[mid][k]); labels.append(f"sheet id {mid} corner {k}")
    for mid, xy in (cfg.get("markers") or {}).items():
        if int(mid) in found:
            obj.append(np.asarray(xy, float)); img.append(marker_centre_px(found[int(mid)]))
            labels.append(f"marker id {mid} centre")
    if len(obj) < 4:
        raise SystemExit(f"only {len(obj)} usable points (need >= 4). Detected marker ids: "
                         f"{sorted(found)}. Is the sheet in view, flat, well lit and in focus?")
    obj = np.asarray(obj, float)
    img = np.asarray(img, float)
    centre = obj.mean(axis=0)                       # table frame: origin at the pattern centre, z = 0
    obj3 = np.column_stack([obj - centre, np.zeros(len(obj))])
    ok, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj3, img, K, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise SystemExit("solvePnP failed")
    cands = []
    for rvec, tvec, err in zip(rvecs, tvecs, np.asarray(errs).flatten()):
        rvec, tvec = cv2.solvePnPRefineLM(obj3, img, K, None, rvec, tvec)
        R, _ = cv2.Rodrigues(rvec)
        cam_in_table = (-R.T @ tvec).flatten()
        proj, _ = cv2.projectPoints(obj3, rvec, tvec, K, None)
        rmse = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
        if cam_in_table[2] > 0:                     # the camera must be above the table
            cands.append((rmse, R, tvec.flatten(), cam_in_table))
    if not cands:
        raise SystemExit("no solution with the camera above the table: check the A/B measurements")
    cands.sort(key=lambda c: c[0])
    rmse, R, t, cam_in_table = cands[0]
    ambiguous = len(cands) > 1 and cands[1][0] < 1.5 * max(rmse, 0.2)

    T = np.eye(4)                                    # camera (OpenCV) -> base
    T[:3, :3] = R.T
    T[:3, 3] = cam_in_table + np.array([centre[0], centre[1], table_z])
    n_cam = R @ np.array([0.0, 0.0, 1.0])            # table normal (base +z) seen from the camera
    plane = [float(n_cam[0]), float(n_cam[1]), float(n_cam[2]), float(cam_in_table[2])]

    # metric check: cast each image point back onto the table plane and compare with the input
    Kinv = np.linalg.inv(K)
    errs_mm = []
    for p_img, p_obj in zip(img, obj):
        ray = T[:3, :3] @ (Kinv @ np.array([p_img[0], p_img[1], 1.0]))
        s = (table_z - T[2, 3]) / ray[2]
        errs_mm.append(1000 * float(np.linalg.norm((T[:3, 3] + s * ray)[:2] - p_obj)))
    return {"T": T, "table_plane_cam": plane, "rmse_px": rmse, "n_points": len(obj), "labels": labels,
            "err_mm_mean": float(np.mean(errs_mm)), "err_mm_max": float(np.max(errs_mm)),
            "tilt_deg": float(np.degrees(np.arccos(np.clip(-T[2, 2], -1, 1)))),
            "print_scale": scale, "ambiguous": ambiguous, "img": img, "table_z": table_z}


# ── frame sources ───────────────────────────────────────────────────────────

def grab_from_profile(profile: str):
    sys.path.insert(0, str(HERE))
    from wrc_demo.config import load_profile
    ccfg = load_profile("cameras", profile)
    if str(ccfg.get("type", "")) == "dm_bridge":
        from wrc_dm_plugin import BridgeCamera
        cam = BridgeCamera(ccfg)
    else:
        from wrc_demo.perception.camera_base import make_camera
        cam = make_camera(ccfg)
    cam.open()
    try:
        cam.warm_up(int(ccfg.get("warmup_frames", 5)))
        frame = cam.get_frame()
    finally:
        cam.close()
    approx = str(ccfg.get("type", "")) == "uvc" and ccfg.get("intrinsics") is None
    return frame.rgb, np.asarray(frame.K, float), approx


def write_profile(src: str, dst: str, res: dict, force: bool) -> Path:
    src_path = REPO / "configs" / "cameras" / f"{src}.yaml"
    dst_path = REPO / "configs" / "cameras" / f"{dst}.yaml"
    if dst_path.exists() and not force:
        raise SystemExit(f"{dst_path} exists (use --force to overwrite)")
    prof = yaml.safe_load(src_path.read_text()) or {}
    prof["extrinsics"] = {"mode": "eye_to_hand", "T": [[round(float(v), 6) for v in row] for row in res["T"]]}
    prof["table_plane_cam"] = [round(v, 6) for v in res["table_plane_cam"]]
    prof["requires_calibration"] = False   # the inline T above is the calibration
    prof["fuse_beliefs"] = True
    header = (f"# Generated by local_dm/calibrate_camera.py on {time.strftime('%Y-%m-%d %H:%M')} from "
              f"profile '{src}'.\n# Marker sheet fit: {res['n_points']} points, {res['rmse_px']:.2f} px RMSE, "
              f"{res['err_mm_mean']:.1f} mm mean / {res['err_mm_max']:.1f} mm max on the table.\n"
              "# Re-run the calibration whenever the camera, the table or the robot base moves.\n")
    dst_path.write_text(header + yaml.safe_dump(prof, sort_keys=False, default_flow_style=None))
    return dst_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--make-sheet", metavar="PNG", help="write the printable A4 marker sheet and exit")
    p.add_argument("--example-config", action="store_true", help="print an example --config file and exit")
    p.add_argument("--camera", help="WRC camera profile to grab a frame from (configs/cameras/<name>.yaml)")
    p.add_argument("--image", help="calibrate from a saved image instead of a live camera")
    p.add_argument("--intrinsics", help="with --image: fx,fy,cx,cy in pixels")
    p.add_argument("--fov-deg", type=float, help="with --image: horizontal field of view (approximate intrinsics)")
    p.add_argument("--config", help="YAML with table_z and the measured sheet crosses / marker centres")
    p.add_argument("--write-profile", metavar="NAME", help="write configs/cameras/NAME.yaml with the result")
    p.add_argument("--force", action="store_true", help="overwrite an existing --write-profile target")
    p.add_argument("--out-dir", default=str(HERE / "calibration"), help="where to save the result + debug image")
    args = p.parse_args(argv)

    if args.example_config:
        print(EXAMPLE_CONFIG, end="")
        return 0
    if args.make_sheet:
        make_sheet(Path(args.make_sheet))
        return 0
    if not args.config or not (args.camera or args.image):
        p.error("need --config and one of --camera / --image  (or --make-sheet / --example-config)")
    if not Path(args.config).is_file():
        raise SystemExit(f"config file not found: {args.config}\n"
                         "Create it first:  calibrate_camera.py --example-config > my_calib.yaml\n"
                         "then put in the measured positions of crosses A and B and the table height.")
    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    if not cfg.get("sheet") and not cfg.get("markers"):
        raise SystemExit("the config needs a 'sheet:' block (crosses A and B) and/or a 'markers:' block")

    approx_K = False
    if args.image:
        bgr = cv2.imread(args.image)
        if bgr is None:
            raise SystemExit(f"cannot read {args.image}")
        h, w = bgr.shape[:2]
        if args.intrinsics:
            fx, fy, cx, cy = (float(v) for v in args.intrinsics.split(","))
        elif args.fov_deg:
            fx = fy = w / (2 * np.tan(np.radians(args.fov_deg) / 2)); cx, cy = w / 2, h / 2
            approx_K = True
        else:
            p.error("--image needs --intrinsics fx,fy,cx,cy or --fov-deg")
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
        name = Path(args.image).stem
    else:
        if not (REPO / "configs" / "cameras" / f"{args.camera}.yaml").is_file():
            names = sorted(f.stem for f in (REPO / "configs" / "cameras").glob("*.yaml"))
            raise SystemExit(f"no camera profile named {args.camera!r}. Available: {', '.join(names)}")
        try:
            bgr, K, approx_K = grab_from_profile(args.camera)
        except Exception as e:  # noqa: BLE001 -- camera backends raise their own error types
            hint = ("Is the camera connected and free (no other app using it)? "
                    "For the simulator camera, start the DM bridge first.")
            if sys.platform == "darwin":
                hint += ("\nOn macOS, 'not authorized to capture video' means the app this terminal runs in "
                         "may not use the camera yet: allow it in System Settings > Privacy & Security > "
                         "Camera (or accept the prompt macOS shows), then run this again.")
            raise SystemExit(f"could not get a frame from camera profile {args.camera!r}: {e}\n{hint}") from None
        name = args.camera

    found = detect(bgr)
    print(f"image {bgr.shape[1]}x{bgr.shape[0]}, fx={K[0, 0]:.1f} fy={K[1, 1]:.1f}; "
          f"detected marker ids: {sorted(found) or 'none'}")
    res = solve(found, cfg, K)
    T = res["T"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dbg = bgr.copy()
    cv2.aruco.drawDetectedMarkers(dbg, [c.reshape(1, 4, 2).astype(np.float32) for c in found.values()],
                                  np.array(list(found)).reshape(-1, 1))
    Rcb, tcb = T[:3, :3].T, -T[:3, :3].T @ T[:3, 3]
    for label, pt in (("base origin", [0, 0, res["table_z"]]), ("+x 10cm", [0.1, 0, res["table_z"]]),
                      ("+y 10cm", [0, 0.1, res["table_z"]])):
        q = K @ (Rcb @ np.array(pt, float) + tcb)
        if q[2] > 0:
            uv = (int(q[0] / q[2]), int(q[1] / q[2]))
            cv2.circle(dbg, uv, 5, (0, 0, 255), -1)
            cv2.putText(dbg, label, (uv[0] + 6, uv[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / f"{name}_debug.jpg"), dbg)
    result = {"T_base_cam": [[round(float(v), 6) for v in row] for row in T],
              "table_plane_cam": [round(v, 6) for v in res["table_plane_cam"]],
              "K": [[round(float(v), 3) for v in row] for row in K], "image_size": [bgr.shape[1], bgr.shape[0]],
              "rmse_px": round(res["rmse_px"], 3), "table_error_mm_mean": round(res["err_mm_mean"], 2),
              "table_error_mm_max": round(res["err_mm_max"], 2), "n_points": res["n_points"],
              "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out_dir / f"{name}.yaml").write_text(yaml.safe_dump(result, sort_keys=False, default_flow_style=None))

    print(f"\ncamera position in the base frame: x={T[0, 3]:.3f}  y={T[1, 3]:.3f}  z={T[2, 3]:.3f} m "
          f"({res['table_plane_cam'][3]:.3f} m above the table), tilted {res['tilt_deg']:.1f} deg from straight down")
    print(f"fit: {res['n_points']} points, {res['rmse_px']:.2f} px RMSE; on the table that is "
          f"{res['err_mm_mean']:.1f} mm mean, {res['err_mm_max']:.1f} mm max")
    if res["print_scale"] is not None and abs(res["print_scale"] - 1) > 0.02:
        print(f"NOTE: the crosses are {res['print_scale'] * AB_MM:.0f} mm apart by your measurement, "
              f"{AB_MM:.0f} mm at 100 % print. The measured value is used; double-check it if unexpected.")
    if res["err_mm_max"] > 5:
        print("WARNING: more than 5 mm error on the table. Check the A/B measurements, that the sheet is flat, "
              "and the intrinsics (lens distortion is not modelled).")
    if res["ambiguous"]:
        print("NOTE: a second camera pose fits almost as well (small flat target seen nearly head-on). "
              "Positions ON the table are still right; add spread-out 'markers:' for tall objects.")
    if approx_K:
        print("NOTE: the intrinsics come from a field-of-view guess. Objects on the table are still located "
              "correctly (the fit absorbs it), tall objects less so. Put real 'intrinsics:' in the profile if you can.")
    print("\nextrinsics:\n  mode: eye_to_hand\n  T:")
    for row in T:
        print("    - [" + ", ".join(f"{v: .6f}" for v in row) + "]")
    print("table_plane_cam: [" + ", ".join(f"{v:.6f}" for v in res["table_plane_cam"]) + "]")
    print(f"\nsaved {out_dir / (name + '.yaml')} and {out_dir / (name + '_debug.jpg')} "
          "(red dots = where the fit puts the base origin and the +x / +y directions: check they look right)")
    if args.write_profile:
        if not args.camera:
            raise SystemExit("--write-profile needs --camera (the profile to copy)")
        path = write_profile(args.camera, args.write_profile, res, args.force)
        print(f"wrote {path}\nuse it with:  python local_dm/run_dm.py --camera {args.write_profile} ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
