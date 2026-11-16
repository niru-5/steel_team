#!/usr/bin/env python
"""Check what the cameras measure, WITHOUT moving the arm (no arm server needed for real cameras).

Takes one picture from each scene camera, finds the objects, and prints where each one stands,
how tall and how wide it is -- measured from the camera outlines, with no depth sensor
(local_dm/two_view.py). Put an object at a spot you measured with a ruler and compare.

  python local_dm/measure_objects.py --camera my_camera --camera2 my_camera2 --table-z 0.0 \\
         --classes "cup,red cup,white box" [--known-objects my_objects.yaml]

  # in the simulator (arm server running), with the depth thrown away; prints the true places too:
  python local_dm/measure_objects.py --camera dm_sim_overhead --camera2 dm_sim_overhead --rgb-only

Pictures with the outlines drawn in are saved under runs/measure_<time>/.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_dm  # noqa: E402  (also switches to the repo folder, which the detector needs)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", default="dm_sim_overhead", help="main scene camera (profile in configs/cameras/)")
    p.add_argument("--camera2", default=None, help="second scene camera (leave out to see what one camera gives)")
    p.add_argument("--arm", default="rebot_dm")
    p.add_argument("--classes", default=None, help="object names for the detector, comma separated")
    p.add_argument("--table-z", type=float, default=None, help="table height above the arm base plane [m]")
    p.add_argument("--sim-camera", default=None)
    p.add_argument("--sim-camera2", default=None)
    p.add_argument("--rgb-only", action="store_true", help="simulator only: throw the rendered depth away")
    p.add_argument("--known-objects", default=None, metavar="YAML")
    p.add_argument("--out", default=None, help="folder for the annotated pictures")
    args = p.parse_args(argv)
    args.port, args.set = 8090, None

    import cv2
    import numpy as np

    import two_view as tv
    import wrc_dm_plugin
    wrc_dm_plugin.register()
    import wrc_demo.apps.demo as demo
    from wrc_demo.perception.colors import detection_color
    from wrc_demo.perception.grounding import Extrinsics

    out = Path(args.out) if args.out else run_dm.REPO / "runs" / ("measure_" + time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    try:
        scene, sim_cams = run_dm.bridge_scene()
    except OSError:
        scene, sim_cams = {}, {}
    cfg = run_dm.build_cfg(args, out, scene, sim_cams)
    table_z = float(cfg.safety.get("table_z", 0.0))
    classes = list(cfg._data["detect_classes"])
    known = tv.load_known(args.known_objects) if args.known_objects else []
    print(f"table height {table_z:+.3f} m, looking for: {', '.join(classes)}"
          + (f", known sizes: {', '.join(k.name for k in known)}" if known else ""))

    detector = demo._make_detector(cfg)
    shots = []                                           # per camera: name, frame, T, detections
    for i, ccfg in enumerate(demo._camera_cfgs(cfg)):
        name = str(ccfg.get("name", f"camera{i}"))
        if "extrinsics" not in ccfg:
            print(f"ERROR: camera '{name}' has no calibration (extrinsics) in its profile -- "
                  "run ./START_WRC.sh dm-calibrate for it first", file=sys.stderr)
            return 2
        cam = demo.make_camera(ccfg)
        cam.open()
        try:
            cam.warm_up(int(ccfg.get("warmup_frames", 5)))
            frame = cam.get_frame()
        finally:
            cam.close()
        T = np.asarray(Extrinsics.from_config(ccfg.get("extrinsics")).cam_to_base(), dtype=float)
        dets = [d for d in detector.detect(frame, classes=classes) if d.mask is not None]
        height_above = float(T[2, 3]) - table_z
        print(f"camera '{name}': {frame.rgb.shape[1]}x{frame.rgb.shape[0]}, {height_above * 100:.0f} cm above the "
              f"table, sees: {', '.join(d.label for d in dets) or 'nothing'}")
        shots.append((name, frame, T, dets))

    def views(shot):
        name, frame, T, dets = shot
        return [tv.View(name, frame.K, T, d.mask, d.label, float(d.conf), detection_color(frame.rgb, d)) for d in dets]

    main_views = views(shots[0])
    other_views = [v for s in shots[1:] for v in views(s)]
    truth = {}
    if scene.get("object_names"):                        # simulator: the true places, for comparison only
        try:
            from wrc_dm_plugin import BridgeClient
            c = BridgeClient()
            truth = c.call("state")[0].get("objects", {})
            c.close()
        except (RuntimeError, OSError):
            pass

    print()
    results = []
    for v in main_views:
        m = tv.measure(v, other_views, table_z, known=known)
        if m is None:
            why = {"small": "too small in the picture", "top view": "seen from almost straight above",
                   "cut off": "cut off at the picture edge"}[tv.why_unusable(v)]
            print(f"- {v.label}: could not be measured ({why})")
            continue
        results.append((v, m))
        print(f"- {v.label}: {m.describe()}")
        for pv in m.views:
            h = f"{pv['height'] * 100:.1f} cm tall" if pv["height"] is not None else "height not visible"
            wd = f"{pv['width'] * 100:.1f} cm wide" if pv["width"] else "width cut off by the picture edge"
            print(f"      camera '{pv['camera']}': {wd}, {h}, {pv['range'] * 100:.0f} cm away")
        if truth:
            near = min(truth.items(), key=lambda kv: (kv[1][0] - m.centre[0]) ** 2 + (kv[1][1] - m.centre[1]) ** 2)
            d = ((near[1][0] - m.centre[0]) ** 2 + (near[1][1] - m.centre[1]) ** 2) ** 0.5
            print(f"      simulator truth: nearest object '{near[0]}' at ({near[1][0]:.3f}, {near[1][1]:.3f}) "
                  f"-> {d * 1000:.0f} mm from the measured place")
    if not main_views:
        print("The main camera sees none of the objects. Check --classes and the lighting.")

    # annotated pictures: every outline, and for measured objects the result
    for name, frame, T, dets in shots:
        img = frame.rgb.copy()
        for d in dets:
            cs, _ = cv2.findContours(d.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cs, -1, (0, 255, 0), 2)
            x0, y0 = int(d.bbox[0]), int(d.bbox[1])
            cv2.putText(img, d.label, (x0, max(y0 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        Tinv = np.linalg.inv(T)
        for v, m in results:                             # the measured middle, top and foot, drawn in every camera
            for z, col in ((table_z, (0, 0, 255)), (table_z + m.height, (255, 0, 0))):
                pc = Tinv[:3, :3] @ np.array([m.centre[0], m.centre[1], z]) + Tinv[:3, 3]
                if pc[2] > 0.05:
                    u = int(round(frame.K[0, 0] * pc[0] / pc[2] + frame.K[0, 2]))
                    w = int(round(frame.K[1, 1] * pc[1] / pc[2] + frame.K[1, 2]))
                    cv2.circle(img, (u, w), 4, col, -1)
        path = out / f"{name}.jpg"
        cv2.imwrite(str(path), img)
        print(f"saved {path}  (green = outlines; red dot = measured foot centre, blue dot = measured top centre)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
