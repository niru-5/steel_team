#!/usr/bin/env python
"""Check that the camera-outline measuring is wired into the pipeline correctly (no simulator).

Uses WRC's real data types and the real functions from run_dm.py, with a stand-in robot whose
two cameras "see" a drawn can and cup (the drawing code of two_view_selftest.py). It checks:

  1. the old behaviour: with a depth-less camera, WRC's flat-table guess puts a standing object
     in the wrong place and at the wrong height (this is the fault seen on the real robot);
  2. the new behaviour: the planned grasp is over the object's true middle, at half its height;
  3. the configuration: a second camera is added without being fused twice, and the
     simulator's cameras can be switched to RGB-only with the right table plane.

  python local_dm/camera_measure_wiring_test.py        # exit code 1 on failure

Not covered: the real detector, the real cameras, the simulator, arm motion.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_dm  # noqa: E402
import two_view_selftest as st  # noqa: E402

from wrc_demo.config import Cfg  # noqa: E402
from wrc_demo.perception.depth_provider import DepthProvider  # noqa: E402
from wrc_demo.perception.grounding import mask_to_points_cam, oriented_bbox  # noqa: E402
from wrc_demo.types import Detection, Frame, ObjectFix, transform_points  # noqa: E402

TABLE_Z = st.TABLE_Z
failures = []


def check(ok: bool, what: str) -> None:
    print(f"  {'ok    ' if ok else 'FAILED'} {what}")
    if not ok:
        failures.append(what)


def bbox_of(mask):
    vs, us = np.nonzero(mask)
    return np.array([us.min(), vs.min(), us.max(), vs.max()], dtype=np.float32)


def fake_robot(objects: dict, target: str, Ta, Tb, K, second_camera=True):
    """A stand-in for WRC's runtime: two cameras that see `objects`, a detector that outlines them."""
    def frame_of(T):
        f = Frame(rgb=np.zeros((st.H, st.W, 3), np.uint8), depth_m=None, K=K, depth_source="none")
        f._dets = [Detection(label=name, conf=0.9, bbox=bbox_of(m), mask=m)
                   for name, pts in objects.items() for m in [st.draw(pts, K, T)]]
        return f

    plane = run_dm.table_plane_in_camera(Ta, TABLE_Z)
    main_frame = DepthProvider(Cfg({"table_plane_cam": plane})).ensure_depth(frame_of(Ta))
    notes = []

    def localize(query, spatial_hint=None):
        det = next(d for d in main_frame._dets if d.label == query)
        pts = transform_points(Ta, mask_to_points_cam(main_frame, det.mask))     # WRC's flat-table guess
        centre, extent, axes = oriented_bbox(pts)
        return main_frame, ObjectFix(label=query, position=centre, points=pts, detection=det, extent=extent, axes=axes)

    cam = lambda T, name: SimpleNamespace(                                        # noqa: E731
        stream=SimpleNamespace(name=name, get_frame=lambda: frame_of(T)),
        extrinsics=SimpleNamespace(cam_to_base=lambda: T))
    rt = SimpleNamespace(
        _localize=localize, last_frame=main_frame,
        watcher=SimpleNamespace(_cams=[cam(Ta, "main")] + ([cam(Tb, "second")] if second_camera else [])),
        detector=SimpleNamespace(detect=lambda f, classes=None: f._dets),
        _default_classes=["cup"], extrinsics=SimpleNamespace(cam_to_base=lambda: Ta),
        memory=SimpleNamespace(add=lambda kind, text, *a, **k: notes.append(text)),
        _plan_grasps=lambda fix, label=None: "WRC planner", _max_width=0.091)
    return rt, notes


def main() -> int:
    K = st.intrinsics()
    rig = st.RIGS["mounted + table level (one high, one 12 cm above the table)"]
    Ta, Tb = st.look_at(*rig[0]), st.look_at(*rig[1])
    can_pts, can = st.solid("can", 0.24, -0.08)
    cup_pts, cup = st.solid("cup", 0.40, 0.15)
    objects = {"red cup": can_pts, "cup": cup_pts}
    cfg = SimpleNamespace(safety={"table_z": TABLE_Z})

    print("1. old behaviour (camera without depth, WRC's flat-table guess):")
    rt, _ = fake_robot(objects, "red cup", Ta, Tb, K)
    run_dm.install_topdown_grasps(rt, cfg, force=True)
    _, fix = rt._localize("red cup")
    old = rt._plan_grasps(fix, "red cup")
    if isinstance(old, str):
        top = float(np.percentile(fix.points[:, 2], 98)) - TABLE_Z
        print(f"     the can looks {top * 100:.1f} cm tall (true: 13.0) -> treated as a flat object, "
              f"its 'middle' {np.linalg.norm(fix.position[:2] - can['centre']) * 100:.1f} cm from the true one")
        check(top < 0.04, "the flat-table guess makes the 13 cm can look flat (the fault on the real robot)")
    else:
        off = np.linalg.norm(old[0].position[:2] - can["centre"])
        print(f"     grasp planned {off * 100:.1f} cm from the can's true middle")
        check(off > 0.03, "the flat-table guess misplaces the can")

    print("2. new behaviour (measured from the two camera outlines):")
    for name, truth, second in (("red cup", can, True), ("cup", cup, True), ("red cup", can, False)):
        rt, notes = fake_robot(objects, name, Ta, Tb, K, second_camera=second)
        run_dm.install_topdown_grasps(rt, cfg, force=True)
        n2 = run_dm.install_camera_measure(rt, cfg, known=[])
        _, fix = rt._localize(name)
        grasps = rt._plan_grasps(fix, name)
        what = f"{'can' if truth is can else 'cup'} with {'two cameras' if second else 'one camera'}"
        if isinstance(grasps, str) or not grasps:
            check(False, f"{what}: no grasp planned")
            continue
        down = [g for g in grasps if abs(g.approach[2] + 1) < 1e-6][0]            # the straight-down grasp
        off = float(np.linalg.norm(down.position[:2] - truth["centre"]))
        dz = float(abs(down.position[2] - (TABLE_Z + 0.5 * truth["height"])))
        tol = 0.004 if second else 0.02
        print(f"     {what}: grasp {off * 1000:.1f} mm from the true middle, {dz * 1000:.1f} mm from half height"
              f" | {notes[0] if notes else ''}")
        check(n2 == (1 if second else 0) and off < tol and dz < 0.006, f"{what}: grasp is on the object")
        check(abs(float(fix.points[:, 2].max()) - (TABLE_Z + truth["height"])) < 0.006,
              f"{what}: the object's stored top height is right")
        if truth is cup:
            check(grasps[0].approach[2] > -0.99, "cup (9 cm rim): the leaning grasp is tried first")

    print("3. configuration:")
    with tempfile.TemporaryDirectory() as tmp:
        base = dict(arm="rebot_dm", classes=None, table_z=0.0, sim_camera=None, sim_camera2=None, port=8090, set=None)
        a = argparse.Namespace(camera="uvc", camera2="uvc", rgb_only=False, **base)
        d = run_dm.build_cfg(a, Path(tmp), {}, {})._data
        check(len(d["cameras"]) == 2 and d["cameras"][1]["fuse_beliefs"] is False and d["camera"] is d["cameras"][0],
              "two real cameras: both loaded, the second one does not add duplicate objects")
        sim = {"cam_corner": Ta.tolist(), "cam_far": Tb.tolist()}
        scene = {"scene": "real", "table_z": TABLE_Z, "bins": {}}
        a = argparse.Namespace(camera="dm_sim_overhead", camera2="dm_sim_overhead", rgb_only=True,
                               **{**base, "table_z": None})
        d = run_dm.build_cfg(a, Path(tmp), scene, sim)._data
        c0, c1 = d["cameras"]
        check(c0["camera"] == "cam_corner" and c1["camera"] == "cam_far" and c0["rgb_only"] and c1["rgb_only"],
              "simulator rehearsal: corner + far cameras, depth thrown away")
        n = np.array(c0["table_plane_cam"])
        on_table = np.linalg.inv(Ta) @ np.array([0.3, 0.1, TABLE_Z, 1.0])
        check(abs(n[:3] @ on_table[:3] + n[3]) < 1e-9, "simulator rehearsal: the table plane is right for the camera")

    print()
    if failures:
        print("FAILED:", *failures, sep="\n  - ")
        return 1
    print("All wiring checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
