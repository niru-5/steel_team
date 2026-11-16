#!/usr/bin/env python
"""Check local_dm/two_view.py on drawn test scenes (no robot, no simulator, no camera).

Objects of known size are placed at known spots, their outlines are drawn exactly as two (or
one) calibrated cameras would see them, and the measured position / height / width is compared
with the truth. Three camera arrangements are tried, plus soft outlines, a slightly wrong
calibration, a second object in the picture, the known-size table, and a blocked camera.

  python local_dm/two_view_selftest.py            # prints the error table; exit code 1 on failure

What this does NOT cover: a real detector's outlines (shadows, reflections, a wrong outline),
and real lens distortion. Those need the simulator's rendered pictures and then the real cameras.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import two_view as tv  # noqa: E402

W, H = 640, 480
TABLE_Z = -0.012


def intrinsics(fov_deg: float = 70.0) -> np.ndarray:
    fx = W / (2 * math.tan(math.radians(fov_deg) / 2))
    return np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1.0]])


def look_at(pos, target) -> np.ndarray:
    """Camera -> base for a camera at `pos` looking at `target`, image upright."""
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    z = target - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z])
    T[:3, 3] = pos
    return T


def solid(kind: str, cx: float, cy: float) -> tuple[np.ndarray, dict]:
    """Surface points of a test object standing on the table, and its true size."""
    ang = np.linspace(0, 2 * np.pi, 240, endpoint=False)
    circle = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    if kind == "can":
        h, top, base = 0.130, 0.066, 0.066
    elif kind == "cup":
        h, top, base = 0.110, 0.090, 0.060
    elif kind == "box":
        h, sx, sy = 0.100, 0.060, 0.040
        c = np.array([[sx, sy], [sx, -sy], [-sx, -sy], [-sx, sy]]) / 2
        edge = np.concatenate([np.linspace(c[i], c[(i + 1) % 4], 40) for i in range(4)])
        pts = np.concatenate([np.column_stack([edge + [cx, cy], np.full(len(edge), TABLE_Z + z)]) for z in (0.0, h)])
        return pts, {"height": h, "width": math.hypot(sx, sy), "width_min": sy, "centre": np.array([cx, cy])}
    else:
        raise ValueError(kind)
    pts = np.concatenate([
        np.column_stack([circle * base / 2 + [cx, cy], np.full(len(ang), TABLE_Z)]),
        np.column_stack([circle * top / 2 + [cx, cy], np.full(len(ang), TABLE_Z + h)]),
    ])
    return pts, {"height": h, "width": max(top, base), "width_min": max(top, base), "centre": np.array([cx, cy])}


def draw(points: np.ndarray, K: np.ndarray, T: np.ndarray, soft: int = 0, rng=None) -> np.ndarray:
    """The object's outline as the camera sees it (all test objects are convex)."""
    Tinv = np.linalg.inv(T)
    pc = points @ Tinv[:3, :3].T + Tinv[:3, 3]
    uv = (pc[:, :2] / pc[:, 2:3]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    big = 8                                                       # draw 8x larger, then shrink: sub-pixel edges
    canvas = np.zeros((H * big, W * big), np.uint8)
    hull = cv2.convexHull(np.round((uv + 0.5) * big).astype(np.int32))   # pixel i is centred on i (OpenCV)
    cv2.fillConvexPoly(canvas, hull, 255)
    mask = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_AREA) > 127
    if soft and rng is not None:                                  # a detector's outline is a pixel or two off
        k = np.ones((3, 3), np.uint8)
        op = cv2.dilate if rng.random() < 0.5 else cv2.erode
        mask = op(mask.astype(np.uint8), k, iterations=soft) > 0
    return mask


RIGS = {
    # name: (camera A position, target), (camera B position, target)
    "corner + far (both raised, like the team's simulator)":
        (((-0.05, -0.45, 0.55), (0.30, 0.05, 0.0)), ((0.85, 0.40, 0.45), (0.30, 0.00, 0.0))),
    "mounted + table level (one high, one 12 cm above the table)":
        (((0.30, -0.65, 0.60), (0.30, 0.05, 0.0)), ((0.85, 0.05, 0.11), (0.30, 0.00, 0.05))),
    "facing each other (the hardest arrangement)":
        (((0.30, -0.65, 0.45), (0.30, 0.00, 0.0)), ((0.30, 0.70, 0.35), (0.30, 0.00, 0.0))),
}
SPOTS = [(0.20, -0.15), (0.20, 0.20), (0.32, 0.00), (0.32, 0.22), (0.45, -0.12), (0.45, 0.15), (0.26, 0.08)]


def errors(m: tv.Measurement, truth: dict) -> tuple[float, float, float]:
    return (float(np.linalg.norm(m.centre - truth["centre"])), abs(m.height - truth["height"]),
            abs(m.width - truth["width"]))


def tilt(T: np.ndarray, deg: float, shift: float, rng) -> np.ndarray:
    """A slightly wrong calibration: the camera is believed `deg` off and `shift` metres away."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    R, _ = cv2.Rodrigues(axis * math.radians(deg))
    T2 = T.copy()
    T2[:3, :3] = R @ T[:3, :3]
    T2[:3, 3] += rng.normal(size=3) / math.sqrt(3) * shift
    return T2


def main() -> int:
    K = intrinsics()
    rng = np.random.default_rng(7)
    failures = []
    print("Errors are |measured - true| in millimetres: mean / worst over 7 spots on the table.\n")
    for rig_name, (ca, cb) in RIGS.items():
        Ta, Tb = look_at(*ca), look_at(*cb)
        print(f"== cameras: {rig_name}")
        for case, soft, calib in (("exact outlines", 0, None), ("soft outlines (1 px off)", 1, None),
                                  ("calibration 0.5 deg / 5 mm off", 0, (0.5, 0.005))):
            for kind in ("can", "cup", "box"):
                two, one = [], []
                for (x, y) in SPOTS:
                    pts, truth = solid(kind, x, y)
                    Ta_b = tilt(Ta, *calib, rng) if calib else Ta      # what the software believes
                    Tb_b = tilt(Tb, *calib, rng) if calib else Tb
                    va = tv.View("A", K, Ta_b, draw(pts, K, Ta, soft, rng), label=kind)
                    vb = tv.View("B", K, Tb_b, draw(pts, K, Tb, soft, rng), label=kind)
                    m = tv.measure(va, [vb], TABLE_Z)
                    if m is None or m.method != "two cameras":
                        failures.append(f"{rig_name} / {case} / {kind} at {(x, y)}: no two-camera result")
                        continue
                    two.append(errors(m, truth))
                    m1 = tv.measure(va, [], TABLE_Z)
                    if m1 is not None:
                        one.append(errors(m1, truth))
                two, one = np.array(two) * 1000, np.array(one) * 1000
                line = f"   {case:32s} {kind:4s} two cameras: position {two[:, 0].mean():4.1f}/{two[:, 0].max():4.1f}  " \
                       f"height {two[:, 1].mean():4.1f}/{two[:, 1].max():4.1f}  width {two[:, 2].mean():4.1f}/{two[:, 2].max():4.1f}"
                if len(one):
                    line += f"   | camera A alone: position {one[:, 0].mean():4.1f}/{one[:, 0].max():4.1f}  " \
                            f"height {one[:, 1].mean():4.1f}/{one[:, 1].max():4.1f}"
                print(line)
                # pass marks: round objects, cameras not facing each other, exact outlines
                if case == "exact outlines" and kind in ("can", "cup") and "facing" not in rig_name:
                    if two[:, 0].max() > 5 or two[:, 1].max() > 6 or two[:, 2].max() > 5:
                        failures.append(f"{rig_name} / {kind}: two-camera error above 5 mm position / 6 mm height / 5 mm width")
        print()

    # -- two objects in view: is each outline paired with the right one, whatever they are called?
    Ta, Tb = look_at(*RIGS["mounted + table level (one high, one 12 cm above the table)"][0]), \
        look_at(*RIGS["mounted + table level (one high, one 12 cm above the table)"][1])
    can_pts, can = solid("can", 0.22, -0.10)
    cup_pts, cup = solid("cup", 0.40, 0.14)
    others = [tv.View("B", K, Tb, draw(cup_pts, K, Tb), label="red cup"),      # deliberately mislabelled
              tv.View("B", K, Tb, draw(can_pts, K, Tb), label="cup")]
    m = tv.measure(tv.View("A", K, Ta, draw(can_pts, K, Ta), label="red cup"), others, TABLE_Z)
    ok = m is not None and errors(m, can)[0] < 0.005 and errors(m, can)[1] < 0.006
    print(f"pairing with two objects and swapped names: {'ok' if ok else 'FAILED'}"
          + (f" -> {m.describe()}" if m else ""))
    if not ok:
        failures.append("pairing: the can was matched with the wrong outline")

    # -- known sizes: the can is recognised by size although the detector calls it 'red cup'
    known = [tv.KnownObject("can", ["red cup", "can", "cup"], 0.130, 0.066),
             tv.KnownObject("cup", ["cup", "red cup"], 0.110, 0.090, base_width=0.060)]
    soft_a = draw(can_pts, K, Ta, 2, np.random.default_rng(1))
    m = tv.measure(tv.View("A", K, Ta, soft_a, label="red cup"), others, TABLE_Z, known=known)
    ok = m is not None and m.known == "can" and abs(m.height - 0.130) < 1e-9
    print(f"known sizes (can called 'red cup', outline 2 px off): {'ok' if ok else 'FAILED'}"
          + (f" -> {m.describe()}" if m else ""))
    if not ok:
        failures.append("known sizes: the can was not recognised")
    m = tv.measure(tv.View("A", K, Ta, draw(cup_pts, K, Ta), label="cup"), others, TABLE_Z, known=known)
    ok = m is not None and m.known == "cup"
    print(f"known sizes (cup): {'ok' if ok else 'FAILED'}" + (f" -> {m.describe()}" if m else ""))
    if not ok:
        failures.append("known sizes: the cup was not recognised")

    # -- second camera blocked (it sees nothing that fits): fall back to one camera
    m = tv.measure(tv.View("A", K, Ta, draw(can_pts, K, Ta), label="red cup"), [others[0]], TABLE_Z, known=known)
    ok = m is not None and m.method == "one camera" and errors(m, can)[0] < 0.02
    print(f"second camera blocked -> one camera: {'ok' if ok else 'FAILED'}" + (f" -> {m.describe()}" if m else ""))
    if not ok:
        failures.append("fallback: no usable single-camera result")

    print()
    if failures:
        print("FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
