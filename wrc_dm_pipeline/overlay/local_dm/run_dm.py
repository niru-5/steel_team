#!/usr/bin/env python
"""Run the WRC pipeline on the reBot B601-DM through the DM bridge (hackathon DM port).

Start local_dm/dm_bridge_server.py first (./START_WRC.sh dm does both). Then:

  python local_dm/run_dm.py --dashboard                 # web UI at http://127.0.0.1:8090/
  python local_dm/run_dm.py --task "pick and place the red cube"
  python local_dm/run_dm.py --task "put the red cube on the blue block" --llm claude

  --llm claude   Claude plans from the camera image + tool results (needs ANTHROPIC_API_KEY,
                 read from the environment or from wrc/.env). Default when a key is present.
  --llm mock     no LLM: only WRC's built-in reflex commands work (look around / go home /
                 open the gripper / pick and place X / put X on Y).

Same wiring as wrc_demo.apps.demo (build_runtime + AgentOrchestrator); the only
additions are the DM arm + sim camera backends (wrc_dm_plugin) and the Claude client.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("YOLO_OFFLINE", "True")
os.chdir(REPO)   # YOLOE resolves its text encoder (mobileclip2_b.ts) relative to the CWD


# The detector names objects from a top-down camera image, so its labels do not always match the
# user's words (a cylinder seen from above is a disc and gets labelled "ball"). Without this note
# Claude sometimes refuses to act because "no yellow cylinder was detected".
LABEL_HINT = """Notes for this robot:
- Object names in tool results come from a camera detector looking straight down, and may differ from \
the user's words. A cylinder or can seen from above is often labelled "ball"; "cube", "block" and "box" \
are used interchangeably.
- If exactly one detected object fits the user's description by colour and rough shape, act on it and \
pass the detector's label to the tool. Say in your summary which label you used.
- pick_and_place without a destination moves the object to a free spot on the table that the robot \
chooses; other objects are not disturbed.
- An entry with state "remembered" is only where an object was seen EARLIER; it is not a second object. \
When a "visible" entry with the same label exists, the visible one is the object: use it and ignore the \
remembered one. Never call a request ambiguous because of a remembered entry.
- Only if several VISIBLE objects could fit, or none does, do not guess: call task_done(success=false) \
and say what you see."""


def load_dotenv(path: Path) -> None:
    """Minimal .env loader: KEY=VALUE lines; existing environment variables win."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if v.strip():
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# Spots where an object may be put down when the task names no destination. Candidates are
# visible to an overhead camera with the arm at home; each one is checked with WRC's own IK
# before it is used, because a straight-down put-down reaches less far than a grasp.
FREE_SPOTS = [(x / 100, y / 100) for x in (24, 28, 32, 36, 40) for y in range(-20, 21, 4)]


def _can_put_down(runtime, cfg, x: float, y: float, z: float | None = None) -> bool:
    """Would WRC's place_at find a pose here (at tool height z, default WRC's release height)?
    Same poses and yaw candidates as skill_place_at."""
    import math

    import numpy as np
    import pinocchio as pin

    g, table_z = cfg.grasp, float(cfg.safety.get("table_z", 0.0))
    release_z = float(z) if z is not None else table_z + float(g.get("release_height_m", 0.05))
    z_cap = float(g.get("topdown_z_max", 0.15)) - 0.005
    release_z = min(release_z, z_cap)
    hover_z = min(release_z + float(g.get("pregrasp_offset_m", 0.12)), z_cap)
    pitch = float(g.get("approach_pitch_rad", 1.2))
    home = np.asarray(cfg.arm.home_q, dtype=float)
    for yaw in (0.0, math.atan2(y, x), math.pi / 4, -math.pi / 4, math.pi / 2, -math.pi / 2):
        T = np.eye(4)
        T[:3, :3] = pin.rpy.rpyToMatrix(0.0, pitch, float(yaw))
        T[:3, 3] = [x, y, hover_z]
        pre = runtime.kin.ik(T, home)
        if not pre.success:
            continue
        T[2, 3] = release_z
        if runtime.kin.ik(T, pre.q).success:
            return True
    return False


def free_spot(runtime, cfg, default):
    """The reachable candidate spot furthest from every object the camera currently sees."""
    def recent():   # seen in the last few seconds (ignores old "remembered" ghosts)
        return [o["position"][:2] for o in runtime.beliefs.summary() if float(o.get("age_s", 99)) <= 5.0]
    seen = recent()
    deadline = time.monotonic() + 4.0
    while not seen and time.monotonic() < deadline:      # right after start-up the first detection may be pending
        time.sleep(0.2)
        seen = recent()

    table_z = float(cfg.safety.get("table_z", 0.0))

    def clearance(s):
        return min((((s[0] - p[0]) ** 2 + (s[1] - p[1]) ** 2) ** 0.5 for p in seen), default=1.0)

    for spot in sorted(FREE_SPOTS, key=clearance, reverse=True):
        if clearance(spot) < 0.09:
            break
        # which object will be held is not known yet: the spot must work for a low and a tall one
        if all(_can_put_down(runtime, cfg, *spot, z=table_z + dz) for dz in (0.035, 0.085)):
            return list(spot)
    return list(default)


BIN_HOVER_M = 0.05     # the tool arrives this far above the bin release height, then comes down


def install_scene_places(runtime, arm, cfg, scene: dict) -> None:
    """Teach WRC's put-down two things it does not know by itself:

    1. how far the held object hangs below the tool. WRC always releases 5 cm above the table,
       which pushes a tall object (a 13 cm can held at mid-height) into the table. The tool height
       at the moment the jaws closed on the standing object tells how far it hangs.
    2. the scene's named places (the foam bin's compartments). WRC can only place on an object the
       camera has detected; "the bin" detected as one white box would put things on the divider.
       A named place is released from above the rim with the tool leaning forward as little as
       reach allows (see bin_release).
    """
    import re

    from wrc_demo.types import SkillError

    table_z = float(cfg.safety.get("table_z", 0.0))
    bins = scene.get("bins") or {}
    g = cfg._data["grasp"]
    place_at, place_on = runtime.skill_place_at, runtime.skill_place_on_object

    def hang() -> float:
        z = getattr(arm, "last_close_tcp_z", None)
        if z is None:
            return float(g.get("release_height_m", 0.05)) - 0.008
        return min(max(float(z) - table_z, 0.015), 0.16)

    def grasp_pitch() -> float:
        """How steeply the tool pointed down when it closed on the object (pi/2 = straight down)."""
        p = getattr(arm, "last_close_pitch", None)
        return 1.5707963267948966 if p is None else min(max(float(p), 0.6), 1.5707963267948966)

    def place_at_aware(x, y, z=None):
        if z is None:
            z = table_z + hang() + 0.008        # object bottom ends ~8 mm above the table
        old = g.get("approach_pitch_rad")
        g["approach_pitch_rad"] = grasp_pitch()  # same tool angle as at the grasp: the object stands as it stood
        try:
            return place_at(float(x), float(y), float(z))
        finally:
            g["approach_pitch_rad"] = old

    def bin_for(label: str):
        key = re.sub(r"[^a-z0-9]", "", str(label).lower())
        for name in bins:
            if re.sub(r"[^a-z0-9]", "", name.lower()) in key:
                return name
        if not bins or not re.search(r"bin|box|container|tray|basket|crate", str(label).lower()):
            return None
        names = sorted(bins, key=lambda n: (bins[n]["center"][0] ** 2 + bins[n]["center"][1] ** 2))
        far = re.search(r"far|second|other|right|back|1|two", str(label).lower())
        return names[-1] if (far and len(names) > 1) else names[0]

    def bin_release(name: str):
        """Joint poses for letting go above a compartment: (q_above, q_release, tool_xyz, pitch) or None.

        Straight down the arm cannot reach over the bin (measured: no solution above either
        compartment), so the tool leans outwards, towards the bin, as little as reach allows. The
        held object then hangs along the leaning tool: its bottom is `h` further along the tool
        axis, so the tool stops short of the compartment centre by that much, and high enough for
        the object's lowest edge to clear the rim by 2.5 cm.
        """
        import math

        import numpy as np
        import pinocchio as pin

        bn, h = bins[name], hang()
        r = min(float(getattr(runtime, "_dm_top_width", 0.06)), 0.10) / 2
        cx, cy = bn["center"]
        seeds = [np.asarray(runtime.arm.get_state().q, dtype=float), np.asarray(cfg.arm.home_q, dtype=float)]
        for pitch in (1.2, 1.0, 0.8, 0.65):
            # the object's own axis: it stood upright when the tool gripped it at grasp_pitch()
            e = pitch + math.pi / 2 - grasp_pitch()
            sp, cp = math.sin(e), math.cos(e)
            z = float(bn["rim_z"]) + 0.025 + h * sp + r * abs(cp)
            for shift in (0.0, 0.03):                        # compartment centre, then 3 cm nearer the robot
                d = math.hypot(cx, cy)
                bx, by = cx * (1 - shift / d), cy * (1 - shift / d)
                yaw = math.atan2(by, bx)
                T = np.eye(4)
                T[:3, :3] = pin.rpy.rpyToMatrix(0.0, pitch, yaw)
                T[:3, 3] = [bx - h * cp * math.cos(yaw), by - h * cp * math.sin(yaw), z + BIN_HOVER_M]
                for seed in seeds:
                    pre = runtime.kin.ik(T, seed)
                    if not pre.success:
                        continue
                    T[2, 3] = z
                    low = runtime.kin.ik(T, pre.q)
                    if low.success:
                        return pre.q, low.q, T[:3, 3].copy(), pitch
                    T[2, 3] = z + BIN_HOVER_M
        return None

    def drop_in_bin(label, name):
        """Carry the held object over the compartment and let go. Every move goes through WRC's
        SafeArm, so the safety harness vets each waypoint as for any other skill."""
        import numpy as np

        runtime._adopt_unknown_held()
        if not runtime.held_object:
            raise SkillError("not holding anything")
        found = bin_release(name)
        if found is None:
            raise SkillError(f"{name} is out of the arm's reach for a release above its rim")
        q_above, q_release, tool, pitch = found
        runtime.memory.add("note", f"'{label}' is the named place {name}: letting go above its rim, tool at "
                                   f"({tool[0]:.2f}, {tool[1]:.2f}, {tool[2]:.2f}) leaning {90 - pitch * 57.3:.0f} deg outwards")
        if not runtime.arm.move_joints(q_above, duration_s=float(g.get("move_duration_s", 2.5))):
            raise SkillError(f"did not get above {name}")
        if not runtime.arm.move_joints(q_release, duration_s=1.5):
            raise SkillError(f"did not reach the release pose above {name}")
        err = float(np.linalg.norm(runtime.kin.fk(np.asarray(runtime.arm.get_state().q, float))[:3, 3] - tool))
        if err > 0.02:
            raise SkillError(f"release pose above {name} missed by {err * 1000:.0f} mm; not letting go")
        runtime.arm.set_gripper(runtime._grip_open, effort=0.6)
        time.sleep(max(float(g.get("close_settle_s", 0.0)), 0.6))
        # Same book-keeping as WRC's place_at: from here the object is placed.
        placed = runtime.held_object
        inside = [float(bins[name]["center"][0]), float(bins[name]["center"][1]), float(bins[name]["floor_z"]) + 0.03]
        runtime.beliefs.update(runtime._held_det_label or placed, np.array(inside), 0.8, color=runtime._held_color)
        runtime.held_object = None
        runtime._held_det_label = None
        runtime._held_color = None
        runtime.memory.add("action", f"dropped {placed!r} into {name}")
        try:
            runtime.arm.move_joints(q_above, duration_s=1.5)
        except Exception as e:  # noqa: BLE001 -- the drop already happened; move_home parks the arm
            runtime.memory.add("note", f"dropped, but moving back up was stopped: {e}")
        return {"placed": placed, "at": [round(v, 3) for v in inside], "placed_in": name}

    def place_on_aware(label):
        name = bin_for(label)
        if name is None:
            return place_on(label)
        return drop_in_bin(label, name)

    runtime._dm_table_release_z = lambda: table_z + hang() + 0.008

    runtime.skill_place_at = place_at_aware
    runtime.skill_place_on_object = place_on_aware


WIDE_OBJECT_M = 0.07   # wider than this at the top: do not try to pass the fingers over the rim first


def install_topdown_grasps(runtime, cfg, force: bool = False) -> bool:
    """For a camera that looks at the table from the side: grasp standing objects straight down.

    WRC's first-choice planner dives along the camera's line of sight, which is right for a camera
    above the table and wrong for one in a corner (it reaches for the object's side from 50 cm away
    and closes on air). Here the object's centre comes from its top: the highest slice of its
    points is the lid or rim, whose middle is the object's axis however slanted the view is
    (measured in the team's scene: 1-2 mm off the true centre). The gripper then comes straight
    down over that centre and closes at half height -- the same grasp the team's own script uses.
    Flat objects (under 4 cm) are left to WRC's planners. Returns True when installed.

    Without a depth camera the object's place and size come from install_camera_measure (below)
    instead of from 3D points; `force` installs this planner whatever the camera's angle.
    """
    import numpy as np
    import pinocchio as pin

    from wrc_demo.grasping.obb_grasp import _approach_rotation
    from wrc_demo.types import Grasp, transform_points

    try:
        T = np.asarray(runtime.extrinsics.cam_to_base(), dtype=float)
    except Exception:  # noqa: BLE001 -- no extrinsics: nothing to decide on
        return False
    if T[2, 2] < -0.9 and not force:   # optical axis within ~25 deg of straight down: WRC's planner is fine
        return False
    table_z = float(cfg.safety.get("table_z", 0.0))
    orig = runtime._plan_grasps

    def object_points(fix):
        frame, det = getattr(runtime, "last_frame", None), getattr(fix, "detection", None)
        mask = getattr(det, "mask", None)
        if frame is not None and mask is not None and getattr(frame, "depth_m", None) is not None \
                and mask.shape == frame.depth_m.shape:
            vs, us = np.nonzero(mask.astype(bool) & (frame.depth_m > 0))
            z, K = frame.depth_m[vs, us], frame.K
            pts = np.stack([(us - K[0, 2]) * z / K[0, 0], (vs - K[1, 2]) * z / K[1, 1], z], axis=1)
            return transform_points(np.asarray(runtime.extrinsics.cam_to_base(), dtype=float), pts)
        return np.asarray(fix.points, dtype=float)

    def plan(fix, label=None):
        m = getattr(fix, "dm_measure", None)
        if m is not None:
            # No depth camera: place, height and width were measured from the camera outlines.
            height = float(m.height)
            if height < 0.04:
                return orig(fix, label)
            centre, size, source = np.asarray(m.centre, dtype=float), np.array([m.width_min, m.width]), m.method
            yaws = []                # two outlines do not say which way an oblong object lies
        else:
            pts = object_points(fix)
            obj = pts[pts[:, 2] > table_z + 0.015]                # drop table pixels at the mask edge
            if len(obj) < 40:
                return orig(fix, label)
            top_z = float(np.percentile(obj[:, 2], 98))
            height = top_z - table_z
            if height < 0.04:
                return orig(fix, label)
            rim = obj[obj[:, 2] > top_z - max(0.012, 0.1 * height)]
            lo, hi = np.percentile(rim[:, :2], [2, 98], axis=0)
            centre, size, source = (lo + hi) / 2, hi - lo, "side camera"
            # Straight down. Jaw direction: across the narrow side if the top is clearly elongated, else any.
            yaws = [0.0 if size[0] < size[1] else np.pi / 2] if max(size) > 1.3 * min(size) else []
        width = float(min(size.min() + 0.01, runtime._max_width))  # the jaws are opened fully anyway
        radial = float(np.arctan2(centre[1], centre[0]))
        yaws += [radial, radial + np.pi / 2, 0.0, np.pi / 2, np.pi / 4, -np.pi / 4]
        down = [(np.pi / 2, y) for y in yaws]
        # Leaning in from the robot's side, jaws level. Two uses: (1) an object nearly as wide as
        # the open gripper (the 9 cm cup, 9.1 cm jaws): straight down the fingers must pass the rim
        # with half a millimetre to spare and one ends up inside the cup; leaning, they only meet
        # the cup lower down where it is narrower (about 1 cm spare per side). (2) an object too
        # far away for a straight-down pose.
        wide = float(size.max()) > WIDE_OBJECT_M
        pitches = (1.0, 0.8, 1.2) if wide else (1.2, 1.0, 0.8)
        lean = [(p, radial) for p in pitches] + [(1.0, radial - 0.6), (1.0, radial + 0.6)]
        conf = float(getattr(fix.detection, "conf", 1.0))
        grasps = []
        for i, (pitch, yaw) in enumerate(lean + down if wide else down + lean):
            leaning = pitch < np.pi / 2 - 1e-6
            # close at half height; a leaning grip a little higher, so the object hangs below the jaws
            frac = 0.55 if leaning else 0.5
            pos = np.array([centre[0], centre[1], table_z + max(frac * height, 0.03)])
            # Leaning: the same angle convention as WRC's place_at (tool points `pitch` below level,
            # towards `yaw`; jaws level). WRC's own _approach_rotation measures its angle from the
            # vertical instead, so it is only used for the straight-down case it was tested with.
            R = pin.rpy.rpyToMatrix(0.0, float(pitch), float(yaw)) if leaning else _approach_rotation(np.pi / 2, float(yaw))
            grasps.append(Grasp(position=pos, rotation=R, width_m=width, approach=R[:, 0].copy(),
                                quality=conf * (1.0 - 0.04 * i), label=label or fix.label))
        runtime._last_grasp_z_nudge = 0.0
        runtime._dm_top_width = float(size.max())                 # the bin release needs the object's width
        runtime.memory.add("note", f"{source}: {label or fix.label!r} stands at ({centre[0]:.3f}, {centre[1]:.3f}), "
                                   f"{height * 100:.0f} cm tall, top {size.max() * 100:.0f} cm wide -> "
                                   f"{'leaning grasp first (nearly as wide as the gripper)' if wide else 'straight-down grasp first, leaning if out of reach'}")
        return grasps

    runtime._plan_grasps = plan
    return True


def install_camera_measure(runtime, cfg, known: list, always: bool = False) -> int:
    """No depth camera: measure each object from its outline in the calibrated cameras.

    A webcam gives no distance, so WRC assumes every pixel lies on the table. That is wrong for
    anything standing on it: a 13 cm can is "seen" lying flat behind where it stands, and the
    arm closes at the wrong height. Here, whenever an object is located for a pick, its outline
    in the main camera is paired with the same object's outline in the second scene camera and
    its place, height and width are worked out from the two (local_dm/two_view.py). With no
    second camera, or when the second camera does not see the object, the main camera alone is
    used (less exact). A table of known object sizes replaces the measured size when it fits.

    `always` measures this way even when the camera has real depth (to compare the two).
    Returns the number of second cameras in use.
    """
    import two_view as tv
    from wrc_demo.perception.colors import detection_color
    from wrc_demo.perception.grounding import oriented_bbox
    from wrc_demo.types import SkillError

    table_z = float(cfg.safety.get("table_z", 0.0))
    orig = runtime._localize
    cams = getattr(runtime.watcher, "_cams", None) if runtime.watcher else None
    main_name = str(getattr(cams[0].stream, "name", "main camera")) if cams else "main camera"
    seconds = list(cams[1:]) if cams else []

    def other_views():
        views = []
        for cam in seconds:
            name = str(getattr(cam.stream, "name", "second camera"))
            try:
                f = cam.stream.get_frame()
                if f.T_base_cam is not None:       # a camera riding on the arm: not a fixed viewpoint
                    continue
                T = cam.extrinsics.cam_to_base()
                dets = runtime.detector.detect(f, classes=runtime._default_classes)
            except Exception as e:  # noqa: BLE001 -- a camera hiccup must not stop the pick
                runtime.memory.add("note", f"camera '{name}' gave no usable picture ({str(e)[:80]})")
                continue
            for d in dets:
                if d.mask is not None and d.mask.shape == f.rgb.shape[:2]:
                    views.append(tv.View(name, f.K, T, d.mask, d.label, float(d.conf), detection_color(f.rgb, d)))
        return views

    def localize(query, spatial_hint=None):
        frame, fix = orig(query, spatial_hint=spatial_hint)
        det = getattr(fix, "detection", None)
        mask = getattr(det, "mask", None)
        if (frame is None or frame is not runtime.last_frame or mask is None
                or mask.shape != frame.rgb.shape[:2]
                or (frame.depth_source == "sensor" and not always)):
            return frame, fix
        try:
            T = frame.T_base_cam if frame.T_base_cam is not None else runtime.extrinsics.cam_to_base()
            view = tv.View(main_name, frame.K, T, mask, det.label, float(det.conf), detection_color(frame.rgb, det))
            m = tv.measure(view, other_views(), table_z, known=known)
        except Exception as e:  # noqa: BLE001 -- measuring is an improvement, never a reason to crash
            runtime.memory.add("note", f"could not measure {query!r} from the camera outlines ({str(e)[:100]})")
            return frame, fix
        if m is None:
            if tv.why_unusable(view) == "cut off":
                # Half an outline gives a wrong size, and the flat-table guess is centimetres off for
                # anything standing up. Say so instead of reaching for the wrong place.
                raise SkillError(f"cannot measure {query!r}: it is cut off at the edge of the camera picture "
                                 "and no second camera sees it whole. Move it further into view.")
            runtime.memory.add("note", f"{query!r} is seen from almost straight above; using the flat-table estimate")
            return frame, fix
        if m.height < 0.04:                        # flat on the table: the flat-table estimate is right
            return frame, fix
        pts = m.points()
        fix.position, fix.extent, fix.axes = oriented_bbox(pts)
        fix.points, fix.dm_measure = pts, m
        runtime.memory.add("note", f"measured {query!r} without depth, {m.describe()}")
        return frame, fix

    runtime._localize = localize
    return len(seconds)


# Detector word lists per simulator scene. In the real-workspace scene the plain red can is
# recognised as "red cup" and the foam bin as "white box" (tested on the cam_corner view).
SCENE_CLASSES = {"real": "cup,red cup,white box", "demo": "red block,blue block,yellow ball"}


def bridge_scene() -> tuple[dict, dict]:
    """Ask the arm server what it is running: scene info, and (sim only) its cameras' true poses."""
    from wrc_dm_plugin import BridgeClient
    try:
        c = BridgeClient()
        info = c.call("info")[0]
    except RuntimeError:
        return {}, {}
    cams = {}
    for name in info.get("cameras", []):
        try:
            hdr = c.call("frame", cam=name)[0]
            cams[name] = hdr["T_base_cam"]
        except RuntimeError:
            pass
    c.close()
    return info, cams


def table_plane_in_camera(T_base_cam, table_z: float) -> list:
    """The table surface (z = table_z in the robot base) written in a camera's own axes: nx, ny, nz, d."""
    import numpy as np
    T = np.asarray(T_base_cam, dtype=float)
    n = T[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    return [float(n[0]), float(n[1]), float(n[2]), float(T[2, 3] - table_z)]


def build_cfg(args, run_dir: Path, scene: dict, sim_cams: dict):
    from wrc_demo.config import load_demo_config

    names = [args.camera] + ([args.camera2] if getattr(args, "camera2", None) else [])
    cfg = load_demo_config(cameras=names, arm=args.arm, llm="mock")
    d = cfg._data
    kind = scene.get("scene")
    table_z = args.table_z if args.table_z is not None else (scene.get("table_z") if scene.get("table_z") is not None else 0.03)
    d["safety"]["table_z"] = float(table_z)
    classes = args.classes or SCENE_CLASSES.get(kind, SCENE_CLASSES["demo"])
    d["detect_classes"] = [c.strip() for c in classes.split(",") if c.strip()]
    cams = d.get("cameras") or [d["camera"]]
    for i, c in enumerate(cams):
        if str(c.get("type")) != "dm_bridge" or not sim_cams:
            continue
        # The simulator knows exactly where its cameras are; a real camera needs calibrate_camera.py.
        if i == 0:
            cam = args.sim_camera or ("cam_corner" if kind == "real" else str(c.get("camera", "overhead_rgb")))
        else:
            first = str(cams[0].get("camera"))
            cam = getattr(args, "sim_camera2", None) or ("cam_far" if kind == "real" else
                                                         next((n for n in sim_cams if n != first), first))
        if cam not in sim_cams:
            raise SystemExit(f"the simulator has no camera {cam!r} (it has: {', '.join(sim_cams)})")
        c["camera"] = cam
        if i > 0:
            c["name"] = cam                      # (the main camera keeps its profile name)
        c["extrinsics"] = {"mode": "eye_to_hand", "T": sim_cams[cam]}
        if getattr(args, "rgb_only", False):
            # Rehearse plain webcams: throw the simulator's depth away. WRC then lays every pixel
            # on the table plane, exactly as it does for a real RGB camera.
            c["rgb_only"] = True
            c["table_plane_cam"] = table_plane_in_camera(sim_cams[cam], float(table_z))
    for c in cams[1:]:
        # The second camera is for measuring. Its own flat-table positions of tall objects differ
        # from the main camera's, and fusing both would show every tall object twice.
        c["fuse_beliefs"] = False
    if kind == "real":
        # The team's measured workspace: table 1.2 cm below the arm base, left wall at y = +0.60,
        # right wall at y = -0.38, foam bin on the left. Widen WRC's box to reach the bin, keep the
        # tool out of the bin's volume, and lift WRC's RS-specific ceiling for straight-down poses.
        d["safety"]["workspace"] = {"min": [0.08, -0.34, float(table_z) - 0.01], "max": [0.56, 0.57, 0.60]}
        bins = scene.get("bins") or {}
        if bins:
            x0 = min(b["center"][0] - b["half"][0] for b in bins.values()) - 0.03
            x1 = max(b["center"][0] + b["half"][0] for b in bins.values()) + 0.03
            y0 = min(b["center"][1] - b["half"][1] for b in bins.values()) - 0.03
            rim = max(b["rim_z"] for b in bins.values())
            d["safety"]["keep_out"] = [{"min": [x0, y0, float(table_z) - 0.02], "max": [x1, 0.62, rim]}]
        d["grasp"]["topdown_z_max"] = 0.45
    d["grasp"]["memory_path"] = str(run_dir / "grasp_memory.json")
    # WRC retries a failed pick up to 8 times because its learned planner samples new grasps each
    # time. The planners used here give the same answer for the same scene, so 3 is plenty.
    d["grasp"]["max_pick_attempts"] = 3
    d["grasp"]["drop_zone"] = [0.25, 0.14]           # free spot on the table, inside the workspace
    # Put objects down with the gripper pointing straight down. WRC's default tilts the tool 21 deg
    # (tuned for the RS arm); objects are grasped vertically here, so a tilted release sets tall
    # objects down at an angle and they end up leaning or fall over (measured: 30 deg lean).
    d["grasp"]["approach_pitch_rad"] = 1.5707963267948966
    d["stream"]["host"] = "127.0.0.1"                # dashboard on localhost only
    d["stream"]["port"] = args.port
    for kv in args.set or []:                        # e.g. --set grasp.approach_pitch_rad=1.5708
        key, val = kv.split("=", 1)
        node = d
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = json.loads(val)
    return cfg


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default=None, help="run one task and exit")
    p.add_argument("--dashboard", action="store_true", help="serve the web dashboard and wait for tasks")
    p.add_argument("--llm", choices=["auto", "claude", "mock"], default="auto")
    p.add_argument("--arm", default="rebot_dm", help="arm profile in configs/arms/")
    p.add_argument("--camera", default="dm_sim_overhead", help="camera profile in configs/cameras/")
    p.add_argument("--classes", default=None,
                   help="detector vocabulary: object names the camera should look for, comma separated "
                        "(default depends on the simulator scene)")
    p.add_argument("--table-z", type=float, default=None,
                   help="table height above the arm base plane [m] (default: what the simulator reports, else 0.03)")
    p.add_argument("--sim-camera", default=None,
                   help="which simulator camera WRC looks through (default: cam_corner in the real-workspace "
                        "scene, overhead_rgb in the demo scene)")
    p.add_argument("--camera2", default=None,
                   help="a second fixed scene camera (profile in configs/cameras/). With cameras that have no "
                        "depth, objects are measured from their outlines in both (see local_dm/README.md)")
    p.add_argument("--sim-camera2", default=None,
                   help="which simulator camera --camera2 looks through (default: cam_far in the real-workspace scene)")
    p.add_argument("--rgb-only", action="store_true",
                   help="simulator only: throw the rendered depth away, to rehearse plain webcams")
    p.add_argument("--known-objects", default=None, metavar="YAML",
                   help="table of known object sizes, measured with a ruler (local_dm/known_objects_example.yaml)")
    p.add_argument("--measure", choices=["auto", "outlines", "depth"], default="auto",
                   help="where an object's size comes from: 'auto' = the camera outlines whenever the camera has "
                        "no depth sensor; 'outlines' = always; 'depth' = never (WRC's flat-table guess without depth)")
    p.add_argument("--drop-zone", default="auto",
                   help="where 'pick X' with no destination puts the object: 'auto' = a free spot on the table "
                        "chosen before every task (so objects do not pile up), or fixed 'x,y' in metres")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--reflex", action="store_true",
                   help="with Claude: try WRC's keyword shortcuts first (faster, no API call, but a "
                        "free-form sentence can be misread). Without Claude they are always on.")
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--notes", action="store_true", help="with --task: print the robot's own step-by-step notes")
    p.add_argument("--json-out", default=None, help="with --task: also write the result as JSON (for test scripts)")
    p.add_argument("--set", action="append", metavar="KEY=JSON", help="override a config value")
    args = p.parse_args(argv)
    if not args.task and not args.dashboard:
        p.error("give --task \"...\" or --dashboard")

    load_dotenv(REPO / ".env")
    import wrc_dm_plugin
    wrc_dm_plugin.register()

    from wrc_demo.agent.llm import make_llm
    from wrc_demo.agent.orchestrator import AgentOrchestrator
    from wrc_demo.agent.reflex import ExperienceMemory, FastPlanner
    from wrc_demo.apps.demo import _print_report, build_runtime, shutdown_runtime

    run_dir = Path(args.run_dir) if args.run_dir else REPO / "runs" / ("dm_" + time.strftime("%Y%m%d_%H%M%S"))
    scene, sim_cams = bridge_scene()
    cfg = build_cfg(args, run_dir, scene, sim_cams)
    if scene:
        print(f"[wrc-dm] arm server: {scene.get('robot')} arm, scene '{scene.get('scene')}', "
              f"camera '{cfg._data['camera'].get('camera')}', looking for: {', '.join(cfg._data['detect_classes'])}",
              flush=True)

    have_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    use_claude = args.llm == "claude" or (args.llm == "auto" and have_key)
    if args.llm == "claude" and not have_key:
        print("ERROR: --llm claude needs ANTHROPIC_API_KEY (environment or wrc/.env)", file=sys.stderr)
        return 2
    if use_claude:
        from claude_llm import ClaudeLLM
        hint = LABEL_HINT
        if scene.get("bins"):
            names = ", ".join(f"'{b}'" for b in scene["bins"])
            hint += ("\n- This workspace has a foam bin with compartments you can use as the destination of "
                     f"pick_and_place: {names} (bin_0 is the one nearer the robot). \"the bin\" means bin_0. "
                     "Pass the name exactly, e.g. destination=\"bin_0\". The detector may list the bin as "
                     "\"white box\" and a plain can as \"red cup\".")
        llm = ClaudeLLM(system_suffix=hint)
        llm_name = f"claude ({llm.model}, effort {llm.effort})"
    else:
        llm = make_llm(cfg.llm)
        llm_name = "mock (reflex commands only)"

    print(f"[wrc-dm] arm={args.arm} camera={args.camera} llm={llm_name}", flush=True)
    print(f"[wrc-dm] traces -> {run_dir}", flush=True)
    try:
        runtime, arm = build_runtime(cfg, run_dir, view=False, serve=args.dashboard)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if args.dashboard and runtime.stream_server is None:
        print(f"ERROR: dashboard port {args.port} is already in use (another WRC dashboard running?)",
              file=sys.stderr)
        shutdown_runtime(runtime, arm)
        return 2

    # With Claude connected, Claude reads every instruction (WRC's keyword matcher misreads
    # free-form sentences, e.g. "stack it on top of the blue one" -> destination "top of the
    # blue one"). Without an LLM the keyword shortcuts are the only way to act.
    fast = FastPlanner(ExperienceMemory(run_dir / "experience.json")) if (args.reflex or not use_claude) else None
    if use_claude:                       # every Claude decision shows up in the dashboard narration
        def _show_decision(rec):
            what = f"{rec['tool']}({json.dumps(rec['args'])})" if rec["tool"] else "a text reply"
            runtime.memory.add("note", f"planner Claude ({rec['model']}) chose {what} -- {rec['s']} s, "
                                       f"{rec['in']} input tokens, API request {rec['request_id']}")
        llm.on_call = _show_decision

    if args.notes:                       # print the robot's own notes live (its memory only keeps 15 s)
        _add = runtime.memory.add

        def _add_and_print(kind, text, *a, **k):
            print(f"  note[{kind}] {text}"[:600], flush=True)
            return _add(kind, text, *a, **k)

        runtime.memory.add = _add_and_print

    agent = AgentOrchestrator(
        llm, runtime, advisor=None, max_steps=args.max_steps, decompose=False,
        fast_planner=fast, cfg=cfg,
    )

    install_scene_places(runtime, arm, cfg, scene)
    first = runtime.observe()
    no_depth = first.depth_source != "sensor"
    outlines = args.measure == "outlines" or (args.measure == "auto" and no_depth)
    if install_topdown_grasps(runtime, cfg, force=outlines):
        print("[wrc-dm] side-view camera: standing objects are grasped over their top centre (straight down, or leaning in when wide or far)", flush=True)
    if outlines:
        known = []
        if args.known_objects:
            import two_view
            known = two_view.load_known(args.known_objects)
        n2 = install_camera_measure(runtime, cfg, known, always=args.measure == "outlines")
        print(f"[wrc-dm] no depth camera: objects are measured from their outlines in "
              f"{'the two scene cameras' if n2 else 'the ONE camera (less exact: add --camera2)'}"
              + (f"; known sizes: {', '.join(k.name for k in known)}" if known else ""), flush=True)
    elif no_depth:
        print("[wrc-dm] WARNING: the camera has no depth and --measure depth is set: standing objects will be "
              "placed and sized wrongly", flush=True)

    default_drop = list(cfg._data["grasp"]["drop_zone"])
    if args.drop_zone != "auto":
        default_drop = [float(v) for v in args.drop_zone.split(",")]
        cfg._data["grasp"]["drop_zone"] = default_drop

    def put_down_if_still_holding():
        """A task that ends with something still in the gripper (its put-down failed) must not leave it
        there: the next task would open the gripper and drop it from wherever the arm is."""
        try:
            if not runtime.held_object or runtime.arm.harness.estopped:
                return
            held = runtime.held_object
            spot = free_spot(runtime, cfg, default_drop)
            res = runtime.execute("place_at", {"x": spot[0], "y": spot[1]})
            if res.get("ok"):
                runtime.memory.add("outcome", f"the task ended with '{held}' still in the gripper, so I put it "
                                              f"down on a free spot at ({spot[0]:.2f}, {spot[1]:.2f})")
            else:
                runtime.memory.add("outcome", f"I am still holding '{held}' and could not put it down: "
                                              f"{str(res.get('error'))[:120]}")
            print(f"[wrc-dm] still holding {held!r} after the task -> put down at {spot}: ok={res.get('ok')}", flush=True)
        except Exception as e:  # noqa: BLE001 -- clean-up must never mask the task's own result
            print(f"[wrc-dm] could not put down the held object: {e}", flush=True)

    def run_task(task: str):
        runtime.current_task = task
        try:
            if runtime.arm.harness.estopped:
                runtime.arm.resume()
            if args.drop_zone == "auto":
                cfg._data["grasp"]["drop_zone"] = free_spot(runtime, cfg, default_drop)
            try:
                report = agent.run_task(task)
            except RuntimeError as e:          # e.g. the Claude API rejected the key or is unreachable
                runtime.memory.add("outcome", f"could not do {task!r}: {e}")
                print(f"TASK {task!r}: FAILED before finishing | {e}", flush=True)
                raise
            if report.path == "llm" and not use_claude:
                runtime.memory.add("note", f"{task!r} is not a built-in reflex command and no LLM is "
                                           "connected (set ANTHROPIC_API_KEY).")
            print(f"TASK {task!r}: success={report.success} path={report.path} {report.duration_s}s | "
                  f"{report.summary[:300]}", flush=True)
            return report
        finally:
            put_down_if_still_holding()
            runtime.current_task = None

    def cancel_task():
        runtime.memory.add("outcome", "USER pressed STOP -- aborting the current action")
        runtime.arm.stop()

    try:
        # wait for the first detector pass so commands resolve against a warm world model
        deadline = time.monotonic() + 60
        while not runtime.beliefs.summary() and time.monotonic() < deadline:
            time.sleep(0.2)
        print("[wrc-dm] camera sees:", [(o["label"], o["position"]) for o in runtime.beliefs.summary()], flush=True)

        if args.dashboard:
            runtime.stream_server.set_task_fn(run_task)
            runtime.stream_server.set_cancel_fn(cancel_task)
            print(f"DASHBOARD: http://127.0.0.1:{args.port}/   (Ctrl+C to stop)", flush=True)
            while True:
                time.sleep(5)
        before = arm.sim_objects() if hasattr(arm, "sim_objects") else {}
        try:
            report = run_task(args.task)
        except RuntimeError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        _print_report(report)
        print("\n=== tool log ===")
        for e in report.tool_log:
            r = e.get("result", {})
            print(f"- {e.get('tool')}({json.dumps(e.get('args', {}))}) -> ok={r.get('ok')} {str(r.get('error') or '')[:300]}")
        after = arm.sim_objects() if hasattr(arm, "sim_objects") else {}
        if args.json_out:
            Path(args.json_out).write_text(json.dumps({
                "task": args.task, "success": bool(report.success), "path": report.path,
                "duration_s": report.duration_s, "steps": report.steps, "summary": report.summary,
                "tool_log": [{"tool": e.get("tool"), "args": e.get("args", {}),
                              "ok": e.get("result", {}).get("ok"),
                              "error": str(e.get("result", {}).get("error") or "")[:400]} for e in report.tool_log],
                "claude_calls": getattr(llm, "calls", []),
            }, indent=1))
        if after:
            print("\n=== simulator ground truth (verification only; WRC used the camera) ===")
            for name in after:
                b, a = before.get(name, [0, 0, 0]), after[name]
                moved = sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
                print(f"- {name}: {[round(v, 3) for v in b]} -> {[round(v, 3) for v in a]}  (moved {moved * 100:.1f} cm)")
        return 0 if report.success else 1
    except KeyboardInterrupt:
        return 0
    finally:
        shutdown_runtime(runtime, arm)


if __name__ == "__main__":
    sys.exit(main())
