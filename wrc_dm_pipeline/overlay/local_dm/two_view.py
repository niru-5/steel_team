"""Object position, height and width from plain RGB cameras (no depth sensor).

The problem this solves: a webcam gives no distance. WRC then assumes every pixel lies on the
table, which is right for the table and wrong for anything standing on it -- a 13 cm can is
"seen" lying flat behind where it really stands, so the arm closes at the wrong height.

What is used instead is the object's OUTLINE in each calibrated camera:

  * seen from above, the outline spans a range of directions from the camera. The object's
    middle lies on the line through the middle of that range. Two cameras give two such lines;
    where they cross is the object's position (the mounted camera and the table camera do not
    even have to look at the same side of it).
  * the range of directions times the distance is the object's width.
  * the highest outline point lies on the object's top edge. Its line of sight, followed out to
    the object's distance, gives the height. A camera near table level is the best one for this.
  * one camera alone still works, less exactly: the lowest outline point is where the object
    touches the table, and a line of sight meets the (known) table at one place only.

Everything is combined by how reliable each piece is (a camera almost level with the table is
good for height and poor for distance, and is weighted that way). An optional table of known
object sizes (measured once with a ruler) replaces the measured size when they agree.

Pure numpy; no robot, simulator or WRC imports. Frames: robot base (x ahead, y left, z up),
metres. Camera axes are OpenCV's (x right, y down, z forward); `T` maps camera to base.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# A view that looks at the object more steeply than this (from the horizontal) cannot tell its
# height or its direction "seen from above": it is a top view. Such views are not used here.
STEEP_RAD = math.radians(68)
PIX_NOISE = 1.5          # outline uncertainty in pixels (detector masks are a little soft)
MAX_SIGMA = 0.03         # a position less certain than this [m] is not used


@dataclass
class View:
    """One camera's outline of one object."""
    name: str
    K: np.ndarray                 # (3, 3) intrinsics
    T: np.ndarray                 # (4, 4) camera -> robot base
    mask: np.ndarray              # (H, W) bool
    label: str = ""
    conf: float = 1.0
    colour: str | None = None


@dataclass
class Outline:
    """What one outline says, in robot-base terms."""
    view: View
    cam: np.ndarray               # camera position (3,)
    bearing: float                # direction of the outline's middle, seen from above [rad]
    half_angle: float             # half its angular width, seen from above [rad]
    elev_top: float               # elevation of its highest point [rad]; negative = below camera level
    elev_low: float               # elevation of its lowest point [rad]
    ray: np.ndarray               # unit line of sight through the outline's centroid
    clipped: set = field(default_factory=set)   # image borders the outline touches
    px: float = 0.002             # angular size of one pixel [rad]
    edge_lo: float = 0.0          # direction of the outline's two side edges, seen from above [rad]
    edge_hi: float = 0.0
    lo_cut: bool = False          # that side edge is cut off at the picture border (not the real edge)
    hi_cut: bool = False

    @property
    def side_cut(self) -> bool:
        return self.lo_cut or self.hi_cut


@dataclass
class KnownObject:
    name: str
    labels: list
    height: float
    width: float                  # widest width (for a cup: the rim)
    base_width: float | None = None


@dataclass
class Measurement:
    centre: np.ndarray            # (2,) x, y of the object's middle
    height: float                 # above the table
    width: float                  # widest outline width (for a cup: the rim)
    width_min: float              # the narrower of the two views' widths
    table_z: float
    method: str                   # "two cameras" | "one camera"
    sigma_centre: float           # expected position error [m] (1 sigma, from the geometry)
    views: list = field(default_factory=list)    # per camera: name, width, height, range
    notes: list = field(default_factory=list)
    known: str | None = None      # name of the known object whose size was used
    partner_label: str = ""       # what the other camera called it

    def points(self, n_ring: int = 36, n_levels: int = 9) -> np.ndarray:
        """A cylinder of the measured size at the measured place (for code that wants 3D points)."""
        r = max(self.width, 0.004) / 2
        ang = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
        ring = np.stack([np.cos(ang), np.sin(ang)], axis=1) * r
        pts = []
        for z in np.linspace(0.0, self.height, n_levels):
            pts.append(np.column_stack([ring + self.centre, np.full(n_ring, self.table_z + z)]))
        for f in (0.33, 0.66):                                   # the top face
            pts.append(np.column_stack([ring * f + self.centre, np.full(n_ring, self.table_z + self.height)]))
        return np.concatenate(pts)

    def describe(self) -> str:
        s = (f"{self.method}: stands at ({self.centre[0]:.3f}, {self.centre[1]:.3f}), "
             f"{self.height * 100:.1f} cm tall, {self.width * 100:.1f} cm wide "
             f"(position good to about {self.sigma_centre * 1000:.0f} mm)")
        if self.known:
            s += f"; size taken from the known object '{self.known}'"
        if self.notes:
            s += "; " + "; ".join(self.notes)
        return s


# ----------------------------------------------------------------------------- one outline


def _largest_blob(mask: np.ndarray) -> np.ndarray:
    """Keep the biggest connected piece (detector masks sometimes carry stray specks)."""
    try:
        import cv2
    except ImportError:                                           # pragma: no cover
        return mask
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 2:
        return mask
    return lab == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))


def _extent(x: np.ndarray, skip: int = 2) -> tuple[float, float]:
    k = min(skip, max(len(x) // 20, 0))
    lo = np.partition(x, k)[k]
    hi = np.partition(x, len(x) - 1 - k)[len(x) - 1 - k]
    return float(lo), float(hi)


def outline(view: View, min_pixels: int = 30) -> Outline | None:
    """Reduce a mask to the few angles the geometry needs. None if the view cannot be used."""
    mask = _largest_blob(np.asarray(view.mask, dtype=bool))
    vs, us = np.nonzero(mask)
    if len(us) < min_pixels:
        return None
    h, w = mask.shape
    clipped = set()
    if us.min() <= 1:
        clipped.add("left")
    if us.max() >= w - 2:
        clipped.add("right")
    if vs.min() <= 1:
        clipped.add("top")
    if vs.max() >= h - 2:
        clipped.add("bottom")
    K, T = np.asarray(view.K, dtype=float), np.asarray(view.T, dtype=float)
    d_cam = np.stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones(len(us))], axis=1)
    d = d_cam @ T[:3, :3].T
    centroid = d.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    flat = np.hypot(d[:, 0], d[:, 1])
    elev = np.arctan2(d[:, 2], flat)
    if abs(math.atan2(centroid[2], math.hypot(centroid[0], centroid[1]))) > STEEP_RAD:
        return None                                               # a top view of this object
    b0 = math.atan2(centroid[1], centroid[0])
    rel = np.arctan2(d[:, 1], d[:, 0]) - b0
    rel = (rel + np.pi) % (2 * np.pi) - np.pi
    # The outermost pixels, ignoring the two most extreme on each side. (A percentile would cut
    # too much off a shape that is only wide at one end, such as a cup's rim.)
    lo, hi = _extent(rel)
    e_lo, e_hi = _extent(elev)
    # which side edge (if any) is really the picture border: look at where the border pixels point
    lo_cut = hi_cut = False
    for border in (us <= 1, us >= w - 2):
        if border.any():
            if float(rel[border].mean()) < (lo + hi) / 2:
                lo_cut = True
            else:
                hi_cut = True
    return Outline(view=view, cam=T[:3, 3].copy(), bearing=b0 + (lo + hi) / 2, half_angle=(hi - lo) / 2,
                   elev_top=float(e_hi), elev_low=float(e_lo), ray=centroid, clipped=clipped,
                   px=1.0 / float(K[0, 0]), edge_lo=b0 + lo, edge_hi=b0 + hi, lo_cut=lo_cut, hi_cut=hi_cut)


def why_unusable(view: View, min_pixels: int = 30) -> str:
    """Why an outline cannot be measured: 'small', 'top view' or 'cut off'."""
    mask = np.asarray(view.mask, dtype=bool)
    if int(mask.sum()) < min_pixels:
        return "small"
    return "top view" if outline(view, min_pixels) is None else "cut off"


def _unit(b: float) -> np.ndarray:
    return np.array([math.cos(b), math.sin(b)])


def _range_from_table(o: Outline, table_z: float, base_width: float | None = None):
    """One camera: distance to the object's middle from where its outline meets the table.

    Returns (range to the middle, sigma, width, range to the nearest base point), or None when
    the camera cannot see the foot of the object.
    """
    cam_h = float(o.cam[2]) - table_z
    if cam_h < 0.01 or o.elev_low > -math.radians(1.0) or "bottom" in o.clipped:
        return None
    near = cam_h / math.tan(-o.elev_low)                          # nearest point of the base
    sigma = 0.012 + cam_h / math.sin(o.elev_low) ** 2 * PIX_NOISE * o.px
    if o.side_cut:
        # Part of the object is outside the picture: its own width is unknown, and the lowest
        # visible point may not be the nearest one. Usable only with a width from elsewhere.
        if not base_width:
            return None
        return near + base_width / 2, 2 * sigma + 0.01, None, near
    s = math.sin(o.half_angle)
    if not 0 < s < 0.9:
        return None
    width = 2 * near * s / (1 - s)                                # exact for a round object
    rho = near + (base_width if base_width else width) / 2
    return rho, sigma, width, near


def _height_from_top(o: Outline, rho: float, half_depth: float, table_z: float):
    """Height from the highest outline point, given the distance to the object's middle.

    A camera above the top sees the FAR top edge as the highest point; a camera below the top
    sees the NEAR top edge. Returns (height, sigma) or None.
    """
    if "top" in o.clipped:
        return None
    rho_top = rho + half_depth if o.elev_top < 0 else rho - half_depth
    t = math.tan(o.elev_top)
    height = float(o.cam[2]) + rho_top * t - table_z
    sigma = 0.003 + abs(t) * 0.012 + rho_top * (1 + t * t) * PIX_NOISE * o.px
    return height, sigma


def _ray_gap(a: Outline, b: Outline) -> float:
    """Closest distance between the two lines of sight through the outlines' centroids."""
    n = np.cross(a.ray, b.ray)
    ln = np.linalg.norm(n)
    diff = b.cam - a.cam
    if ln < 1e-6:
        return float(np.linalg.norm(np.cross(diff, a.ray)))
    return float(abs(diff @ n) / ln)


# ----------------------------------------------------------------------------- measuring


def measure_one(o: Outline, table_z: float, base_width: float | None = None,
                width: float | None = None) -> Measurement | None:
    """Position, height and width from a single camera (less exact than two)."""
    r = _range_from_table(o, table_z, base_width)
    if r is None or o.side_cut:
        return None
    rho, sigma, w_est, _ = r
    w = width or w_est
    centre = o.cam[:2] + rho * _unit(o.bearing)
    ht = _height_from_top(o, rho, w / 2, table_z)
    if ht is None:
        return None
    m = Measurement(centre=centre, height=ht[0], width=w, width_min=w, table_z=table_z,
                    method="one camera", sigma_centre=sigma,
                    views=[{"camera": o.view.name, "width": w_est, "height": ht[0], "range": rho}])
    return m if _plausible(m) else None


def measure_two(a: Outline, b: Outline, table_z: float, base_width: float | None = None):
    """Position, height and width from two cameras. Returns (Measurement, mismatch score) or None.

    The score says how badly the two outlines fit one object (metres; small is good). It is
    what pairs an object in one picture with the same object in the other.

    An object partly outside one picture still counts: the side edge that IS visible touches
    the object, so its middle lies half a width inside that edge (the width comes from the
    other camera).
    """
    outs = (a, b)
    if a.side_cut and b.side_cut:
        return None                               # no camera sees the whole width
    guess = _cross(a, b)
    # first guess of the width, refined below once the distances are known
    widths = [None, None]
    for i, o in enumerate(outs):
        r0 = _range_from_table(o, table_z)
        if not o.side_cut:
            widths[i] = r0[2] if r0 is not None else 0.06
    centre = cov = None
    ranges = [None, None]
    for _ in range(3):
        rows, rhs, sig = [], [], []
        for i, o in enumerate(outs):
            other_w = widths[1 - i]
            dist = (float(np.linalg.norm(centre - o.cam[:2])) if centre is not None else
                    float(np.linalg.norm(guess - o.cam[:2])) if guess is not None else 0.7)
            if not o.side_cut:                    # the middle of the outline points at the object's middle
                n = np.array([-math.sin(o.bearing), math.cos(o.bearing)])
                rows.append(n)
                rhs.append(n @ o.cam[:2])
                sig.append(max(0.002, dist * PIX_NOISE * o.px))
            elif not (o.lo_cut and o.hi_cut) and other_w:
                beta = o.edge_hi if o.lo_cut else o.edge_lo           # the edge that is really the object's
                n = np.array([-math.sin(beta), math.cos(beta)])       # points towards larger directions
                rows.append(n)
                rhs.append(n @ o.cam[:2] + (-1.0 if o.lo_cut else 1.0) * other_w / 2)
                sig.append(max(0.004, 2 * dist * PIX_NOISE * o.px))
            ranges[i] = _range_from_table(o, table_z, base_width or (other_w if o.side_cut else None))
            if ranges[i] is not None:
                u = _unit(o.bearing if not o.side_cut else math.atan2(o.ray[1], o.ray[0]))
                rows.append(u)
                rhs.append(u @ o.cam[:2] + ranges[i][0])
                sig.append(ranges[i][1])
        if len(rows) < 2:
            return None
        A, y, wgt = np.array(rows), np.array(rhs), 1.0 / np.array(sig)
        info = (A * wgt[:, None]).T @ (A * wgt[:, None])
        if np.linalg.cond(info) > 1e8:
            return None
        cov = np.linalg.inv(info)
        centre = cov @ ((A * wgt[:, None]).T @ (y * wgt))
        rho = [float(np.linalg.norm(centre - o.cam[:2])) for o in outs]
        for i, o in enumerate(outs):
            if not o.side_cut:
                widths[i] = 2 * rho[i] * math.sin(o.half_angle)
    sigma_c = float(math.sqrt(max(np.linalg.eigvalsh(cov).max(), 0.0)))
    if min(rho) < 0.05:
        return None
    # the object must be in front of both cameras
    if any((centre - o.cam[:2]) @ o.ray[:2] <= 0 for o in outs):
        return None
    usable_w = [w for w in widths if w]
    full = [w if w else usable_w[0] for w in widths]              # a cut view borrows the other's width
    # how far the object reaches along each line of sight: the other camera's width when the
    # views are square to each other, its own when they look along the same line
    c = abs(math.cos(a.bearing - b.bearing))
    half_depth = [0.5 * ((1 - c) * full[1] + c * full[0]), 0.5 * ((1 - c) * full[0] + c * full[1])]
    hs = [_height_from_top(o, rh, hd, table_z) for o, rh, hd in zip(outs, rho, half_depth)]
    got = [h for h in hs if h is not None]
    if not got:
        return None
    hw = np.array([1.0 / h[1] ** 2 for h in got])
    height = float(np.array([h[0] for h in got]) @ hw / hw.sum())

    m = Measurement(centre=centre, height=height, width=max(usable_w), width_min=min(usable_w),
                    table_z=table_z, method="two cameras", sigma_centre=sigma_c,
                    views=[{"camera": o.view.name, "width": w, "height": (h[0] if h else None), "range": rh}
                           for o, w, h, rh in zip(outs, widths, hs, rho)],
                    partner_label=b.view.label)
    if not _plausible(m):
        return None
    # how well do the two outlines agree that this is ONE object?
    score = _ray_gap(a, b)
    for o, r, rh in zip(outs, ranges, rho):
        if r is not None:                       # each camera's own distance estimate vs. the joint one
            score += max(0.0, abs(rh - r[0]) - 2 * r[1])
    if len(got) == 2:
        dh = abs(got[0][0] - got[1][0])
        score += max(0.0, dh - 2 * (got[0][1] + got[1][1]))
        if dh > 0.025:
            m.notes.append(f"the two cameras disagree on the height by {dh * 100:.1f} cm "
                           f"(used {height * 100:.1f} cm, weighted towards the camera nearer table level)")
    cut = [o.view.name for o in outs if o.side_cut or o.clipped]
    if cut:
        m.notes.append("partly outside the picture of " + " and ".join(f"'{n}'" for n in cut))
    return m, score


def _cross(a: Outline, b: Outline):
    """Where the two 'middle of the outline' directions cross, seen from above (None if parallel)."""
    ua, ub = _unit(a.bearing), _unit(b.bearing)
    det = ua[0] * (-ub[1]) - (-ub[0]) * ua[1]
    if abs(det) < 0.2:                           # under ~12 degrees apart: no clean crossing
        return None
    diff = b.cam[:2] - a.cam[:2]
    t = (diff[0] * (-ub[1]) - (-ub[0]) * diff[1]) / det
    return a.cam[:2] + t * ua


def _plausible(m: Measurement) -> bool:
    return (0.005 < m.height < 0.60 and 0.005 < m.width < 0.45 and np.all(np.isfinite(m.centre)))


# ----------------------------------------------------------------------------- known sizes


def load_known(path) -> list:
    """Read the known-object table (YAML): see local_dm/known_objects_example.yaml."""
    import yaml
    data = yaml.safe_load(Path(path).read_text()) or {}
    out = []
    for e in data.get("objects", []):
        out.append(KnownObject(name=str(e["name"]),
                               labels=[str(s).lower() for s in e.get("labels", [e["name"]])],
                               height=float(e["height_m"]), width=float(e["width_m"]),
                               base_width=(float(e["base_width_m"]) if e.get("base_width_m") else None)))
    return out


def match_known(m: Measurement, labels, known: list) -> KnownObject | None:
    """The known object this measurement is, if any.

    Detector names are unreliable (a plain can is called "red cup"), so the name only narrows
    the choice; the measured size has to agree too. Tolerances: height within 3 cm or 30 %,
    width within 2.5 cm or 35 %.
    """
    labels = {str(s).lower() for s in labels if s}
    best, best_d = None, None
    for k in known:
        if labels and not (labels & set(k.labels)):
            continue
        dh, dw = abs(m.height - k.height), abs(m.width - k.width)
        if dh > max(0.03, 0.30 * k.height) or dw > max(0.025, 0.35 * k.width):
            continue
        d = dh / k.height + dw / k.width
        if best is None or d < best_d:
            best, best_d = k, d
    return best


# ----------------------------------------------------------------------------- top level


def measure(primary: View, others: list, table_z: float, known: list | None = None,
            max_mismatch: float = 0.06) -> Measurement | None:
    """Measure the object outlined in `primary`, using the best-fitting outline in `others`.

    `others` are all objects the second camera sees (any names). The partner is the one whose
    outline fits the same place in space; the same name or colour only breaks ties. With no
    fitting partner the single-camera estimate is returned, and None if that fails too.
    """
    a = outline(primary)
    if a is None:
        return None
    best = None
    for v in others:
        b = outline(v)
        if b is None:
            continue
        res = measure_two(a, b, table_z)
        if res is None:
            continue
        m, score = res
        if score > max_mismatch or m.sigma_centre > MAX_SIGMA:
            continue
        if v.label and primary.label and v.label != primary.label:
            score += 0.01                          # a different name: only a tie-breaker
        if v.colour and primary.colour and v.colour != primary.colour:
            score += 0.02
        if best is None or score < best[1]:
            best = (m, score, b)
    if best is not None:
        m, _, b = best
        k = match_known(m, [primary.label, b.view.label], known or [])
        if k is not None:
            _use_known(m, k)
        return m
    m = measure_one(a, table_z)
    if m is None:
        return None
    m.notes.append("only one camera saw it, so the position is less exact")
    k = match_known(m, [primary.label], known or [])
    if k is not None:
        # one camera places the middle from the object's foot: the known base width sharpens that
        m2 = measure_one(a, table_z, base_width=k.base_width or k.width, width=k.width)
        if m2 is not None:
            m2.notes = m.notes
            m = m2
        _use_known(m, k)
    return m


def _use_known(m: Measurement, k: KnownObject) -> None:
    m.notes.append(f"cameras measured {m.height * 100:.1f} x {m.width * 100:.1f} cm")
    m.height, m.width, m.known = k.height, k.width, k.name
    m.width_min = min(m.width_min, k.width)
