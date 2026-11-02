"""Sample and render cross_line examples for one scene.

For each example: A stands on the navmesh and sees objects X and Y; B stands on the far
side of the X-Y line and looks back at them, so left/right should flip. Both views are
rendered (RGB + instance mask), and per-view mask statistics and 3D-centre labels are
saved. Mask-based labels and the clean rule are applied later (relabel_visible.py,
clean_rule.py) from the saved stats; nothing is filtered here except the sampling
constraints (visibility, bearing margin, B on the far side).

    python repro/render/render_examples.py --set test --scene apt_0 --n 14 --seed 0

Output: $PP_DATA/<set>/<scene>/{examples.jsonl, <id>_A_rgb.png, <id>_A_sem.png, ...}
"""
import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import B_DIST, EYE, MIN_MARGIN, MIN_VIS, PP_DATA, YAW_JITTER
from common.sim import assign_semantic_ids, make_sim, set_pose, template_name, world_aabb
from common.vocab import EXCLUDED_PAIRS, VOCAB


# ---------- geometry (habitat: yaw 0 faces -Z, positive yaw turns left) ----------

def forward(yaw):
    r = math.radians(yaw)
    return np.array([-math.sin(r), 0.0, -math.cos(r)])


def right(yaw):
    r = math.radians(yaw)
    return np.array([math.cos(r), 0.0, -math.sin(r)])


def yaw_towards(src, dst):
    d = np.asarray(dst) - np.asarray(src)
    return math.degrees(math.atan2(-d[0], -d[2]))


def bearing(pos, yaw, target):
    """Angle of target from the view direction, degrees; positive = to the right."""
    d = np.asarray(target) - np.asarray(pos)
    return math.degrees(math.atan2(d @ right(yaw), d @ forward(yaw)))


def wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0


def side_of_line(p, a, b):
    """Sign of p relative to the X-Y line on the floor plane (x, z)."""
    return np.sign((b[0] - a[0]) * (p[2] - a[2]) - (b[2] - a[2]) * (p[0] - a[0]))


def floor_dist(p, q):
    return float(np.hypot(p[0] - q[0], p[2] - q[2]))


# ---------- scene objects ----------

def vocab_objects(sim):
    """{name: {"sid", "center"}} for vocab templates that occur exactly once in the scene."""
    id_to_obj = assign_semantic_ids(sim)
    by_template = {}
    for sid, obj in id_to_obj.items():
        by_template.setdefault(template_name(obj.handle), []).append((sid, obj))
    objs = {}
    for template, name in VOCAB.items():
        if len(by_template.get(template, [])) != 1:
            continue
        sid, obj = by_template[template][0]
        lo, hi = world_aabb(obj)
        objs[name] = {"sid": sid, "center": ((lo + hi) / 2).tolist(), "handle": obj.handle}
    return objs


# ---------- rendering + per-view stats ----------

def render(sim, pos, yaw):
    set_pose(sim, pos, yaw)
    obs = sim.get_sensor_observations()
    return obs["rgb"][..., :3], obs["semantic"].astype(np.int32)


def mask_stats(sem, sid):
    m = sem == sid
    n = int(m.sum())
    if n == 0:
        return {"vis_pct": 0.0, "cx": None, "xmin": None, "xmax": None}
    xs = np.nonzero(m)[1]
    return {"vis_pct": round(100.0 * n / m.size, 3), "cx": round(float(xs.mean()), 1),
            "xmin": int(xs.min()), "xmax": int(xs.max())}


def view_ok(sem, pos, yaw, X, Y):
    """Sampling constraint for one view: both objects visible enough, bearings apart."""
    sx, sy = mask_stats(sem, X["sid"]), mask_stats(sem, Y["sid"])
    if sx["vis_pct"] < MIN_VIS or sy["vis_pct"] < MIN_VIS:
        return False
    return abs(bearing(pos, yaw, X["center"]) - bearing(pos, yaw, Y["center"])) >= MIN_MARGIN


def view_record(sem, pos, yaw, X, Y):
    bx, by = bearing(pos, yaw, X["center"]), bearing(pos, yaw, Y["center"])
    return {
        "pos": [round(float(v), 4) for v in pos], "yaw": round(yaw, 2),
        "X": {**mask_stats(sem, X["sid"]), "bearing": round(bx, 2),
              "dist": round(floor_dist(pos, X["center"]), 3)},
        "Y": {**mask_stats(sem, Y["sid"]), "bearing": round(by, 2),
              "dist": round(floor_dist(pos, Y["center"]), 3)},
        "rel_centroid": "left" if bx < by else "right",
    }


# ---------- sentences ----------

def describe(a_pos, a_yaw, b_pos, b_yaw):
    """Sentence telling A where B stands and looks, in A's egocentric frame."""
    d = np.asarray(b_pos) - np.asarray(a_pos)
    dist = round(float(np.hypot(d[0], d[2])) * 2) / 2
    ang = math.degrees(math.atan2(d @ right(a_yaw), d @ forward(a_yaw)))  # + = right
    where = "ahead of you" if abs(ang) <= 90 else "behind you"
    if 15 < abs(ang) < 165:
        where += " and to your right" if ang > 0 else " and to your left"
    turn = round(wrap(b_yaw - a_yaw) / 15) * 15  # + = turned left
    if turn == 0:
        facing = "facing the same direction as you"
    elif abs(turn) == 180:
        facing = "turned about 180 degrees compared with your heading"
    else:
        facing = (f"turned about {abs(turn)} degrees to the {'left' if turn > 0 else 'right'} "
                  f"compared with your heading")
    return f"Observer B is standing about {dist:.1f} metres {where}, {facing}."


IDENTITY_SENTENCE = "Observer B is standing where you are standing, facing the same direction as you."


# ---------- sampling ----------

def sample_A(sim, rng, X, Y, tries=200):
    mid = (np.array(X["center"]) + np.array(Y["center"])) / 2
    for _ in range(tries):
        pos = np.array(sim.pathfinder.get_random_navigable_point())
        if min(floor_dist(pos, X["center"]), floor_dist(pos, Y["center"])) < 1.0:
            continue
        yaw = yaw_towards(pos, mid) + rng.uniform(-YAW_JITTER, YAW_JITTER)
        _, sem = render(sim, pos, yaw)
        if view_ok(sem, pos, yaw, X, Y):
            return pos, yaw, sem
    return None


def sample_B_cross(sim, rng, a_pos, X, Y, tries=200):
    xc, yc = np.array(X["center"]), np.array(Y["center"])
    mid = (xc + yc) / 2
    line = (yc - xc) * np.array([1, 0, 1])
    line /= np.linalg.norm(line) + 1e-9
    normal = np.array([-line[2], 0.0, line[0]])
    a_side = side_of_line(a_pos, xc, yc)
    if side_of_line(mid + normal, xc, yc) == a_side:
        normal = -normal  # point to the side opposite A
    for _ in range(tries):
        t = rng.uniform(-0.5, 1.5)  # along the line, a bit past either end
        cand = xc + t * (yc - xc) + rng.uniform(*B_DIST) * normal
        cand[1] = a_pos[1]
        pos = np.array(sim.pathfinder.snap_point(cand))
        if not np.all(np.isfinite(pos)) or side_of_line(pos, xc, yc) != -a_side:
            continue
        yaw = yaw_towards(pos, mid) + rng.uniform(-YAW_JITTER, YAW_JITTER)
        _, sem = render(sim, pos, yaw)
        if view_ok(sem, pos, yaw, X, Y):
            return pos, yaw, sem
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, help="output subfolder, e.g. test / train / smoke")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--n", type=int, default=14, help="examples to render for this scene")
    ap.add_argument("--per-pair", type=int, default=3, help="max examples per object pair")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    sim = make_sim(args.scene, semantic=True)
    sim.pathfinder.seed(args.seed)
    objs = vocab_objects(sim)
    pairs = [(a, b) for a, b in itertools.permutations(sorted(objs), 2)
             if frozenset({a, b}) not in EXCLUDED_PAIRS]
    print(f"{args.scene}: objects {sorted(objs)}; {len(pairs)} ordered pairs", flush=True)

    out = PP_DATA / args.set / args.scene
    out.mkdir(parents=True, exist_ok=True)
    used = {}
    n_done, attempts = 0, 0
    with open(out / "examples.jsonl", "w") as f:
        while n_done < args.n and attempts < 20 * args.n:
            attempts += 1
            xn, yn = pairs[rng.integers(len(pairs))]
            if used.get(frozenset({xn, yn}), 0) >= args.per_pair:
                continue
            X, Y = objs[xn], objs[yn]
            a = sample_A(sim, rng, X, Y)
            if a is None:
                continue
            a_pos, a_yaw, _ = a
            b = sample_B_cross(sim, rng, a_pos, X, Y)
            if b is None:
                continue
            b_pos, b_yaw, _ = b

            ex_id = f"{args.scene}_cross_line_s{args.seed}_{n_done:02d}"
            rec = {"id": ex_id, "scene": args.scene, "mode": "cross_line", "set": args.set,
                   "X": xn, "Y": yn, "X_handle": X["handle"], "Y_handle": Y["handle"],
                   "X_center": X["center"], "Y_center": Y["center"], "eye": EYE,
                   "sentence_B": describe(a_pos, a_yaw, b_pos, b_yaw),
                   "sentence_identity": IDENTITY_SENTENCE}
            for view, (pos, yaw) in {"A": (a_pos, a_yaw), "B": (b_pos, b_yaw)}.items():
                rgb, sem = render(sim, pos, yaw)
                cv2.imwrite(str(out / f"{ex_id}_{view}_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(out / f"{ex_id}_{view}_sem.png"), sem.astype(np.uint16))
                rec[view] = view_record(sem, pos, yaw, X, Y)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            used[frozenset({xn, yn})] = used.get(frozenset({xn, yn}), 0) + 1
            n_done += 1
            print(f"  {ex_id}: {xn} vs {yn}  A {rec['A']['rel_centroid']}  "
                  f"B {rec['B']['rel_centroid']}  | {rec['sentence_B']}", flush=True)
    print(f"{args.scene}: {n_done} examples in {attempts} attempts -> {out}")
    sim.close()


if __name__ == "__main__":
    main()
