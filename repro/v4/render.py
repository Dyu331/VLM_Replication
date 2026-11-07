"""Render candidate v4 questions in one or more rooms.

For each layout: three objects of different categories (from the split's pool) at random free
spots, randomly turned. For each A photo of a layout: every object in turn is B; the other two are
X and Y in random order. A question is kept as a candidate when it passes the validity rules;
the balancing happens later (select.py).

    python repro/v4/render.py --set smoke --split train --rooms v3_sc0_staging_01 --photos 20
    python repro/v4/render.py --set train --split train --worker 0 --workers 4 --photos 140

Per room, $V4_DATA/<set>/<room>/:
  <photo>_A.png, <photo>_A_sem.png        A's photo (input) and its object mask (ids 1-3)
  <qid>_B.png, <qid>_B_sem.png            the view from B's spot, B hidden (direct-B question, labels)
  candidates.jsonl                        one row per valid question
  stats.json                              attempts and why candidates were rejected
"""
import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v4.config_v4 import (A_DIST, EDGE_PX, MAX_OCCLUDED, MIN_ANGLE_A, MIN_ANGLE_B, MIN_B_LINE, MIN_B_XY,
                          MIN_VIS_PCT, OBJ_MIN_GAP, RES, TEST_CATS, TEST_ROOMS_V4, TRAIN_CATS,
                          TRAIN_ROOMS_V4, V4_DATA)
from v4.scene import (Room, bearing, dist_to_line, floor_dist, grow, inside, mask_stats,
                      side_of_line, yaw_towards)

A_PER_LAYOUT = 2        # A photos per layout [ours]
A_TRIES = 80
LAYOUT_TRIES = 40


def view_problems(stats, names, view, edges=True):
    """Visibility rules for one view: big enough, not cut off at the border (A's photo only; their
    examples show cut-off objects from B's spot), no left-right overlap."""
    probs = []
    for n in names:
        s = stats[n]
        if s["vis_pct"] < MIN_VIS_PCT:
            probs.append(f"{view}: {n} too small")
        elif edges and (s["xmin"] < EDGE_PX or s["xmax"] > RES - 1 - EDGE_PX or s["ymin"] < EDGE_PX
              or s["ymax"] > RES - 1 - EDGE_PX):
            probs.append(f"{view}: {n} cut off")
    if probs:
        return probs
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sa, sb = stats[a], stats[b]
            if not (sa["xmax"] < sb["xmin"] or sb["xmax"] < sa["xmin"]):
                probs.append(f"{view}: overlap")
    return probs


def place_layout(room, rng, pool, assets_by_cat):
    cats = list(rng.choice(pool, 3, replace=False))
    placed = []
    for sid, cat in enumerate(cats, start=1):
        asset = assets_by_cat[cat][rng.integers(len(assets_by_cat[cat]))]
        for _ in range(50):
            p = np.array(room.pf.get_random_navigable_point())
            if any(floor_dist(p, o["centre"]) < OBJ_MIN_GAP for o in placed):
                continue
            o = room.add(asset, p[0], p[2], float(rng.uniform(0, 360)), sid, others=placed)
            if o is not None:
                o.update(category=cat, asset=asset["id"], yaw=None)
                placed.append(o)
                break
        else:
            for o in placed:
                room.remove(o)
            return None
    return placed


def sample_A(room, rng, objs, why):
    centroid = np.mean([o["centre"] for o in objs], axis=0)
    for _ in range(A_TRIES):
        p = np.array(room.pf.get_random_navigable_point())
        d = [floor_dist(p, o["centre"]) for o in objs]
        if not all(A_DIST[0] <= x <= A_DIST[1] for x in d):
            why["A: distance"] += 1
            continue
        if any(inside((p[0], p[2]), grow(o["foot"], 0.3)) for o in objs):
            why["A: inside an object"] += 1
            continue
        xz = (float(p[0]), float(p[2]))
        if not all(room.line_of_sight(xz, o, objs) for o in objs):
            why["A: view blocked by wall"] += 1
            continue
        yaw = yaw_towards(p, centroid)
        rgb, sem, depth = room.render(xz, yaw)
        stats = {o["sid"]: mask_stats(sem, o["sid"]) for o in objs}
        probs = view_problems(stats, [1, 2, 3], "A")
        if probs:
            for pr in set(probs):
                why[pr.replace(" 1", "").replace(" 2", "").replace(" 3", "")] += 1
            continue
        # share of each object hidden behind the other two: render it alone and compare
        hidden = {}
        for o in objs:
            rest = [p for p in objs if p is not o]
            for p in rest:
                room.hide(p)
            try:
                full = int((room.render(xz, yaw)[1] == o["sid"]).sum())
            finally:
                for p in rest:
                    room.show(p)
            hidden[o["sid"]] = 1 - stats[o["sid"]]["vis_pct"] / 100 * sem.size / max(full, 1)
        if max(hidden.values()) > MAX_OCCLUDED:
            why["A: object hidden behind another"] += 1
            continue
        for sid, h in hidden.items():
            stats[sid]["hidden"] = round(max(h, 0.0), 3)
        return xz, yaw, rgb, sem, depth, stats
    return None


def rank_word(r):
    return ["nearest", "middle", "farthest"][r]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, help="output folder under $V4_DATA (smoke / train / test)")
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--rooms", nargs="*", help="default: all rooms of the split")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--photos", type=int, default=20, help="A photos per room")
    ap.add_argument("--debug", action="store_true", help="also save A's raw render")
    args = ap.parse_args()

    rooms = args.rooms or (TRAIN_ROOMS_V4 if args.split == "train" else TEST_ROOMS_V4)
    rooms = rooms[args.worker::args.workers]
    pool = TRAIN_CATS if args.split == "train" else TEST_CATS
    assets = json.loads((V4_DATA / "assets.json").read_text())
    by_cat = {c: [a for a in assets if a["category"] == c] for c in pool}

    for scene in rooms:
        t0 = time.time()
        seed = int(hashlib.md5(f"{scene}-{args.split}-{args.set}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        room = Room(scene, seed=seed % 100000)
        out = V4_DATA / args.set / scene
        out.mkdir(parents=True, exist_ok=True)
        why, n_photos, n_q, layout_id = Counter(), 0, 0, 0
        with open(out / "candidates.jsonl", "w") as f:
            for _ in range(args.photos * 20):
                if n_photos >= args.photos:
                    break
                objs = place_layout(room, rng, pool, by_cat)
                why["layouts tried"] += 1
                if objs is None:
                    why["layout: objects do not fit"] += 1
                    continue
                layout_id += 1
                for _a in range(A_PER_LAYOUT):
                    if n_photos >= args.photos:
                        break
                    a = sample_A(room, rng, objs, why)
                    if a is None:
                        break
                    a_xz, a_yaw, rgb, sem, depth, a_stats = a
                    photo = f"{scene}_L{layout_id:03d}_P{n_photos:03d}"
                    img = room.composite(rgb, sem, depth, a_xz, a_yaw, keep={1, 2, 3})
                    sem_save = np.where(np.isin(sem, [1, 2, 3]), sem, 0).astype(np.uint8)
                    cv2.imwrite(str(out / f"{photo}_A.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(out / f"{photo}_A_sem.png"), sem_save)
                    if args.debug:
                        cv2.imwrite(str(out / f"{photo}_A_raw.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    n_photos += 1
                    a_pos = [a_xz[0], room.floor_y, a_xz[1]]
                    dA = [floor_dist(a_pos, o["centre"]) for o in objs]
                    ranks = np.argsort(np.argsort(dA))
                    for bi in range(3):
                        why["questions tried"] += 1
                        xi, yi = [i for i in range(3) if i != bi]
                        if rng.random() < 0.5:
                            xi, yi = yi, xi
                        B, X, Y = objs[bi], objs[xi], objs[yi]
                        bx, xx, yx = (bearing(a_pos, a_yaw, o["centre"]) for o in (B, X, Y))
                        angle_A = abs(xx - yx)
                        b_line = dist_to_line(B["centre"], X["centre"], Y["centre"])
                        b_x, b_y = floor_dist(B["centre"], X["centre"]), floor_dist(B["centre"], Y["centre"])
                        probs = []
                        if angle_A < MIN_ANGLE_A:
                            probs.append("A: X and Y < 10 deg apart")
                        if b_line < MIN_B_LINE:
                            probs.append("B < 0.5 m from the X-Y line")
                        if min(b_x, b_y) < MIN_B_XY:
                            probs.append("B < 0.8 m from X or Y")
                        if probs:
                            for p in probs:
                                why[p] += 1
                            continue
                        b_xz = (B["centre"][0], B["centre"][2])
                        mid = (np.array(X["centre"]) + np.array(Y["centre"])) / 2
                        b_yaw = yaw_towards(B["centre"], mid)
                        room.hide(B)
                        try:
                            if not all(room.line_of_sight(b_xz, o, [X, Y]) for o in (X, Y)):
                                why["B: view blocked by wall"] += 1
                                continue
                            rgb_b, sem_b, depth_b = room.render(b_xz, b_yaw)
                        finally:
                            room.show(B)
                        b_stats = {o["sid"]: mask_stats(sem_b, o["sid"]) for o in (X, Y)}
                        probs = view_problems(b_stats, [X["sid"], Y["sid"]], "B", edges=False)
                        bxx, byx = (bearing(B["centre"], b_yaw, o["centre"]) for o in (X, Y))
                        angle_B = abs(bxx - byx)
                        if angle_B < MIN_ANGLE_B:
                            probs.append("B: X and Y < 15 deg apart")
                        if probs:
                            for p in set(probs):
                                why[p] += 1
                            continue
                        sX, sY, sB = a_stats[X["sid"]], a_stats[Y["sid"]], a_stats[B["sid"]]
                        relA = "left" if sX["cx"] < sY["cx"] else "right"
                        relB = "left" if b_stats[X["sid"]]["cx"] < b_stats[Y["sid"]]["cx"] else "right"
                        relA_geo = "left" if xx < yx else "right"
                        relB_geo = "left" if bxx < byx else "right"
                        flip_geo = side_of_line(a_pos, X["centre"], Y["centre"]) != side_of_line(B["centre"], X["centre"], Y["centre"])
                        if relA != relA_geo or relB != relB_geo:
                            why["mask label != geometry label"] += 1
                            continue
                        if (relA != relB) != flip_geo:
                            why["flip != side of line"] += 1
                            continue
                        side = ("left" if sB["cx"] < min(sX["cx"], sY["cx"]) else
                                "right" if sB["cx"] > max(sX["cx"], sY["cx"]) else "between")
                        qid = f"{photo}_B{bi}"
                        img_b = room.composite(rgb_b, sem_b, depth_b, b_xz, b_yaw, keep={X["sid"], Y["sid"]})
                        cv2.imwrite(str(out / f"{qid}_B.png"), cv2.cvtColor(img_b, cv2.COLOR_RGB2BGR))
                        sem_b_save = np.where(np.isin(sem_b, [X["sid"], Y["sid"]]), sem_b, 0).astype(np.uint8)
                        cv2.imwrite(str(out / f"{qid}_B_sem.png"), sem_b_save)
                        rec = {
                            "qid": qid, "photo": photo, "room": scene, "split": args.split, "set": args.set,
                            "layout": layout_id,
                            "B": B["category"], "X": X["category"], "Y": Y["category"],
                            "sid": {"B": B["sid"], "X": X["sid"], "Y": Y["sid"]},
                            "objects": [{"sid": o["sid"], "category": o["category"], "asset": o["asset"],
                                         "centre": [round(v, 3) for v in o["centre"]],
                                         "foot": np.round(o["foot"], 3).tolist(),
                                         "height": round(o["height"], 3)} for o in objs],
                            "A_pos": [round(v, 3) for v in a_pos], "A_yaw": round(a_yaw, 2),
                            "B_pos": [round(v, 3) for v in B["centre"]], "B_yaw": round(b_yaw, 2),
                            "A_view": {"B": sB, "X": sX, "Y": sY,
                                       "bearing": {"B": round(bx, 2), "X": round(xx, 2), "Y": round(yx, 2)}},
                            "B_view": {"X": b_stats[X["sid"]], "Y": b_stats[Y["sid"]],
                                       "bearing": {"X": round(bxx, 2), "Y": round(byx, 2)}},
                            "dist": {"AB": round(dA[bi], 3), "AX": round(dA[xi], 3), "AY": round(dA[yi], 3),
                                     "BX": round(b_x, 3), "BY": round(b_y, 3),
                                     "XY": round(floor_dist(X["centre"], Y["centre"]), 3),
                                     "B_line": round(b_line, 3)},
                            "angle_A": round(angle_A, 2), "angle_B": round(angle_B, 2),
                            "relA": relA, "relB": relB, "flip": relA != relB,
                            "depth_rank": rank_word(int(ranks[bi])), "image_side": side,
                            "img_A": f"{scene}/{photo}_A.png", "sem_A": f"{scene}/{photo}_A_sem.png",
                            "img_B": f"{scene}/{qid}_B.png", "sem_B": f"{scene}/{qid}_B_sem.png",
                        }
                        f.write(json.dumps(rec) + "\n")
                        n_q += 1
                        why["valid questions"] += 1
                for o in objs:
                    room.remove(o)
        stats = {"room": scene, "photos": n_photos, "questions": n_q, "layouts": layout_id,
                 "seconds": round(time.time() - t0, 1), "floor_y": room.floor_y,
                 "navigable_area": round(room.pf.navigable_area, 2), "counts": dict(why)}
        (out / "stats.json").write_text(json.dumps(stats, indent=1))
        print(f"{scene}: {n_photos} photos, {n_q} questions in {stats['seconds']} s | {dict(why)}", flush=True)
        room.close()


if __name__ == "__main__":
    main()
