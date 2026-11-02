"""Mask-based labels, the clean rule, and the centroid-vs-mask audit (report Section 1).

Reads every $PP_DATA/<set>/<scene>/examples.jsonl, writes $PP_DATA/<set>/examples_labeled.jsonl
with, per example:
  relA / relB                    mask label: X left of Y iff X's visible-pixel centre is left of Y's
  relA_centroid / relB_centroid  3D-centre bearing label (kept for the audit)
  sentence_B                     plain sentence + landmark clauses, as in the report (the plain
                                 one is kept as sentence_B_base):
      "B stands next to the <obj>."  another vocab object (not X/Y) within NEXT_TO m of B
      "B is facing the <obj>."       a vocab object within FACING_DEG of B's view centre
  clean, reject_reasons          clean rule, applied to both views:
      - X and Y masks do not overlap horizontally
      - mask centres >= MIN_PX_GAP px apart
      - observer >= MIN_OBS_DIST m from both objects' 3D centres
      - cross_line only: the mask labels for A and B differ
and prints the Section-1 summary.

    python repro/render/label_and_clean.py --set test
"""
import argparse
import json
import math
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import MIN_OBS_DIST, MIN_PX_GAP, PP_DATA
from common.vocab import VOCAB

NEXT_TO = 1.0      # metres from B to the object's footprint
FACING_DEG = 10.0  # degrees between B's view direction and the object's centre
FACING_MAX = 6.0   # metres; farther objects are not "faced"


@lru_cache(maxsize=None)
def vocab_footprints(scene):
    """{name: (xmin, xmax, zmin, zmax, cx, cz)} for vocab objects unique in the scene."""
    objs = json.loads((PP_DATA / "scene_index" / f"{scene}.json").read_text())["objects"]
    counts = Counter(o["template"] for o in objs)
    out = {}
    for o in objs:
        name = VOCAB.get(o["template"])
        if name and counts[o["template"]] == 1:
            (x0, _, z0), (x1, _, z1) = o["aabb_min"], o["aabb_max"]
            out[name] = (x0, x1, z0, z1, (x0 + x1) / 2, (z0 + z1) / 2)
    return out


def landmark_clauses(rec):
    """Report-style "next to" / "facing" clauses for B, from B's saved pose."""
    (bx, _, bz), yaw = rec["B"]["pos"], rec["B"]["yaw"]
    objs = vocab_footprints(rec["scene"])
    clauses = []
    near = []
    for name, (x0, x1, z0, z1, _, _) in objs.items():
        if name in (rec["X"], rec["Y"]):
            continue
        d = math.hypot(max(x0 - bx, 0, bx - x1), max(z0 - bz, 0, bz - z1))
        if d <= NEXT_TO:
            near.append((d, name))
    if near:
        clauses.append(f"B stands next to the {min(near)[1]}.")
    fwd = (-math.sin(math.radians(yaw)), -math.cos(math.radians(yaw)))
    faced = []
    for name, (*_, cx, cz) in objs.items():
        dx, dz = cx - bx, cz - bz
        dist = math.hypot(dx, dz)
        if 0 < dist <= FACING_MAX:
            ang = math.degrees(math.acos(max(-1, min(1, (dx * fwd[0] + dz * fwd[1]) / dist))))
            if ang <= FACING_DEG:
                faced.append((ang, name))
    if faced:
        clauses.append(f"B is facing the {min(faced)[1]}.")
    return clauses


def mask_label(view):
    return "left" if view["X"]["cx"] < view["Y"]["cx"] else "right"


def view_problems(view, name):
    x, y = view["X"], view["Y"]
    probs = []
    if not (x["xmax"] < y["xmin"] or y["xmax"] < x["xmin"]):
        probs.append(f"the objects overlap in {name}'s photo")
    gap = abs(x["cx"] - y["cx"])
    if gap < MIN_PX_GAP:
        probs.append(f"their centres are only {gap:.0f} px apart for {name}")
    near = min(x["dist"], y["dist"])
    if near < MIN_OBS_DIST:
        probs.append(f"{name} stands {near:.1f} m from an object")
    return probs


def label(rec):
    rec["sentence_B_base"] = rec.get("sentence_B_base", rec["sentence_B"])
    rec["sentence_B"] = " ".join([rec["sentence_B_base"]] + landmark_clauses(rec))
    rec["relA"], rec["relB"] = mask_label(rec["A"]), mask_label(rec["B"])
    rec["relA_centroid"], rec["relB_centroid"] = rec["A"]["rel_centroid"], rec["B"]["rel_centroid"]
    reasons = view_problems(rec["A"], "A") + view_problems(rec["B"], "B")
    if rec["mode"] == "cross_line" and rec["relA"] == rec["relB"]:
        reasons.append("the answer does not differ for A and B")
    rec["reject_reasons"] = reasons
    rec["clean"] = not reasons
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    args = ap.parse_args()

    root = PP_DATA / args.set
    recs = []
    for f in sorted(root.glob("*/examples.jsonl")):
        recs += [label(json.loads(l)) for l in open(f) if l.strip()]
    with open(root / "examples_labeled.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    n = len(recs)
    clean = [r for r in recs if r["clean"]]
    print(f"set {args.set}: {n} rendered in {len({r['scene'] for r in recs})} rooms, "
          f"{len(clean)} clean ({100 * len(clean) / max(n, 1):.0f}%)")

    # why examples were rejected (an example can have several reasons)
    kinds = {"overlap": "overlap", "px apart": "centres too close",
             "stands": "observer < 1.5 m", "does not differ": "no flip"}
    for key, label_ in kinds.items():
        k = sum(any(key in s for s in r["reject_reasons"]) for r in recs)
        print(f"  rejected for {label_:18s}: {k}")

    # audit: centroid label vs mask label, per view
    for v in "AB":
        dis = [r for r in recs if r[f"rel{v}"] != r[f"rel{v}_centroid"]]
        msg = f"  {v} views where centroid label != mask label: {len(dis)} of {n}"
        if dis:
            near = [min(r[v]["X"]["dist"], r[v]["Y"]["dist"]) for r in dis]
            msg += f" (median distance to nearer object {np.median(near):.2f} m)"
        print(msg)
    flips = sum(r["relA"] != r["relB"] for r in recs)
    print(f"  examples whose mask labels flip between A and B: {flips} of {n}")
    nxt = sum("next to" in r["sentence_B"] for r in clean)
    fac = sum("is facing" in r["sentence_B"] for r in clean)
    print(f"  clean sentences with 'next to': {nxt}, with 'facing': {fac} (of {len(clean)})")
    left = sum(r["relB"] == "left" for r in clean)
    print(f"  clean examples with B's answer = left: {left} of {len(clean)}")
    print(f"-> {root / 'examples_labeled.jsonl'}")


if __name__ == "__main__":
    main()
