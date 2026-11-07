"""Balance the candidate questions and run the data checks (PLAN_V4.md Phase 2).

    python repro/v4/balance.py --train-set train --test-set test
    python repro/v4/balance.py --train-set smoke --test-set smoke     # smoke: split by the 'split' field

Steps, per split:
  1. drop questions with B between X and Y in the picture [theirs, inferred: their data has none],
     and questions where X and Y masks touch in the view from B's spot (MIN_GAP_B_PX) [amber]
  2. within each (B depth rank x B image side) group keep equally many of the four
     (A's answer x B's answer) combinations, so flips are 50% in every group [theirs]
  3. scale down to the target size, keeping that balance [theirs: sizes]
Then the metadata shortcut check: logistic regression from bookkeeping facts, trained on the train
split and scored on the test split [theirs], plus a flip version [violet, ours].

Writes $V4_DATA/<set>/questions_<split>.jsonl and $V4_DATA/v4_data_summary.json.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v4.config_v4 import CATEGORIES, MIN_GAP_B_PX, SHORTCUT_MAX, TARGET, V4_DATA

RANKS = ["nearest", "middle", "farthest"]
SIDES = ["left", "between", "right"]
CELLS = [("left", "left"), ("right", "right"), ("left", "right"), ("right", "left")]


def load(set_name, split):
    rows = []
    for f in sorted((V4_DATA / set_name).glob("*/candidates.jsonl")):
        rows += [json.loads(l) for l in open(f) if l.strip()]
    return [r for r in rows if r["split"] == split]


def gap_B(r):
    """Empty pixel columns between the X and Y masks in the view from B's spot (masks never overlap)."""
    x, y = r["B_view"]["X"], r["B_view"]["Y"]
    return max(x["xmin"] - y["xmax"], y["xmin"] - x["xmax"]) - 1


def balance(rows, target, rng):
    groups = {}
    for r in rows:
        groups.setdefault((r["depth_rank"], r["image_side"]), {}).setdefault((r["relA"], r["relB"]), []).append(r)
    per = {g: min(len(cells.get(c, [])) for c in CELLS) for g, cells in groups.items()}
    total = 4 * sum(per.values())
    if target and total > target:
        scale = target / total
        per = {g: int(np.floor(n * scale)) for g, n in per.items()}
    out = []
    for g, cells in groups.items():
        for c in CELLS:
            pool = cells.get(c, [])
            idx = rng.permutation(len(pool))[:per[g]]
            out += [pool[i] for i in idx]
    return sorted(out, key=lambda r: r["qid"]), per


def table(rows):
    cells = Counter(f"{r['relA'][0].upper()}/{r['relB'][0].upper()}" for r in rows)
    flip = lambda key, vals: {v: {"n": sum(r[key] == v for r in rows),
                                  "flip": (float(np.mean([r["flip"] for r in rows if r[key] == v]))
                                           if any(r[key] == v for r in rows) else None)} for v in vals}
    cats = Counter()
    for r in rows:
        for role in "BXY":
            cats[r[role]] += 1
    return {"questions": len(rows), "photos": len({r["photo"] for r in rows}),
            "rooms": len({r["room"] for r in rows}), "categories": sorted({r[k] for r in rows for k in "BXY"}),
            "cells": dict(cells), "by_rank": flip("depth_rank", RANKS), "by_side": flip("image_side", SIDES),
            "category_mentions": dict(cats), "B_category": dict(Counter(r["B"] for r in rows)),
            "left_rate_B": float(np.mean([r["relB"] == "left" for r in rows])) if rows else None}


def checks(name, t):
    out = []
    n = [t["cells"].get(k, 0) for k in ("L/L", "R/R", "L/R", "R/L")]
    ok = min(n) > 0 and max(n) / max(min(n), 1) <= 1.25
    out.append({"check": f"four A x B answer cells balanced ({name})", "value": t["cells"], "pass": ok})
    for key in ("by_rank", "by_side"):
        rates = {k: v["flip"] for k, v in t[key].items() if v["n"]}
        ok = all(0.45 <= r <= 0.55 for r in rates.values())
        label = "B depth rank" if key == "by_rank" else "B image side"
        out.append({"check": f"flip about half within each {label} group ({name})",
                    "value": {k: f"{v:.0%}" for k, v in rates.items()}, "pass": ok})
    return out


# ---------- metadata shortcut check ----------

def features(rows, kind):
    F = []
    for r in rows:
        f = []
        if kind in ("rank", "rank+side", "all"):
            f += [float(r["depth_rank"] == v) for v in RANKS]
        if kind in ("side", "rank+side", "all"):
            f += [float(r["image_side"] == v) for v in SIDES]
        if kind in ("distances+angles", "all"):
            d = r["dist"]
            f += [d["AB"], d["AX"], d["AY"], d["BX"], d["BY"], d["XY"], d["B_line"], r["angle_A"], r["angle_B"],
                  abs(r["A_view"]["bearing"]["B"]), abs(r["A_view"]["bearing"]["X"]), abs(r["A_view"]["bearing"]["Y"])]
        if kind in ("categories", "all"):
            for role in "BXY":
                f += [float(r[role] == c) for c in CATEGORIES]
        F.append(f)
    return np.array(F, dtype=float)


def within_a(pred, rows):
    """Class-balanced accuracy for B's answer, computed within each A answer and averaged."""
    relA = np.array([r["relA"] for r in rows])
    relB = np.array([r["relB"] for r in rows])
    scores = []
    for a in ("left", "right"):
        m = relA == a
        cls = [np.mean(pred[m & (relB == b)] == b) for b in ("left", "right") if np.any(m & (relB == b))]
        if len(cls) == 2:
            scores.append(np.mean(cls))
    return float(np.mean(scores)) if scores else float("nan")


NUMERIC = ["A-B distance", "A-X distance", "A-Y distance", "B-X distance", "B-Y distance", "X-Y distance",
           "B to X-Y line", "X-Y angle from A", "X-Y angle from B", "|bearing of B| from A",
           "|bearing of X| from A", "|bearing of Y| from A"]
SINGLE_MAX = 0.65  # [added, ours]: no single bookkeeping fact may predict the flip above this


def flip_score(Xtr, Xte, train, test):
    clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000))
    clf.fit(Xtr, [r["flip"] for r in train])
    p, y = clf.predict(Xte), np.array([r["flip"] for r in test])
    return float(np.mean([np.mean(p[y == c] == c) for c in (True, False)]))


def single_facts(train, test):
    """[added] each bookkeeping fact on its own -> does it flip? (test split, class-balanced)."""
    out = {}
    Ftr, Fte = features(train, "distances+angles"), features(test, "distances+angles")
    for i, name in enumerate(NUMERIC):
        out[name] = flip_score(Ftr[:, [i]], Fte[:, [i]], train, test)
    out["B depth rank"] = flip_score(features(train, "rank"), features(test, "rank"), train, test)
    out["B image side"] = flip_score(features(train, "side"), features(test, "side"), train, test)
    for role in "BXY":
        one = lambda rs: np.array([[float(r[role] == c) for c in CATEGORIES] for r in rs])
        out[f"{role}'s category"] = flip_score(one(train), one(test), train, test)
    return out


def shortcut(train, test):
    res = {}
    for kind in ("all", "rank", "side", "distances+angles", "categories", "rank+side"):
        Xtr, Xte = features(train, kind), features(test, kind)
        clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000))
        # [theirs] facts -> B's answer, scored within A's answer
        clf.fit(Xtr, [r["relB"] for r in train])
        theirs = within_a(clf.predict(Xte), test)
        # [violet] facts -> does it flip? (the same question with A's answer factored out)
        clf.fit(Xtr, [r["flip"] for r in train])
        p = clf.predict(Xte)
        y = np.array([r["flip"] for r in test])
        flip = float(np.mean([np.mean(p[y == c] == c) for c in (True, False)]))
        res[kind] = {"relB_within_A": theirs, "flip": flip}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-set", default="train")
    ap.add_argument("--test-set", default="test")
    ap.add_argument("--no-target", action="store_true", help="keep every balanced question (smoke)")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    summary = {"sets": {"train": args.train_set, "test": args.test_set}, "splits": {}, "checks": []}
    selected = {}
    for split, set_name in (("train", args.train_set), ("test", args.test_set)):
        cands = load(set_name, split)
        no_between = [r for r in cands if r["image_side"] != "between"]
        kept = [r for r in no_between if gap_B(r) >= MIN_GAP_B_PX]
        sel, per = balance(kept, None if args.no_target else TARGET[split], rng)
        selected[split] = sel
        out = V4_DATA / set_name / f"questions_{split}.jsonl"
        with open(out, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in sel)
        stats = Counter()
        rooms = sorted({r["room"] for r in cands})
        for room in rooms:
            s = json.loads((V4_DATA / set_name / room / "stats.json").read_text())
            stats.update(s["counts"])
            stats["photos rendered"] += s["photos"]
        summary["splits"][split] = {
            "candidates": table(cands), "selected": table(sel), "between_dropped": len(cands) - len(no_between),
            "touching_B_dropped": len(no_between) - len(kept),
            "per_group_cell": {f"{k[0]}/{k[1]}": v for k, v in per.items()}, "funnel": dict(stats),
            "rooms": rooms}
        summary["checks"] += checks(split, table(sel))
        print(f"{split}: {len(cands)} candidates ({len(cands) - len(no_between)} with B between, "
              f"{len(no_between) - len(kept)} with X and Y < {MIN_GAP_B_PX} px apart from B) -> {len(sel)} selected "
              f"from {len({r['photo'] for r in sel})} photos -> {out}")

    if selected["train"] and selected["test"]:
        sc = shortcut(selected["train"], selected["test"])
        summary["shortcut"] = sc
        summary["checks"].append({"check": f"metadata shortcut: B's answer from bookkeeping facts <= {SHORTCUT_MAX:.0%} "
                                           f"(test, within A's answer)", "value": f"{sc['all']['relB_within_A']:.0%}",
                                  "pass": sc["all"]["relB_within_A"] <= SHORTCUT_MAX})
        single = single_facts(selected["train"], selected["test"])
        summary["shortcut_single"] = single
        top = max(single, key=single.get)
        summary["checks"].append({"check": f"[added] flip from any single bookkeeping fact <= {SINGLE_MAX:.0%} (test)",
                                  "value": f"highest: {top} {single[top]:.0%}", "pass": single[top] <= SINGLE_MAX})
        summary["checks"].append({"check": "[added, information] flip from all bookkeeping facts together (test)",
                                  "value": f"{sc['all']['flip']:.0%}: the full geometry decides the flip",
                                  "pass": None})
        print("shortcut check (theirs: B's answer within A's | added: flip):")
        for k, v in sc.items():
            print(f"  {k:18s} {v['relB_within_A']:.0%} | {v['flip']:.0%}")
    for c in summary["checks"]:
        print(f"  {'info' if c['pass'] is None else 'pass' if c['pass'] else 'FAIL'}  {c['check']}: {c['value']}")
    path = V4_DATA / f"v4_data_summary{'_' + args.train_set if args.train_set != 'train' else ''}.json"
    path.write_text(json.dumps(summary, indent=1, default=str))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
