"""Steering redesign (PLAN_V4_STEER.md): numpy-only helpers shared by grads.py (vlm env) and grads_bagel.py
(bagel env), and later by the steering scripts.

The lean = log-odds of "left" against "right" for the first answer word:
LSE(left, Left) - LSE(right, Right) of the final logits at the token before the answer.
Gradient positions: GRAD_POS = token before the answer, last mention of X, last mention of Y.
"""
import json

import numpy as np

from v4.config_v4 import V4_RESULTS

GRAD_POS = ["answer", "name_X", "name_Y"]
FD_TARGETS = (0.25, 1.0)      # finite-difference check along the question's own gradient: predicted lean change
FD_B_MULT = (1.0, 4.0, 16.0)  # ... and along B's direction, in multiples of B's gap


def read_rows(model, split):
    return [json.loads(l) for l in open(V4_RESULTS / model / split / "rows.jsonl")]


def b_direction(model, L):
    """B's direction at hidden-state index L, token before the answer, training rooms (as in steer2_dirs.py):
    1/2 [(mean LL - mean LR) + (mean RL - mean RR)], first letter A's answer, second B's. Returns (unit, gap)."""
    rows = read_rows(model, "train")
    A = np.array([r["relA"] == "left" for r in rows])
    B = np.array([r["relB"] == "left" for r in rows])
    X = np.asarray(np.load(V4_RESULTS / model / "train" / "states.npy", mmap_mode="r")[0, L], dtype=np.float64)
    d = np.mean([X[(A == a) & B].mean(0) - X[(A == a) & ~B].mean(0) for a in (False, True)], axis=0)
    gap = float(np.linalg.norm(d))
    return d / gap, gap


def fd_layers(n_layers, headline):
    """Layers for the finite-difference check: every 4th from 2, plus the headline layer (never the last)."""
    return sorted(set(range(2, n_layers, 4)) | {int(headline)})


def headline(model):
    return int(json.loads((V4_RESULTS / model / "steer2" / "reliability.json").read_text())["headline_layer"])


def fd_summary(records, path):
    """records: dicts with layer, kind ('own_grad' or 'B'), size, pred, actual. Slope of actual on pred (through 0),
    correlation and median relative error, per kind; written to path and printed."""
    out = {"records": records, "summary": {}}
    for kind in sorted({r["kind"] for r in records}):
        rs = [r for r in records if r["kind"] == kind]
        p, a = np.array([r["pred"] for r in rs]), np.array([r["actual"] for r in rs])
        slope = float(p @ a / (p @ p)) if p @ p > 0 else float("nan")
        corr = float(np.corrcoef(p, a)[0, 1]) if len(rs) > 2 and p.std() > 0 and a.std() > 0 else float("nan")
        rel = float(np.median(np.abs(a - p) / np.maximum(np.abs(p), 1e-3)))
        out["summary"][kind] = {"n": len(rs), "slope": slope, "corr": corr, "median_rel_error": rel,
                                "median_abs_pred": float(np.median(np.abs(p))), "median_abs_actual": float(np.median(np.abs(a)))}
        print(f"finite-difference check, {kind}: n={len(rs)}  slope actual/pred {slope:.2f}  corr {corr:.3f}  "
              f"median relative error {rel:.0%}  (median |pred| {np.median(np.abs(p)):.3f}, |actual| {np.median(np.abs(a)):.3f})",
              flush=True)
        for size in sorted({r["size"] for r in rs}):
            ss = [r for r in rs if r["size"] == size]
            pp, aa = np.array([r["pred"] for r in ss]), np.array([r["actual"] for r in ss])
            print(f"    size {size:g}: mean pred {pp.mean():+.3f}  mean actual {aa.mean():+.3f}  "
                  f"corr {np.corrcoef(pp, aa)[0, 1] if len(ss) > 2 and pp.std() > 0 and aa.std() > 0 else float('nan'):.3f}",
                  flush=True)
    path.write_text(json.dumps(out, indent=1))
    return out
