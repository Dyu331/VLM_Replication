"""Steering, redesigned (PLAN_V4_STEER.md), CPU part.

  reliability   Phase 1: are B's and A's directions real signals, and do they need denoising?
                Per layer, at the token before the answer, from the saved hidden states:
                  split-half   rooms split in two (10 random splits); cosine between the directions of the halves
                  null         the same with B's (or A's) labels shuffled inside each group: what noise alone gives
                  train/test   cosine between the direction from the training rooms and from the test rooms
                  gap          plain |d| against the split-half estimate sqrt(d1 . d2) (noise inflates |d|)
                  quality      within A's answer, how well the projection onto B's direction separates test
                               B-left from B-right (threshold: midpoint of the training class means)
                  outliers     share of the training-state variance in the top 1 and 2 dimensions
                each for the raw direction and for the direction inside the top 256 principal components
                (fit on the training states, no labels).
                Headline layer: earliest layer within 0.01 of the maximum cross-validated B-reader score.

    python repro/v4/steer2_dirs.py reliability --model qwen3vl
Writes $V4_RESULTS/<model>/steer2/reliability.json.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v4.analyze import N_PCA, labels, load_rows, within_a
from v4.config_v4 import V4_RESULTS

N_SPLITS = 10
POS = 0  # token before the answer


def unit(v):
    return v / np.linalg.norm(v)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def mean_diff(X, by, target):
    """1/2 sum over the two values of `by` of (mean target=1 - mean target=0). For B: by = A's answer."""
    return np.mean([X[(by == v) & (target == 1)].mean(0) - X[(by == v) & (target == 0)].mean(0) for v in (0, 1)], axis=0)


def shuffled(target, by, rng):
    t = target.copy()
    for v in (0, 1):
        idx = np.flatnonzero(by == v)
        t[idx] = t[rng.permutation(idx)]
    return t


def split_half(X, by, target, rooms, proj, rng):
    """Cosines between the two halves' directions, and their dot products (for the gap estimate)."""
    names = np.unique(rooms)
    cs, dots = [], []
    for _ in range(N_SPLITS):
        half = set(rng.permutation(names)[: len(names) // 2])
        m = np.array([r in half for r in rooms])
        d1, d2 = proj(mean_diff(X[m], by[m], target[m])), proj(mean_diff(X[~m], by[~m], target[~m]))
        cs.append(cos(d1, d2))
        dots.append(float(d1 @ d2))
    return cs, dots


def quality(d, Xtr, Atr, Btr, Xte, Ate, Bte):
    """Within A's answer: classify B by the projection onto d, threshold = midpoint of the training class means."""
    u = unit(d)
    ptr, pte = Xtr @ u, Xte @ u
    pred = np.zeros(len(Xte), int)
    for v in (0, 1):
        thr = (ptr[(Atr == v) & (Btr == 1)].mean() + ptr[(Atr == v) & (Btr == 0)].mean()) / 2
        m = Ate == v
        pred[m] = (pte[m] > thr).astype(int)
    return within_a(pred, Bte, Ate)


def layer_stats(model, L):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    rtr, rte = load_rows(model, "train"), load_rows(model, "test")
    ytr, yte = labels(rtr), labels(rte)
    rooms = np.array([r["room"] for r in rtr])
    Xtr = np.asarray(np.load(V4_RESULTS / model / "train" / "states.npy", mmap_mode="r")[POS, L], dtype=np.float64)
    Xte = np.asarray(np.load(V4_RESULTS / model / "test" / "states.npy", mmap_mode="r")[POS, L], dtype=np.float64)
    pca = PCA(min(N_PCA, len(Xtr) - 1), random_state=0).fit(Xtr)
    P = pca.components_
    proj = {"raw": lambda d: d, "pca": lambda d: P.T @ (P @ d)}

    var = Xtr.var(0)
    top = np.sort(var)[::-1]
    out = {"layer": L, "state_norm": float(np.linalg.norm(Xtr, axis=1).mean()),
           "outlier_top1": float(top[0] / var.sum()), "outlier_top2": float(top[:2].sum() / var.sum()),
           "outlier_dims": [int(i) for i in np.argsort(var)[::-1][:2]],
           "pca_var_kept": float(pca.explained_variance_ratio_.sum())}
    for name, target, by in (("B", "B", "A"), ("A", "A", "B")):
        dtr_raw = mean_diff(Xtr, ytr[by], ytr[target])
        dte_raw = mean_diff(Xte, yte[by], yte[target])
        for kind, f in proj.items():
            rng = np.random.default_rng(1000 + L)
            dtr, dte = f(dtr_raw), f(dte_raw)
            cs, dots = split_half(Xtr, ytr[by], ytr[target], rooms, f, rng)
            ncs = []
            for _ in range(N_SPLITS):
                ncs.append(split_half(Xtr, ytr[by], shuffled(ytr[target], ytr[by], rng), rooms, f, rng)[0][0])
            out[f"{name}_{kind}"] = {
                "split_half": float(np.mean(cs)), "split_half_min": float(np.min(cs)),
                "null": float(np.mean(ncs)), "null_max": float(np.max(ncs)),
                "train_test": cos(dtr, dte),
                "gap": float(np.linalg.norm(dtr)),
                "gap_split": float(np.sqrt(max(np.mean(dots), 0.0))),
                "kept_by_pca": float(np.linalg.norm(f(dtr_raw)) / np.linalg.norm(dtr_raw)),
                "quality": quality(dtr, Xtr, ytr["A"], ytr["B"], Xte, yte["A"], yte["B"]) if name == "B" else None,
            }
    return out


def headline_layer(an):
    cv = {r["layer"]: r["targets"]["B"]["cv"] for r in an["layers"]}
    top = max(cv.values())
    return min(L for L, v in cv.items() if v >= top - 0.01)


def reliability(args):
    an = json.loads((V4_RESULTS / args.model / "analysis.json").read_text())
    layers = sorted(r["layer"] for r in an["layers"])
    res = Parallel(n_jobs=args.jobs)(delayed(layer_stats)(args.model, L) for L in layers)
    res.sort(key=lambda r: r["layer"])
    head = headline_layer(an)
    out_dir = V4_RESULTS / args.model / "steer2"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "reliability.json").write_text(json.dumps({"model": args.model, "headline_layer": head,
                                                          "n_splits": N_SPLITS, "layers": res}, indent=1))
    print(f"{args.model}: headline layer L{head} (earliest within 0.01 of the max cross-validated B score)")
    print("  layer   B split-half raw/pca (null)   B train-test raw/pca   B gap ratio raw/pca   B quality raw/pca"
          "   A split-half raw/pca   top-1/2 var")
    for r in res:
        b, bp, a, ap = r["B_raw"], r["B_pca"], r["A_raw"], r["A_pca"]
        mark = "*" if r["layer"] == head else " "
        print(f" {mark}L{r['layer']:2d}   {b['split_half']:+.2f}/{bp['split_half']:+.2f} ({b['null']:+.2f})"
              f"          {b['train_test']:+.2f}/{bp['train_test']:+.2f}"
              f"            {b['gap'] / max(b['gap_split'], 1e-9):.2f}/{bp['gap'] / max(bp['gap_split'], 1e-9):.2f}"
              f"            {b['quality']:.0%}/{bp['quality']:.0%}"
              f"          {a['split_half']:+.2f}/{ap['split_half']:+.2f}"
              f"          {r['outlier_top1']:.0%}/{r['outlier_top2']:.0%}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("reliability")
    p.add_argument("--model", required=True)
    p.add_argument("--jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    args = ap.parse_args()
    {"reliability": reliability}[args.cmd](args)


if __name__ == "__main__":
    main()
