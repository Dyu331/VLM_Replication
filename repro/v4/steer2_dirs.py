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
  build         after the gradients (Phase 2): per layer, at the token before the answer,
                  realistic directions   top principal directions of the training states holding SPAN_CUT of
                           their variance; the gradients are projected onto them first (most of a gradient points
                           where states never go, and B and the random controls only point where they do)
                  g        mean realistic gradient of the final lean over training questions, unit length
                  G        top right-singular vectors of the realistic training gradients (not centred, so g lies
                           in it): enough for 90% of their squared length, capped at K_CAP
                  B, A     difference-of-means directions (training rooms), unit length; gap = plain length
                  D        depth-rank direction (farthest minus middle, averaged over the four A x B cells):
                           structured but unrelated to left/right, the specificity control
                  B_perp, A_perp, D_perp   with G removed, unit length
                  rand     N_RAND data-shaped random directions (random mixes of the centred training states,
                           outlier dimensions left out where the top two hold > OUTLIER_SHARE of the variance),
                           G removed, unit length
                Gate 2 check on the realistic part of the TEST questions' gradients: |cos| with each question's
                gradient, as a share of its |cos| with g; repeated at SPAN_CUTS_CHECK for robustness.

    python repro/v4/steer2_dirs.py reliability --model qwen3vl
    python repro/v4/steer2_dirs.py build --model qwen3vl
Writes $V4_RESULTS/<model>/steer2/reliability.json, dirs.npz and gate2.json.
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
K_CAP, K_SHARE = 10, 0.90
N_RAND = 10
OUTLIER_SHARE = 0.20
SPAN_CUT = 0.99                  # realistic directions: top principal directions of the training states
SPAN_CUTS_CHECK = (0.95, 0.999)  # robustness check


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
              f"            {(b['gap'] / b['gap_split'] if b['gap_split'] > 0 else float('nan')):.2f}/{(bp['gap'] / bp['gap_split'] if bp['gap_split'] > 0 else float('nan')):.2f}"
              f"            {b['quality']:.0%}/{bp['quality']:.0%}"
              f"          {a['split_half']:+.2f}/{ap['split_half']:+.2f}"
              f"          {r['outlier_top1']:.0%}/{r['outlier_top2']:.0%}", flush=True)


def realistic(Xtr, cut):
    """Orthonormal rows spanning the directions along which the training states vary: the top principal
    directions of the centred states holding a share `cut` of their variance."""
    Xc = Xtr - Xtr.mean(0)
    _, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
    share = np.cumsum(sv ** 2) / np.sum(sv ** 2)
    return Vt[: int(np.searchsorted(share, cut) + 1)]


def subspace(Gtr, P):
    """g and G from the training gradients projected onto the realistic directions P."""
    C = Gtr @ P.T                                           # coordinates in the realistic space
    _, sv, Vt = np.linalg.svd(C, full_matrices=False)       # not centred, so g lies in G
    share = np.cumsum(sv ** 2) / np.sum(sv ** 2)
    k90 = int(np.searchsorted(share, K_SHARE) + 1)
    k = min(k90, K_CAP)
    return unit(C.mean(0) @ P), Vt[:k] @ P, k, k90, float(share[k - 1])


def layer_build(model, L, out1, rel):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    ytr = labels(load_rows(model, "train"))
    Xtr = np.asarray(np.load(V4_RESULTS / model / "train" / "states.npy", mmap_mode="r")[POS, L], dtype=np.float64)
    Gtr = np.asarray(np.load(V4_RESULTS / model / "steer2" / "train" / "grads.npy", mmap_mode="r")[POS, L], dtype=np.float64)
    Gte = np.asarray(np.load(V4_RESULTS / model / "steer2" / "test" / "grads.npy", mmap_mode="r")[POS, L], dtype=np.float64)
    H = Xtr.shape[1]
    dB = mean_diff(Xtr, ytr["A"], ytr["B"])
    dA = mean_diff(Xtr, ytr["B"], ytr["A"])
    cells = [(ytr["A"] == a) & (ytr["B"] == b) for a in (0, 1) for b in (0, 1)]
    dD = np.mean([Xtr[c & (ytr["rank"] == 2)].mean(0) - Xtr[c & (ytr["rank"] == 1)].mean(0) for c in cells], axis=0)
    keep = np.ones(H, bool)
    if out1["outlier_top2"] > OUTLIER_SHARE:
        keep[out1["outlier_dims"]] = False
    rng = np.random.default_rng(2000 + L)
    Xc = Xtr - Xtr.mean(0)
    raw_rand = []
    for _ in range(N_RAND):
        r = Xc.T @ rng.standard_normal(len(Xc))
        r[~keep] = 0
        raw_rand.append(r)

    def one(cut):
        P = realistic(Xtr, cut)
        g, Gb, k, k90, share_k = subspace(Gtr, P)
        perp = lambda v: unit(v - Gb.T @ (Gb @ v))
        te = (Gte @ P.T) @ P                                # test gradients, realistic part
        real_share = np.sum(te ** 2, 1) / np.maximum(np.sum(Gte ** 2, 1), 1e-30)
        gn = te / np.maximum(np.linalg.norm(te, axis=1, keepdims=True), 1e-12)
        med = lambda v: float(np.median(np.abs(gn @ v)))
        rand = np.stack([perp(r) for r in raw_rand])
        c_g = med(g)
        st = {"cut": cut, "span": len(P), "k": k, "k90": k90, "share_k": share_k,
              "grad_realistic_share": float(np.median(real_share)),
              "test_grad_in_G": float(np.mean(np.sum((gn @ Gb.T) ** 2, 1))),
              "cos_g": c_g, "cos_B": med(unit(dB)), "cos_A": med(unit(dA)), "cos_D": med(unit(dD)),
              "cos_Bperp": med(perp(dB)), "cos_Aperp": med(perp(dA)), "cos_Dperp": med(perp(dD)),
              "cos_rand": float(np.median(np.abs(gn @ rand.T))),
              "B_overlap_g": float(unit(dB) @ g), "A_overlap_g": float(unit(dA) @ g),
              "B_kept": float(np.linalg.norm(dB - Gb.T @ (Gb @ dB)) / np.linalg.norm(dB))}
        for n in ("Bperp", "Aperp", "Dperp", "rand", "B"):
            st[f"ratio_{n}"] = st[f"cos_{n}"] / c_g
        vecs = {"B": unit(dB), "A": unit(dA), "D": unit(dD), "g": g,
                "B_perp": perp(dB), "A_perp": perp(dA), "D_perp": perp(dD)}
        return st, vecs, Gb, rand

    main_st, vecs, Gb, rand = one(SPAN_CUT)
    stats = {"layer": L, **main_st, "gap": float(np.linalg.norm(dB)), "gap_split": rel["B_raw"]["gap_split"],
             "gap_D": float(np.linalg.norm(dD)), "state_norm": float(np.linalg.norm(Xtr, axis=1).mean()),
             "outliers_left_out": int((~keep).sum()),
             "robustness": [one(c)[0] for c in SPAN_CUTS_CHECK]}
    return stats, vecs, Gb.astype(np.float32), rand.astype(np.float32)


def build(args):
    root = V4_RESULTS / args.model / "steer2"
    rel = json.loads((root / "reliability.json").read_text())
    by_layer = {r["layer"]: r for r in rel["layers"]}
    n = max(by_layer)
    layers = list(range(1, n))                             # the last index can be after the final norm: skipped
    res = Parallel(n_jobs=args.jobs)(delayed(layer_build)(args.model, L, by_layer[L], by_layer[L]) for L in layers)
    stats = [r[0] for r in res]
    H = res[0][1]["B"].shape[0]
    vec = {name: np.zeros((n + 1, H), np.float32) for name in res[0][1]}
    Gs = np.zeros((n + 1, K_CAP, H), np.float32)
    R = np.zeros((n + 1, N_RAND, H), np.float32)
    ks = np.zeros(n + 1, int)
    gap = np.zeros(n + 1)
    for L, (st, v, Gb, rand) in zip(layers, res):
        for name in vec:
            vec[name][L] = v[name]
        Gs[L, :len(Gb)] = Gb
        ks[L], R[L], gap[L] = len(Gb), rand, st["gap"]
    np.savez(root / "dirs.npz", G=Gs, k=ks, rand=R, gap=gap, headline_layer=rel["headline_layer"], **vec)
    (root / "gate2.json").write_text(json.dumps({"model": args.model, "headline_layer": rel["headline_layer"],
                                                 "k_cap": K_CAP, "k_share": K_SHARE, "span_cut": SPAN_CUT,
                                                 "layers": stats}, indent=1))
    head = rel["headline_layer"]
    print(f"{args.model}: realistic g and G (span: {SPAN_CUT:.0%} of the training states' variance) -> dirs.npz")
    print("Gate 2 on the realistic part of the TEST gradients: median |cos| with each question's gradient, as a share of g's")
    print("  layer span  k (k90) grad real  in G   cos g   B      B_perp  A_perp  D_perp  rand   | B.g     A.g    B kept")
    for st in stats:
        mark = "*" if st["layer"] == head else " "
        print(f" {mark}L{st['layer']:2d} {st['span']:4d}  {st['k']:2d} ({st['k90']:3d})  {st['grad_realistic_share']:5.0%}  "
              f"{st['test_grad_in_G']:5.0%}  {st['cos_g']:.3f}  {st['ratio_B']:6.1%} {st['ratio_Bperp']:6.1%}  "
              f"{st['ratio_Aperp']:6.1%}  {st['ratio_Dperp']:6.1%}  {st['ratio_rand']:6.1%} | {st['B_overlap_g']:+.3f}  "
              f"{st['A_overlap_g']:+.3f}  {st['B_kept']:.0%}", flush=True)
    print("robustness at the headline layer (span cut: span, k, B_perp/g, rand/g, B.g):")
    hs = next(st for st in stats if st["layer"] == head)
    for r in [hs] + hs["robustness"]:
        print(f"  {r['cut']:.3f}: span {r['span']:4d}  k {r['k']:2d} ({r['k90']})  B_perp {r['ratio_Bperp']:.1%}  "
              f"rand {r['ratio_rand']:.1%}  B.g {r['B_overlap_g']:+.3f}  A.g {r['A_overlap_g']:+.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("reliability")
    p.add_argument("--model", required=True)
    p.add_argument("--jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    p = sub.add_parser("build")
    p.add_argument("--model", required=True)
    p.add_argument("--jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    args = ap.parse_args()
    {"reliability": reliability, "build": build}[args.cmd](args)


if __name__ == "__main__":
    main()
