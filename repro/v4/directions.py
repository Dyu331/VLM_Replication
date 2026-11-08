"""Phase 6a (follow-up, violet): steering directions at the token before the answer, every layer,
computed on the TRAINING rooms only (steering is then tested on the test rooms).

Directions (each scaled to unit length; steer.py pushes by k x B's gap, the same length for all):
  B            difference of means for B's answer, inside each A answer, averaged:
               1/2 [(mean LL - mean LR) + (mean RL - mean RR)]   (first letter A, second B; points to B = left)
               This cancels A's answer and "flips?" and leaves B's answer.  Its length = "B's gap".
  B_normal     the B-answer reader's normal vector (standardize -> PCA 256 -> logistic regression, C as in
               analysis.json), mapped back to the hidden space: components^T w / feature scale
  B_minus_say  B with its "say left" component removed
  say          difference of means by the model's own answer (says left minus says right)
  random_0..4  random directions

    python repro/v4/directions.py --model qwen3vl
Writes $V4_RESULTS/<model>/directions.npz.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v4.analyze import N_PCA, labels, load_rows, lr
from v4.config_v4 import V4_RESULTS

NAMES = ["B", "B_normal", "B_minus_say", "say"] + [f"random_{i}" for i in range(5)]


def unit(v):
    return v / np.linalg.norm(v)


def mean_diff_B(X, A, B):
    return np.mean([X[(A == a) & (B == 1)].mean(0) - X[(A == a) & (B == 0)].mean(0) for a in (0, 1)], axis=0)


def layer_dirs(model, L, C):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    y = labels(load_rows(model, "train"))
    S = np.load(V4_RESULTS / model / "train" / "states.npy", mmap_mode="r")
    X = np.asarray(S[0, L], dtype=np.float64)
    dB = mean_diff_B(X, y["A"], y["B"])
    m = y["model"]
    say = X[m == 1].mean(0) - X[m == 0].mean(0)
    sc = StandardScaler().fit(X)
    p = PCA(min(N_PCA, len(X) - 1), random_state=0).fit(sc.transform(X))
    clf = lr(C).fit(p.transform(sc.transform(X)), y["B"])   # label 1 = left, so +w points to B = left
    normal = (p.components_.T @ clf.coef_[0]) / sc.scale_
    s = unit(say)
    perp = dB - (dB @ s) * s
    rand = np.random.default_rng(1000 + L).standard_normal((5, X.shape[1]))
    D = np.stack([unit(v) for v in [dB, normal, perp, say, *rand]])
    return {"layer": L, "dirs": D.astype(np.float32), "gap": float(np.linalg.norm(dB)),
            "say_gap": float(np.linalg.norm(say)),
            "state_norm": float(np.linalg.norm(X, axis=1).mean()),
            "cos_B_say": float(unit(dB) @ s), "cos_B_normal": float(unit(dB) @ unit(normal)),
            "perp_frac": float(np.linalg.norm(perp) / np.linalg.norm(dB))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    args = ap.parse_args()
    an = json.loads((V4_RESULTS / args.model / "analysis.json").read_text())
    Cs = {r["layer"]: r["targets"]["B"]["C"] for r in an["layers"]}
    layers = sorted(Cs)
    res = Parallel(n_jobs=args.jobs)(delayed(layer_dirs)(args.model, L, Cs[L]) for L in layers)
    res.sort(key=lambda r: r["layer"])
    out = {k: np.array([r[k] for r in res]) for k in res[0] if k != "dirs"}
    np.savez(V4_RESULTS / args.model / "directions.npz", names=np.array(NAMES), dirs=np.stack([r["dirs"] for r in res]),
             best_layer=an["best_layer"], **out)
    print(f"{args.model}: directions for {len(layers)} layers -> directions.npz")
    for r in res:
        print(f"  L{r['layer']:2d} gap {r['gap']:7.2f} (|h| {r['state_norm']:8.1f})  cos(B, say) {r['cos_B_say']:+.2f}  "
              f"cos(B, normal) {r['cos_B_normal']:+.2f}  B minus say keeps {r['perp_frac']:.0%}", flush=True)


if __name__ == "__main__":
    main()
