"""Phase 4 (PLAN_V4.md): readers and controls on the v4 hidden states, for one model.

Reader [theirs]: standardize -> PCA(256) -> balanced logistic regression; C by 5-fold CV grouped by
training room from C_GRID [grid ours]; trained on the training rooms, scored on the test rooms.
Scores are class-balanced (a constant reader gets chance). B's answer is scored within A's answer:
the balanced score inside A = left and inside A = right, averaged [theirs].

Six readers at the token before the answer, every layer [theirs]: A's answer, the model's answer,
B's answer, flips?, B's depth rank (3 classes; 'nearest' has only 32 questions per split, see note),
B's image side (2 classes in practice: there is no 'between').

Controls for B's answer [theirs unless marked]:
  permutation band   100 readers on shuffled training labels (C as chosen for the real reader); 2.5-97.5%
  scrambled in room  labels shuffled among questions of the same training room (5 shuffles), scored
                     against the true test labels [scoring ours, the user's choice]
  text only          the prompt with no picture, every layer
  layer 0            the embedding layer (layer 0 of the six-reader run)
  vision, whole      vision features averaged over the image
  vision, per object [violet] B, X and Y features side by side; Qwen3-VL also its DeepStack maps
Token sweep [theirs]: the B-answer reader at the names of B / X / Y and the image tokens of B / X / Y.
Added [violet]: three-object language-model reader (B, X, Y image tokens side by side, every layer);
B's answer split by the X-Y angle from B.
Uncertainty: 1000 bootstrap resamples of test *photos* [the user's choice]; 95% intervals.

    python repro/v4/analyze.py --model qwen3vl [--jobs 16] [--layers 0 20 32]
Writes $V4_RESULTS/<model>/analysis.json and prints a summary.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v4.config_v4 import V4_RESULTS
from v4.prompts_v4 import POSITIONS

N_PCA, FOLDS, N_BOOT, N_PERM, N_SCRAMBLE = 256, 5, 1000, 100, 5
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
RANKS = ["nearest", "middle", "farthest"]


# ---------- labels and scores ----------

def load_rows(model, split):
    return [json.loads(l) for l in open(V4_RESULTS / model / split / "rows.jsonl")]


def labels(rows):
    lr = lambda w: 1 if w == "left" else 0
    return {
        "A": np.array([lr(r["relA"]) for r in rows]),
        "model": np.array([lr(r["answer"]) if r["answer"] in ("left", "right") else -1 for r in rows]),
        "B": np.array([lr(r["relB"]) for r in rows]),
        "flip": np.array([int(r["flip"]) for r in rows]),
        "rank": np.array([RANKS.index(r["depth_rank"]) for r in rows]),
        "side": np.array([1 if r["image_side"] == "left" else 0 for r in rows]),
    }


def balanced(pred, y):
    cls = np.unique(y)
    return float(np.mean([np.mean(pred[y == c] == c) for c in cls]))


def within_a(pred, y, a):
    """Balanced score for B's answer computed inside each A answer, then averaged."""
    parts = [balanced(pred[a == v], y[a == v]) for v in (0, 1) if np.any(a == v) and len(np.unique(y[a == v])) == 2]
    return float(np.mean(parts)) if parts else float("nan")


TARGETS = ["A", "model", "B", "flip", "rank", "side"]


def scorer(target, A):
    """score(pred, idx) on the questions idx of a split whose A labels are A."""
    if target == "B":
        return lambda pred, y, idx: within_a(pred, y, A[idx])
    return lambda pred, y, idx: balanced(pred, y)


# ---------- reader machinery ----------

def prep(Xtr, Xte, groups):
    """Standardize + PCA fitted once per CV fold and once on all training rooms."""
    folds = []
    for tr, va in GroupKFold(FOLDS).split(Xtr, groups=groups):
        sc = StandardScaler().fit(Xtr[tr])
        p = PCA(min(N_PCA, len(tr) - 1), random_state=0).fit(sc.transform(Xtr[tr]))
        folds.append((tr, va, p.transform(sc.transform(Xtr[tr])), p.transform(sc.transform(Xtr[va]))))
    sc = StandardScaler().fit(Xtr)
    p = PCA(min(N_PCA, len(Xtr) - 1), random_state=0).fit(sc.transform(Xtr))
    return folds, p.transform(sc.transform(Xtr)), p.transform(sc.transform(Xte))


def lr(C):
    return LogisticRegression(C=C, class_weight="balanced", max_iter=3000)


def fit(P, ytr, yte, score_tr, score_te, keep=None):
    """Pick C by grouped CV, refit on all training rooms, predict the test rooms."""
    folds, Ztr, Zte = P
    keep = np.ones(len(ytr), bool) if keep is None else keep
    best, best_c = -1, None
    for C in C_GRID:
        s = []
        for tr, va, ztr, zva in folds:
            m_tr, m_va = keep[tr], keep[va]
            if len(np.unique(ytr[tr][m_tr])) < 2:
                continue
            pred = lr(C).fit(ztr[m_tr], ytr[tr][m_tr]).predict(zva[m_va])
            s.append(score_tr(pred, ytr[va][m_va], va[m_va]))
        if s and np.mean(s) > best:
            best, best_c = float(np.mean(s)), C
    clf = lr(best_c).fit(Ztr[keep], ytr[keep])
    pred = clf.predict(Zte)
    return {"score": score_te(pred, yte, np.arange(len(yte))), "C": best_c, "cv": best}, pred


def boot_ci(pred, y, score_te, boots):
    vals = [score_te(pred[b], y[b], b) for b in boots]
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


# ---------- per-layer work ----------

class Data:
    def __init__(self, model):
        self.model = model
        self.rows = {s: load_rows(model, s) for s in ("train", "test")}
        self.y = {s: labels(self.rows[s]) for s in self.rows}
        self.groups = np.array([r["room"] for r in self.rows["train"]])
        rng = np.random.default_rng(0)
        photos = {}
        for i, r in enumerate(self.rows["test"]):
            photos.setdefault(r["photo"], []).append(i)
        plist = list(photos.values())
        self.boots = [np.concatenate([plist[j] for j in rng.integers(0, len(plist), len(plist))]) for _ in range(N_BOOT)]

    def arr(self, split, name):
        return np.load(V4_RESULTS / self.model / split / name, mmap_mode="r")

    def scorers(self, target):
        return (scorer(target, self.y["train"]["A"]), scorer(target, self.y["test"]["A"]))


def run_B(D, P, extra=None):
    """B-answer reader on prepared features, with its bootstrap interval."""
    s_tr, s_te = D.scorers("B")
    res, pred = fit(P, D.y["train"]["B"], D.y["test"]["B"], s_tr, s_te)
    res["ci"] = boot_ci(pred, D.y["test"]["B"], s_te, D.boots)
    return res, pred


def layer_job(model, L):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    D = Data(model)
    yt, ye = D.y["train"], D.y["test"]
    S_tr, S_te = D.arr("train", "states.npy"), D.arr("test", "states.npy")
    get = lambda S, p: np.asarray(S[p, L], dtype=np.float32)
    out = {"layer": L, "targets": {}, "sweep": {}}
    rng = np.random.default_rng(L)

    # six readers at the token before the answer
    P = prep(get(S_tr, 0), get(S_te, 0), D.groups)
    for t in TARGETS:
        s_tr, s_te = D.scorers(t)
        keep = yt[t] >= 0
        res, pred = fit(P, yt[t], ye[t], s_tr, s_te, keep=keep)
        if t == "model":  # score against the model's own (valid) answers
            m = ye[t] >= 0
            res["score"] = balanced(pred[m], ye[t][m])
            res["ci"] = boot_ci(pred, ye[t], lambda p, y, i: balanced(p[y >= 0], y[y >= 0]), D.boots)
        else:
            res["ci"] = boot_ci(pred, ye[t], s_te, D.boots)
        out["targets"][t] = res
        if t == "B":
            out["pred_B"] = pred.tolist()
            best_c = res["C"]
            # permutation band: shuffled training labels, the same C
            folds, Ztr, Zte = P
            perm = [s_te(lr(best_c).fit(Ztr, rng.permutation(yt["B"])).predict(Zte), ye["B"], np.arange(len(ye["B"])))
                    for _ in range(N_PERM)]
            out["perm"] = [float(np.percentile(perm, 2.5)), float(np.percentile(perm, 97.5))]
            # scrambled within room: shuffle B's answer among each training room's questions
            scr = []
            for _ in range(N_SCRAMBLE):
                ys = yt["B"].copy()
                for g in np.unique(D.groups):
                    idx = np.flatnonzero(D.groups == g)
                    ys[idx] = ys[rng.permutation(idx)]
                scr.append(fit(P, ys, ye["B"], s_tr, s_te)[0]["score"])
            out["scrambled"] = {"mean": float(np.mean(scr)), "max": float(np.max(scr))}
    del P

    # text only
    T_tr, T_te = D.arr("train", "text_only.npy"), D.arr("test", "text_only.npy")
    out["text_only"] = run_B(D, prep(np.asarray(T_tr[L], np.float32), np.asarray(T_te[L], np.float32), D.groups))[0]

    # token sweep: the B-answer reader at the other saved positions
    for p in range(1, len(POSITIONS)):
        out["sweep"][POSITIONS[p]] = run_B(D, prep(get(S_tr, p), get(S_te, p), D.groups))[0]

    # [violet] three-object language-model reader: B, X, Y image tokens side by side
    cat = lambda S: np.concatenate([get(S, p) for p in (4, 5, 6)], axis=1)
    out["three_objects"] = run_B(D, prep(cat(S_tr), cat(S_te), D.groups))[0]
    return out


def vision_jobs(model):
    D = Data(model)
    out = {}
    V_tr, V_te = D.arr("train", "vision.npy"), D.arr("test", "vision.npy")
    f = lambda V, ks: np.concatenate([np.asarray(V[k], np.float32) for k in ks], axis=1)
    out["vision_whole"] = run_B(D, prep(f(V_tr, [0]), f(V_te, [0]), D.groups))[0]
    out["vision_objects"] = run_B(D, prep(f(V_tr, [1, 2, 3]), f(V_te, [1, 2, 3]), D.groups))[0]
    ds = V4_RESULTS / model / "train" / "deepstack.npy"
    if ds.exists():
        G_tr, G_te = D.arr("train", "deepstack.npy"), D.arr("test", "deepstack.npy")
        g = lambda G, ks: np.concatenate([np.asarray(G[k, j], np.float32) for k in ks for j in (1, 2, 3)], axis=1)
        for k in range(G_tr.shape[0]):
            out[f"deepstack_{k}_objects"] = run_B(D, prep(g(G_tr, [k]), g(G_te, [k]), D.groups))[0]
        out["deepstack_all_objects"] = run_B(D, prep(g(G_tr, range(G_tr.shape[0])), g(G_te, range(G_te.shape[0])),
                                                     D.groups))[0]
    return out


# ---------- groups, checks, summary ----------

def group_scores(D, pred, key, values):
    pred, y, A = np.array(pred), D.y["test"]["B"], D.y["test"]["A"]
    res = {}
    for v in values:
        m = np.array([key(r) == v for r in D.rows["test"]])
        if m.sum() == 0:
            continue
        idx = np.flatnonzero(m)
        sc = within_a(pred[idx], y[idx], A[idx])
        boots = [b[np.isin(b, idx)] for b in D.boots]  # the group's questions within each photo resample
        vals = [within_a(pred[b], y[b], A[b]) for b in boots if len(b)]
        res[v] = {"n": int(m.sum()), "score": sc,
                  "ci": [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]}
    return res


def behaviour(rows):
    n = len(rows)
    f = lambda c: sum(1 for r in rows if c(r)) / n
    words = [r["answer"] for r in rows]
    top = max(f(lambda r: r["answer"] == "left"), f(lambda r: r["answer"] == "right"))
    return {"n": n, "gives_B": f(lambda r: r["answer"] == r["relB"]), "gives_A": f(lambda r: r["answer"] == r["relA"]),
            "says_left": f(lambda r: r["answer"] == "left"), "top_word": top,
            "direct_A": f(lambda r: r["direct_A"] == r["relA"]), "direct_B": f(lambda r: r["direct_B"] == r["relB"]),
            "invalid": words.count("invalid") / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--layers", type=int, nargs="*", help="default: all")
    args = ap.parse_args()
    t0 = time.time()
    D = Data(args.model)
    n_layers = D.arr("train", "states.npy").shape[1]
    layers = args.layers if args.layers else list(range(n_layers))
    print(f"{args.model}: {n_layers} layer outputs; train {len(D.rows['train'])}, test {len(D.rows['test'])}; "
          f"running {len(layers)} layers on {args.jobs} workers", flush=True)

    jobs = [delayed(layer_job)(args.model, L) for L in layers] + [delayed(vision_jobs)(args.model)]
    results = Parallel(n_jobs=args.jobs, verbose=5)(jobs)
    per_layer, vis = results[:-1], results[-1]
    per_layer.sort(key=lambda r: r["layer"])

    # best layer for B's answer, and the group breakdowns there
    best = max(per_layer, key=lambda r: r["targets"]["B"]["score"])
    groups = {
        "depth_rank": group_scores(D, best["pred_B"], lambda r: r["depth_rank"], RANKS),
        "image_side": group_scores(D, best["pred_B"], lambda r: r["image_side"], ["left", "right"]),
        # [violet] X-Y angle from B: near-lined-up views vs wider ones
        "angle_B": group_scores(D, best["pred_B"], lambda r: "15-25 deg" if r["angle_B"] < 25 else ">= 25 deg",
                                ["15-25 deg", ">= 25 deg"]),
    }
    for r in per_layer:
        r.pop("pred_B", None)

    beh = {s: behaviour(D.rows[s]) for s in ("train", "test")}
    mid_late = [r for r in per_layer if r["layer"] >= n_layers // 2]
    rd = lambda x: round(x, 2)  # pass marks are compared on whole percentages, as printed
    checks = []
    for s in ("train", "test"):
        checks.append({"group": 2, "check": f"direct-A (plain question on A's photo) >= 90% ({s})",
                       "value": beh[s]["direct_A"], "result": "pass" if beh[s]["direct_A"] >= .9 else "fail"})
        checks.append({"group": 2, "check": f"direct-B (plain question from B's spot) >= 90% ({s})",
                       "value": beh[s]["direct_B"], "result": "pass" if beh[s]["direct_B"] >= .9 else "fail"})
        tw = beh[s]["top_word"]
        checks.append({"group": 3, "check": f"task answers not one constant word ({s}; > 95% blocks, > 80% marks)",
                       "value": tw, "result": "fail" if tw > .95 else "mark" if tw > .8 else "pass"})
    a_best = max(r["targets"]["A"]["score"] for r in mid_late)
    m_best = max(r["targets"]["model"]["score"] for r in mid_late)
    checks.append({"group": 4, "check": "extraction works: A's-answer or model's-answer reader >= 90% in the mid/late layers",
                   "value": max(a_best, m_best), "result": "pass" if max(a_best, m_best) >= .9 else "fail"})
    vw = vis["vision_whole"]["score"]
    checks.append({"group": 5, "check": "whole-image vision reader for B's answer <= 55%", "value": vw,
                   "result": "pass" if rd(vw) <= .55 else "mark"})
    l0 = per_layer[0]["targets"]["B"]["score"]
    checks.append({"group": 5, "check": "layer-0 reader for B's answer <= 55%", "value": l0,
                   "result": "pass" if rd(l0) <= .55 else "mark"})
    to = max(r["text_only"]["score"] for r in per_layer)
    checks.append({"group": 5, "check": "text-only reader for B's answer <= 55% at every layer", "value": to,
                   "result": "pass" if rd(to) <= .55 else "mark"})
    sc = max(r["scrambled"]["max"] for r in per_layer)
    checks.append({"group": 6, "check": "control task (B's answer scrambled within room) <= 62% at every layer",
                   "value": sc, "result": "pass" if rd(sc) <= .62 else "fail"})
    pb = max(r["perm"][1] for r in per_layer)
    checks.append({"group": 6, "check": "permutation band stays near chance (max <= 60%)", "value": pb,
                   "result": "pass" if rd(pb) <= .60 else "fail"})
    vo = vis["vision_objects"]["score"]
    checks.append({"group": 7, "check": "[added] per-object vision reader for B's answer (information: how much the picture "
                                        "features already give away)", "value": vo, "result": "info"})

    out = {"model": args.model, "n_layers": n_layers, "layers": per_layer, "vision": vis, "groups": groups,
           "best_layer": best["layer"], "behaviour": beh, "checks": checks,
           "notes": {"depth_rank": "3 classes as in the PhD's design; 'nearest' has only 32 questions per split, "
                                   "so that class's recall, and the 3-class score, is noisy.",
                     "image_side": "2 classes in practice (no 'between'): chance is 50%, not 33%.",
                     "bootstrap": f"{N_BOOT} resamples of test photos (the user's choice).",
                     "scrambled": "trained on labels shuffled within each training room, scored against the true "
                                  "test labels (the user's choice)."},
           "seconds": round(time.time() - t0)}
    path = V4_RESULTS / args.model / "analysis.json"
    path.write_text(json.dumps(out, indent=1))

    print(f"\n{args.model}: B's answer peaks at layer {best['layer']}: {best['targets']['B']['score']:.0%} "
          f"[{best['targets']['B']['ci'][0]:.0%}-{best['targets']['B']['ci'][1]:.0%}]")
    for r in per_layer:
        t = r["targets"]
        print(f"  L{r['layer']:2d} A {t['A']['score']:.0%} model {t['model']['score']:.0%} B {t['B']['score']:.0%} "
              f"flip {t['flip']['score']:.0%} rank {t['rank']['score']:.0%} side {t['side']['score']:.0%} | perm "
              f"{r['perm'][0]:.0%}-{r['perm'][1]:.0%} scr {r['scrambled']['max']:.0%} text {r['text_only']['score']:.0%} "
              f"3obj {r['three_objects']['score']:.0%} | sweep " +
              " ".join(f"{k} {v['score']:.0%}" for k, v in r["sweep"].items()), flush=True)
    print("vision:", {k: f"{v['score']:.0%}" for k, v in vis.items()})
    print("groups at best layer:", {k: {g: f"{v['score']:.0%} (n={v['n']})" for g, v in d.items()} for k, d in groups.items()})
    for c in checks:
        print(f"  {c['result']:4s}  {c['check']}: {c['value']:.0%}")
    print(f"-> {path} ({out['seconds']} s)")


if __name__ == "__main__":
    main()
