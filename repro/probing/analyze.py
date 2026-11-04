"""Left/right readers per layer, the report's two checks, chance band, bootstrap and the B score.

Design follows the report's Section 3 (choices it does not state are marked "ours"):
  - one reader per layer: standardize -> PCA(256) -> class-weighted logistic regression,
    fitted on the training rooms only (ours: standardize, PCA on train only, C=1)
  - direct reader: trained on direct questions on A's and B's photos (s4 + s2), label = the
    model's own answer; identity reader: trained on the identity prompt, same labels
  - ceiling: 5-fold cross-validation grouped by room on the training prompts (ours: grouped CV)
  - B score: reader verdict on setting 1 vs B's answer, averaged over B's two answer classes
  - Check 1: at the last layer, class-balanced agreement with the model's own setting-1 answer
    must reach 95% of the ceiling
  - Check 2: B score on the identity prompt (B's answer = A's there) must be >= 80% (trusted layer)
  - chance band: 100 readers on shuffled labels (2.5-97.5%); bootstrap: 1000 resamples (ours: counts)

    python repro/probing/analyze.py --model qwen3vl [--marked]
Writes $PP_RESULTS/probe/<model>/analysis[_marked].json and prints a summary.
--marked: every setting on the outlined photos (extract_*.py --marked).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import PP_RESULTS

N_PCA, N_PERM, N_BOOT, FOLDS = 256, 100, 1000, 5
CHECK1_FRAC, CHECK2_MIN = 0.95, 0.80
rng = np.random.default_rng(0)


def load(model, set_name, setting):
    d = PP_RESULTS / "probe" / model / set_name
    rows = [json.loads(l) for l in open(d / f"{setting}.jsonl")]
    return np.load(d / f"{setting}.npy", mmap_mode="r"), rows


def balanced(pred, ref):
    """Mean over ref's classes of P(pred == ref | class); nan if a class is missing."""
    vals = [np.mean(pred[ref == c] == c) for c in (0, 1) if np.any(ref == c)]
    return float(np.mean(vals)) if len(vals) == 2 else float("nan")


def enc(words):
    return np.array([1 if w == "left" else 0 for w in words])


class Reader:
    """Standardize + PCA + class-weighted logistic regression for one layer."""

    def __init__(self, X):
        self.scaler = StandardScaler().fit(X)
        self.pca = PCA(n_components=min(N_PCA, X.shape[0] - 1, X.shape[1]), random_state=0).fit(self.scaler.transform(X))

    def z(self, X):
        return self.pca.transform(self.scaler.transform(X))

    @staticmethod
    def lr(Z, y):
        return LogisticRegression(class_weight="balanced", max_iter=3000, C=1.0).fit(Z, y)


def ceiling(X, y, groups):
    """Held-out agreement with the model's own answers, CV grouped by room."""
    preds = np.empty_like(y)
    for tr, te in GroupKFold(FOLDS).split(X, y, groups):
        r = Reader(X[tr])
        preds[te] = Reader.lr(r.z(X[tr]), y[tr]).predict(r.z(X[te]))
    return balanced(preds, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--marked", action="store_true")
    args = ap.parse_args()
    m = args.model
    sfx = "_marked" if args.marked else ""

    # ---- training prompts: labels are the model's own answers; invalid answers dropped ----
    def train_set(settings):
        Xs, ys, gs = [], [], []
        for s in settings:
            X, rows = load(m, "train", s + sfx)
            keep = [i for i, r in enumerate(rows) if r["answer"] in ("left", "right")]
            Xs.append(np.asarray(X[keep], dtype=np.float32))
            ys.append(enc([rows[i]["answer"] for i in keep]))
            gs += [rows[i]["scene"] for i in keep]
        return np.concatenate(Xs), np.concatenate(ys), np.array(gs)

    Xd, yd, gd = train_set(["s4", "s2"])
    Xi, yi, gi = train_set(["s1_identity"])
    X1, rows1 = load(m, "test", "s1" + sfx)
    Xid, rowsid = load(m, "test", "s1_identity" + sfx)
    relA, relB = enc([r["relA"] for r in rows1]), enc([r["relB"] for r in rows1])
    ans1 = np.array([1 if r["answer"] == "left" else 0 if r["answer"] == "right" else -1 for r in rows1])
    valid1 = ans1 >= 0
    correct1 = ans1 == relB
    relA_id = enc([r["relA"] for r in rowsid])
    n_layers = X1.shape[1]
    print(f"{m}{sfx}: {n_layers} layer outputs; train direct {len(yd)} (left {yd.mean():.0%}), "
          f"train identity {len(yi)}, test {len(rows1)}; model correct on s1: {correct1.mean():.0%}")

    # consistency with the Section-2 eval answers
    eval_path = PP_RESULTS / m / "test" / f"s1{sfx}.jsonl"
    if eval_path.exists():
        ev = {json.loads(l)["id"]: json.loads(l)["answer"] for l in open(eval_path)}
        same = np.mean([ev.get(r["id"]) == r["answer"] for r in rows1])
        print(f"  setting-1 answers identical to the Section-2 eval: {same:.1%}")

    out = {"model": m, "marked": args.marked, "n_layers": n_layers, "layers": []}
    for L in range(n_layers):
        rec = {"layer": L}
        x1 = np.asarray(X1[:, L], dtype=np.float32)
        xid = np.asarray(Xid[:, L], dtype=np.float32)
        for name, X, y, g in (("direct", Xd[:, L], yd, gd), ("identity", Xi[:, L], yi, gi)):
            r = Reader(X)
            clf = Reader.lr(r.z(X), y)
            v1, vid = clf.predict(r.z(x1)), clf.predict(r.z(xid))
            rec[name] = {
                "ceiling": ceiling(X, y, g),
                "check1": balanced(v1[valid1], ans1[valid1]),  # agreement with the model's s1 answer
                "check2": balanced(vid, relA_id),               # identity: B's answer = A's
                "b_score": balanced(v1, relB),
                "left_rate": float(v1.mean()),
                "b_right": balanced(v1[correct1], relB[correct1]),
                "b_wrong": balanced(v1[~correct1], relB[~correct1]),
                "two_by_two": {f"{'correct' if c else 'wrong'}_{'left' if w else 'right'}":
                               (float(np.mean(v1[(correct1 == c) & (relB == w)] == w))
                                if np.any((correct1 == c) & (relB == w)) else None)
                               for c in (True, False) for w in (1, 0)},
            }
            if name == "direct":
                # bootstrap over test examples, and shuffled-label chance band (PCA reused)
                idx = rng.integers(0, len(v1), size=(N_BOOT, len(v1)))
                boots = [balanced(v1[i], relB[i]) for i in idx]
                rec[name]["b_ci"] = [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))]
                Z, z1 = r.z(X), r.z(x1)
                perm = [balanced(Reader.lr(Z, rng.permutation(y)).predict(z1), relB) for _ in range(N_PERM)]
                rec[name]["chance"] = [float(np.percentile(perm, 2.5)), float(np.percentile(perm, 97.5))]
        out["layers"].append(rec)
        d = rec["direct"]
        print(f"  L{L:2d} ceil {d['ceiling']:.2f} chk1 {d['check1']:.2f} chk2 {d['check2']:.2f} "
              f"B {d['b_score']:.2f} [{d['b_ci'][0]:.2f},{d['b_ci'][1]:.2f}] chance "
              f"[{d['chance'][0]:.2f},{d['chance'][1]:.2f}] left {d['left_rate']:.2f} | "
              f"id-reader chk1 {rec['identity']['check1']:.2f} B {rec['identity']['b_score']:.2f}", flush=True)

    # ---- summary: checks and outcome over trusted layers ----
    last = out["layers"][-1]
    summary = {}
    for name in ("direct", "identity"):
        s = last[name]
        summary[name] = {"check1_last": s["check1"], "ceiling_last": s["ceiling"],
                         "check1_pass": s["check1"] >= CHECK1_FRAC * s["ceiling"]}
    trusted = [l for l in out["layers"] if l["direct"]["check2"] >= CHECK2_MIN]
    below = [l["layer"] for l in trusted if l["direct"]["b_score"] < l["direct"]["chance"][0]]
    above = [l["layer"] for l in trusted if l["direct"]["b_score"] > l["direct"]["chance"][1]]
    inside = [l["layer"] for l in trusted if l["layer"] not in below + above]
    summary["trusted_layers"] = [l["layer"] for l in trusted]
    summary["holds_A"], summary["holds_B"], summary["near_chance"] = below, above, inside
    out["summary"] = summary
    print(f"\nCheck 1 (last layer): direct {summary['direct']['check1_last']:.0%} vs bar "
          f"{CHECK1_FRAC * summary['direct']['ceiling_last']:.0%} -> {'pass' if summary['direct']['check1_pass'] else 'FAIL'}; "
          f"identity reader {summary['identity']['check1_last']:.0%} vs bar "
          f"{CHECK1_FRAC * summary['identity']['ceiling_last']:.0%} -> {'pass' if summary['identity']['check1_pass'] else 'FAIL'}")
    print(f"Check 2: {len(trusted)} trusted layers {summary['trusted_layers']}")
    print(f"Trusted layers holding A's answer: {len(below)}, near chance: {len(inside)}, holding B's answer: {len(above)}")
    path = PP_RESULTS / "probe" / m / f"analysis{sfx}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
