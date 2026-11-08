"""Phase 6b (follow-up, violet): logit lens at the token before the answer, plus the mental-rotation
check (kept in the background unless it shows a clear pattern).

Logit lens: each layer's test state -> the model's final norm -> the output rows for left/Left/right/Right.
  "left" score = logsumexp(left, Left), "right" likewise; gap = left - right (Gemma: final softcap applied).
  Per layer: how often the lens says B's answer on flip questions (where B and A differ), A's answer,
  the model's final answer; the mean gap signed toward B's answer.
  Angles: cosine between the output's left-minus-right read-out direction (norm weight x row difference)
  and each steering direction (directions.npz); and the linear logit change from a 1x push along it.
  The last layer is checked against the saved first-token scores (some models store it already normed).

Rotation check: the B-answer reader at the best layer (refit as in analyze.py), scored within A, grouped by
the turn between A's and B's headings (0-60, 60-120, 120-180 degrees); also averaged over B-to-line distance
terciles inside each group, since a big turn tends to go with B being far across the line.

    python repro/v4/lens.py --model qwen3vl
Writes $V4_RESULTS/<model>/lens.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v4.analyze import N_PCA, Data, group_scores, lr, within_a
from v4.config_v4 import V4_DATA, V4_RESULTS

EPS = 1e-6


def softcap(model):
    if model == "bagel":
        return None
    try:
        from eval.run_vlm import MODELS
        cfg = json.loads((Path(MODELS[model][0]) / "config.json").read_text())
        return cfg.get("text_config", cfg).get("final_logit_softcapping")
    except Exception as e:  # noqa: BLE001
        print("softcap lookup failed:", e)
        return None


def lse2(a, b):
    m = np.maximum(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))


def lens_gap(H, w, U, cap, norm=True):
    """H [N, hidden] -> left-minus-right score gap per question."""
    if norm:
        H = H / np.sqrt((H ** 2).mean(1, keepdims=True) + EPS) * w
    z = H @ U.T
    if cap:
        z = cap * np.tanh(z / cap)
    return lse2(z[:, 0], z[:, 1]) - lse2(z[:, 2], z[:, 3])


def rotation(D, model, best, C):
    S_tr = np.load(V4_RESULTS / model / "train" / "states.npy", mmap_mode="r")
    S_te = np.load(V4_RESULTS / model / "test" / "states.npy", mmap_mode="r")
    Xtr, Xte = (np.asarray(S[0, best], np.float32) for S in (S_tr, S_te))
    sc = StandardScaler().fit(Xtr)
    p = PCA(min(N_PCA, len(Xtr) - 1), random_state=0).fit(sc.transform(Xtr))
    pred = lr(C).fit(p.transform(sc.transform(Xtr)), D.y["train"]["B"]).predict(p.transform(sc.transform(Xte)))
    q = {r["qid"]: r for r in map(json.loads, open(V4_DATA / "test" / "questions_test.jsonl"))}
    turn = lambda r: abs((q[r["qid"]]["A_yaw"] - q[r["qid"]]["B_yaw"] + 180) % 360 - 180)
    bins = ["0-60", "60-120", "120-180"]
    bin_of = lambda r: bins[min(int(turn(r) // 60), 2)]
    out = {"layer": best, "by_turn": group_scores(D, pred, bin_of, bins)}
    rows, y, A = D.rows["test"], D.y["test"]["B"], D.y["test"]["A"]
    bl = np.array([r["dist"]["B_line"] for r in rows])
    terc = np.digitize(bl, np.quantile(bl, [1 / 3, 2 / 3]))
    b = np.array([bin_of(r) for r in rows])
    flip = D.y["test"]["flip"]
    for v in bins:
        parts = [within_a(pred[(b == v) & (terc == t)], y[(b == v) & (terc == t)], A[(b == v) & (terc == t)])
                 for t in range(3) if ((b == v) & (terc == t)).sum() >= 20]
        out["by_turn"][v]["score_line_matched"] = float(np.nanmean(parts)) if parts else None
        out["by_turn"][v]["flip_share"] = float(flip[b == v].mean())
        out["by_turn"][v]["B_line_median"] = float(np.median(bl[b == v]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    root = V4_RESULTS / args.model
    D = Data(args.model)
    rows = D.rows["test"]
    lens = np.load(root / "lens.npz")
    U, w = lens["unembed"].astype(np.float64), lens["norm_weight"].astype(np.float64)
    cap = softcap(args.model)
    S = np.load(root / "test" / "states.npy", mmap_mode="r")
    n_layers = S.shape[1]
    yB, yA, ym, flip = D.y["test"]["B"], D.y["test"]["A"], D.y["test"]["model"], D.y["test"]["flip"].astype(bool)
    sB = np.where(yB == 1, 1.0, -1.0)

    # which form the last layer is stored in: compare with the saved first-token scores
    saved = np.array([lse2(r["scores"]["left"], r["scores"]["Left"]) - lse2(r["scores"]["right"], r["scores"]["Right"])
                      for r in rows])
    last = np.asarray(S[0, n_layers - 1], np.float64)
    err = {f: float(np.abs(lens_gap(last, w, U, cap, norm=f) - saved).mean()) for f in (True, False)}
    last_normed_already = err[False] < err[True]
    print(f"last layer vs saved scores: mean |gap error| with norm {err[True]:.3f}, without {err[False]:.3f} "
          f"-> {'already normed' if last_normed_already else 'apply norm'}; softcap {cap}", flush=True)

    dirs = np.load(root / "directions.npz")
    names = list(dirs["names"])
    readout = w * (U[:2].mean(0) - U[2:].mean(0))   # left-minus-right read-out in the residual stream
    per_layer = []
    for L in range(n_layers):
        H = np.asarray(S[0, L], np.float64)
        normed = not (L == n_layers - 1 and last_normed_already)
        g = lens_gap(H, w, U, cap, norm=normed)
        says_left = g > 0
        rms = np.sqrt((H ** 2).mean(1)).mean()
        d = dirs["dirs"][L].astype(np.float64)
        r = {"layer": L,
             "says_B_on_flip": float((says_left == (yB == 1))[flip].mean()),
             "says_B_on_noflip": float((says_left == (yB == 1))[~flip].mean()),
             "says_A": float((says_left == (yA == 1)).mean()),
             "agrees_with_model": float((says_left == (ym == 1))[ym >= 0].mean()),
             "gap_toward_B_flip": float((g * sB)[flip].mean()),
             "gap_toward_B_noflip": float((g * sB)[~flip].mean()),
             "cos_readout": {n: float(d[i] @ readout / np.linalg.norm(readout)) for i, n in enumerate(names[:4])},
             # linear estimate of the logit-gap change from a push of 1x B's gap (ignores the norm's rescaling)
             "lin_push_1x": {n: float(dirs["gap"][L] * (d[i] @ readout) / rms) for i, n in enumerate(names[:4])}}
        per_layer.append(r)

    an = json.loads((root / "analysis.json").read_text())
    best = an["best_layer"]
    C = next(r["targets"]["B"]["C"] for r in an["layers"] if r["layer"] == best)
    rot = rotation(D, args.model, best, C)
    out = {"model": args.model, "softcap": cap, "last_layer_check": err, "last_normed_already": bool(last_normed_already),
           "layers": per_layer, "rotation": rot}
    (root / "lens.json").write_text(json.dumps(out, indent=1))

    print(f"{args.model} logit lens (test; flip questions = where B's answer differs from A's):")
    for r in per_layer:
        c = r["cos_readout"]
        print(f"  L{r['layer']:2d} lens says B on flip {r['says_B_on_flip']:.0%} | says A {r['says_A']:.0%} | "
              f"agrees w/ final {r['agrees_with_model']:.0%} | gap->B flip {r['gap_toward_B_flip']:+.2f} | "
              f"cos(readout): B {c['B']:+.3f} normal {c['B_normal']:+.3f} B-say {c['B_minus_say']:+.3f} "
              f"say {c['say']:+.3f}", flush=True)
    print(f"rotation check at L{best}:")
    for v, s in rot["by_turn"].items():
        print(f"  turn {v:>7s}: B reader {s['score']:.0%} [{s['ci'][0]:.0%}-{s['ci'][1]:.0%}] n={s['n']} | "
              f"line-matched {s['score_line_matched'] if s['score_line_matched'] is None else format(s['score_line_matched'], '.0%')} | "
              f"flip share {s['flip_share']:.0%} | median B-line {s['B_line_median']:.2f} m")


if __name__ == "__main__":
    main()
