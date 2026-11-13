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


# ---------- Part 2 (Phase 4): the steering run ----------

STRENGTHS = [1.0, 4.0, 8.0, 16.0, 32.0]               # multiples of B's (plain) gap at the pushed layer
N_RAND = 10
DIR_NAMES = ["B", "B_perp", "A_perp", "D_perp", "g"] + [f"rand_{i}" for i in range(N_RAND)]
BROKEN = 0.5                                          # less than half the probability on left/Left/right/Right


def steer_layers(n_states, head):
    """Every second hidden-state index from 2, plus the headline layer; never the last (it can be after the final norm)."""
    return sorted(set(range(2, n_states - 1, 2)) | {int(head)})


def load_dirs(model):
    """Directions from steer2_dirs.py build: [layer, name, H] in DIR_NAMES order, plus B's gap per layer."""
    z = np.load(V4_RESULTS / model / "steer2" / "dirs.npz")
    D = np.stack([z[n] for n in DIR_NAMES[:5]] + [z["rand"][:, i] for i in range(N_RAND)], axis=1)
    return D, z["gap"], int(z["headline_layer"])


def deltas(D_l, gap_l, sign, strengths=STRENGTHS):
    """Rows: [unpushed] + [direction d at strength k, for d in DIR_NAMES, k in strengths]. sign = +1 if B's answer is left."""
    rows = [np.zeros(D_l.shape[1], np.float32)]
    for d in D_l:
        for k in strengths:
            rows.append(sign * k * gap_l * d)
    return np.stack(rows).astype(np.float32)


def lse2(a, b):
    m = np.maximum(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))


def lean4(z):
    """z [..., 4] float32 scores for left/Left/right/Right -> lean."""
    return lse2(z[..., 0], z[..., 1]) - lse2(z[..., 2], z[..., 3])


def photo_boots(rows, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    photos = {}
    for i, r in enumerate(rows):
        photos.setdefault(r["photo"], []).append(i)
    plist = list(photos.values())
    return [np.concatenate([plist[j] for j in rng.integers(0, len(plist), len(plist))]) for _ in range(n)]


def summarize2(z, mass, base_full, rows, layers, head, path, strengths=STRENGTHS):
    """z [N, nL, K, 4] float32 word scores, mass [N, nL, K] probability on the four words, base_full [N] lean from
    a full forward pass. Shift = (pushed lean - unpushed lean) toward each question's B answer."""
    sB = np.array([1.0 if r["relB"] == "left" else -1.0 for r in rows])
    flip = np.array([bool(r["flip"]) for r in rows])
    lean = lean4(z)                                              # [N, nL, K]
    shift = (lean - lean[:, :, :1]) * sB[:, None, None]
    givesB = np.sign(lean) == sB[:, None, None]
    broken = mass < BROKEN
    boots = photo_boots(rows)
    check = {"cached_step_matches_full_pass": float(np.mean(np.sign(lean[:, :, 0]) == np.sign(base_full)[:, None])),
             "max_lean_diff": float(np.abs(lean[:, :, 0] - base_full[:, None]).max())}
    out = {"layers": layers, "names": DIR_NAMES, "strengths": strengths, "headline_layer": int(head), "n": len(rows),
           "n_flip": int(flip.sum()), "check": check,
           "baseline_gives_B_flip": float(givesB[flip, 0, 0].mean()), "per_layer": []}
    for li, L in enumerate(layers):
        ent = {"layer": L, "dirs": {}}
        for di, n in enumerate(DIR_NAMES):
            ent["dirs"][n] = []
            for ki, k in enumerate(strengths):
                j = 1 + di * len(strengths) + ki
                s = shift[:, li, j]
                bv = [s[b].mean() for b in boots]
                ent["dirs"][n].append({"k": k, "shift": float(s.mean()),
                                       "ci": [float(np.percentile(bv, 2.5)), float(np.percentile(bv, 97.5))],
                                       "shift_flip": float(s[flip].mean()), "shift_noflip": float(s[~flip].mean()),
                                       "gives_B_flip": float(givesB[flip, li, j].mean()),
                                       "broken": float(broken[:, li, j].mean())})
        out["per_layer"].append(ent)
    # decision at the headline layer (PLAN_V4_STEER.md)
    hl = next(e for e in out["per_layer"] if e["layer"] == head)
    dec = {}
    for k in (16.0, 32.0):
        ki = strengths.index(k)
        bp = hl["dirs"]["B_perp"][ki]
        rnd = [hl["dirs"][f"rand_{i}"][ki]["shift"] for i in range(N_RAND)]
        dec[str(k)] = {"B_perp": bp["shift"], "B_perp_ci": bp["ci"], "rand_max": float(max(rnd)),
                       "rand_mean": float(np.mean(rnd)), "D_perp": hl["dirs"]["D_perp"][ki]["shift"],
                       "broken": bp["broken"],
                       "pass": bool(bp["shift"] > max(rnd) and bp["ci"][0] > np.mean(rnd)
                                    and bp["shift"] > hl["dirs"]["D_perp"][ki]["shift"] and bp["broken"] < 0.10)}
    out["decision"] = {"by_strength": dec, "content_acted_on": all(v["pass"] for v in dec.values())}
    path.write_text(json.dumps(out, indent=1))

    print(f"check: cached step = full pass (sign of the lean) {check['cached_step_matches_full_pass']:.1%}, "
          f"max |lean difference| {check['max_lean_diff']:.3f}")
    print(f"baseline: gives B's answer on flip questions {out['baseline_gives_B_flip']:.0%} (n={out['n_flip']})")
    named = ["B", "B_perp", "A_perp", "D_perp", "g"]
    for ki, k in enumerate(strengths):
        print(f"\nshift in the lean toward B's answer at {k:g}x B's gap (logits; broken % if any):")
        print("  layer " + " ".join(f"{n:>13s}" for n in named) + "   rand mean [max]")
        for e in out["per_layer"]:
            cells = []
            for n in named:
                c = e["dirs"][n][ki]
                cells.append(f"{c['shift']:+7.3f}" + (f"({c['broken']:.0%})" if c["broken"] >= 0.01 else "     "))
            rnd = [e["dirs"][f"rand_{i}"][ki]["shift"] for i in range(N_RAND)]
            mark = "*" if e["layer"] == head else " "
            print(f" {mark}L{e['layer']:2d} " + " ".join(f"{c:>13s}" for c in cells) +
                  f"   {np.mean(rnd):+.3f} [{max(rnd):+.3f}]", flush=True)
    for k, v in dec.items():
        print(f"decision at L{head}, {k}x: B_perp {v['B_perp']:+.3f} [{v['B_perp_ci'][0]:+.3f}, {v['B_perp_ci'][1]:+.3f}], "
              f"random max {v['rand_max']:+.3f}, mean {v['rand_mean']:+.3f}, depth {v['D_perp']:+.3f}, "
              f"broken {v['broken']:.0%} -> {'pass' if v['pass'] else 'fail'}")
    print("B's content acted on (both strengths pass):", out["decision"]["content_acted_on"])
    return out
