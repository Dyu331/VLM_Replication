"""Shared by steer.py (Qwen3-VL, Gemma 4) and steer_bagel.py: numpy only, so both envs can import it.

Design (agreed with the user, 2026-11-07):
  push at the token before the answer, at one layer at a time (every 2nd layer below the last, plus the
  B-reader's best layer), toward each question's B answer;
  directions from directions.npz (training rooms), all scaled to the SAME length: k x B's gap at that layer,
  k in STRENGTHS; plus one unpushed row (the baseline);
  the answer is read at the first answer word only: left = logsumexp(left, Left) vs right likewise
  (the user's option i). "Broken" = less than half the probability on the four words.
"""
import json

import numpy as np

STRENGTHS = [0.5, 1.0, 2.0, 4.0]


def steer_layers(n_states, best):
    """Hidden-state indices to push (index l = output of decoder layer l). The last is skipped: some models
    store it after the final norm, so a direction from it would not match the residual stream."""
    return sorted(set(range(2, n_states - 1, 2)) | {int(best)})


def deltas(dirs_l, gap_l, sign, strengths=STRENGTHS):
    """Rows: [unpushed] + [direction d at strength k for d, k]. sign = +1 if B's answer is left, else -1."""
    rows = [np.zeros(dirs_l.shape[1], np.float32)]
    for d in dirs_l:
        for k in strengths:
            rows.append(sign * k * gap_l * d)
    return np.stack(rows)


def lse2(a, b):
    m = np.maximum(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))


def summarize(lp, base_full, rows, layers, names, best, path, strengths=STRENGTHS):
    """lp [N, n_layers, rows, 4] log-probs of left/Left/right/Right; base_full [N, 4] from the full forward pass."""
    relB = np.array([r["relB"] == "left" for r in rows])
    relA = np.array([r["relA"] == "left" for r in rows])
    flip = np.array([bool(r["flip"]) for r in rows])
    says_left = lse2(lp[..., 0], lp[..., 1]) > lse2(lp[..., 2], lp[..., 3])          # [N, nL, rows]
    mass = np.exp(lp).sum(-1)
    gives_B = says_left == relB[:, None, None]

    full_left = lse2(base_full[:, 0], base_full[:, 1]) > lse2(base_full[:, 2], base_full[:, 3])
    saved = np.array([r["answer"] == "left" for r in rows])
    valid = np.array([r["answer"] in ("left", "right") for r in rows])
    check = {"cached_step_matches_full_pass": float((says_left[:, :, 0] == full_left[:, None]).mean()),
             "full_pass_matches_saved_answer": float((full_left == saved)[valid].mean()),
             "max_gap_diff": float(np.abs((lse2(lp[:, :, 0, 0], lp[:, :, 0, 1]) - lse2(lp[:, :, 0, 2], lp[:, :, 0, 3]))
                                          - (lse2(base_full[:, 0], base_full[:, 1]) - lse2(base_full[:, 2], base_full[:, 3]))[:, None]).max())}
    out = {"layers": layers, "names": names, "strengths": strengths, "best_layer": int(best), "n": len(rows),
           "n_flip": int(flip.sum()), "check": check,
           "baseline": {"gives_B_flip": float(gives_B[flip, 0, 0].mean()), "gives_B_noflip": float(gives_B[~flip, 0, 0].mean()),
                        "broken": float((mass[:, 0, 0] < .5).mean())},
           "per_layer": []}
    for li, L in enumerate(layers):
        ent = {"layer": L, "dirs": {}}
        for di, n in enumerate(names):
            ent["dirs"][n] = []
            for ki, k in enumerate(strengths):
                j = 1 + di * len(strengths) + ki
                ent["dirs"][n].append({"k": k, "gives_B_flip": float(gives_B[flip, li, j].mean()),
                                       "gives_B_noflip": float(gives_B[~flip, li, j].mean()),
                                       "gives_A_flip": float((says_left[flip, li, j] == relA[flip]).mean()),
                                       "broken": float((mass[:, li, j] < .5).mean())})
        out["per_layer"].append(ent)
    path.write_text(json.dumps(out, indent=1))

    print(f"check: cached step = full pass {check['cached_step_matches_full_pass']:.1%}; full pass = saved answer "
          f"{check['full_pass_matches_saved_answer']:.1%}; max gap diff {check['max_gap_diff']:.3f}")
    print(f"baseline: gives B's answer on flip questions {out['baseline']['gives_B_flip']:.0%} "
          f"(n={out['n_flip']}), no-flip {out['baseline']['gives_B_noflip']:.0%}")
    rnd = [n for n in names if n.startswith("random")]
    head = "  layer " + " ".join(f"{n[:11]:>11s}" for n in names if not n.startswith("random")) + "     random"
    for ki, k in enumerate(strengths):
        print(f"gives B's answer on flip questions at {k}x B's gap (broken %):\n{head}")
        for ent in out["per_layer"]:
            cells = [f"{ent['dirs'][n][ki]['gives_B_flip']:6.0%} ({ent['dirs'][n][ki]['broken']:2.0%})"
                     for n in names if not n.startswith("random")]
            r = np.mean([ent["dirs"][n][ki]["gives_B_flip"] for n in rnd])
            mark = "*" if ent["layer"] == best else " "
            print(f"  {mark}L{ent['layer']:2d} " + " ".join(f"{c:>11s}" for c in cells) + f" {r:9.0%}", flush=True)
    return out
