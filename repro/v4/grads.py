"""Steering redesign, Phase 2 (PLAN_V4_STEER.md): gradients of the final lean, Qwen3-VL and Gemma 4.

Per question, one forward and one backward pass on the task prompt (image + text, as in extract.py). The lean is
LSE(left, Left) - LSE(right, Right) of the final logits at the token before the answer. Hooks keep the gradient
of the lean with respect to the hidden state at every layer (index 0 = input embeddings, index i = output of
decoder layer i, the same indexing as states.npy and the steering hooks) at three positions: the token before the
answer, the last mention of X, the last mention of Y. Weights get no gradients.

Finite-difference check (--fd N, on the first N test questions): push the hidden state at the token before the
answer at a few layers and compare the real change in the lean with the gradient's prediction, along the
question's own gradient (predicted change +-0.25 and +-1) and along B's direction (1x, 4x, 16x B's gap).

    python repro/v4/grads.py --model qwen3vl --limit 8 --fd 8 --tag _smoke
    python repro/v4/grads.py --model qwen3vl --fd 16
Writes $V4_RESULTS/<model>/steer2<tag>/<split>/{grads.npy [3, L+1, N, H] float32, lean.npy [N]} and fd_check.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v4.config_v4 import V4_DATA, V4_RESULTS
from v4.extract import Runner
from v4.prompts_v4 import TASK
from v4.steer2_common import FD_B_MULT, FD_TARGETS, GRAD_POS, b_direction, fd_layers, fd_summary, headline


class Hooks:
    """grab: keep the input embeddings (made a leaf that needs a gradient) and every decoder layer's output.
    push: add self.delta[i] to the output of layer i at the last position (no-grad passes)."""

    def __init__(self, model):
        self.grab, self.outs, self.delta = False, {}, {}
        model.get_input_embeddings().register_forward_hook(self._emb)
        for i, layer in enumerate(model.model.language_model.layers, start=1):
            layer.register_forward_hook(self._post(i))

    def _emb(self, module, args, out):
        if self.grab and 0 not in self.outs:
            out = out.detach().requires_grad_(True)
            self.outs[0] = out
            return out
        return None

    def _post(self, i):
        def fn(module, args, out):
            h = out[0] if isinstance(out, tuple) else out
            if self.grab:
                h.retain_grad()
                self.outs[i] = h
                return None
            d = self.delta.get(i)
            if d is None:
                return None
            h = h.clone()
            h[:, -1, :] += d.to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return fn


def lean_of(logits, wid):
    z = logits.float()
    return torch.logsumexp(z[wid[:2]], 0) - torch.logsumexp(z[wid[2:]], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["qwen3vl", "gemma4"])
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fd", type=int, default=16, help="finite-difference check on the first N test questions")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    R = Runner(args.model)
    R.model.requires_grad_(False)
    dev = R.model.device
    wid = torch.tensor(R.word_ids, device=dev)
    hk = Hooks(R.model)
    n_layers = len(R.model.model.language_model.layers)
    root = V4_RESULTS / args.model / f"steer2{args.tag}"

    def prompt(split, q):
        img = Image.open(V4_DATA / split / q["img_A"]).convert("RGB")
        return R.inputs(TASK.format(B=q["B"], X=q["X"], Y=q["Y"]), img)

    def grad_pass(inp, q):
        hk.grab, hk.outs = True, {}
        with torch.enable_grad():
            out = R.model(**inp, use_cache=False)
            hk.grab = False
            lean = lean_of(out.logits[0, -1], wid)
            lean.backward()
        pos = [inp["input_ids"].shape[1] - 1, R.name_pos(inp, q["X"], last=True), R.name_pos(inp, q["Y"], last=True)]
        g = torch.stack([hk.outs[i].grad[0, pos] for i in range(n_layers + 1)], 1)   # [3, L+1, H]
        res = g.float().cpu().numpy(), float(lean.detach())
        hk.outs = {}
        del out
        return res

    for split in args.splits:
        qs = [json.loads(l) for l in open(V4_DATA / split / f"questions_{split}.jsonl")][:args.limit]
        rows = [json.loads(l) for l in open(V4_RESULTS / args.model / split / "rows.jsonl")][:args.limit]
        assert all(q["qid"] == r["qid"] for q, r in zip(qs, rows))
        out_dir = root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        N, G, leans, t0 = len(qs), None, np.zeros(len(qs), np.float32), time.time()
        for i, q in enumerate(qs):
            g, leans[i] = grad_pass(prompt(split, q), q)
            if G is None:
                G = np.lib.format.open_memmap(out_dir / "grads.npy", "w+", np.float32, (len(GRAD_POS), g.shape[1], N, g.shape[2]))
            G[:, :, i] = g
            saved = rows[i]["scores"]
            if i < 3:
                ref = np.logaddexp(saved["left"], saved["Left"]) - np.logaddexp(saved["right"], saved["Right"])
                print(f"  check q{i}: lean here {leans[i]:+.3f}, from the saved answer scores {ref:+.3f}", flush=True)
            if i % 100 == 0 or i == N - 1:
                print(f"{args.model} {split} {i + 1}/{N}  {time.time() - t0:.0f}s  |grad| at the answer token, "
                      f"layers 10/20/last-1: " + " ".join(f"{np.linalg.norm(g[0, L]):.2e}" for L in (10, 20, n_layers - 1)),
                      flush=True)
        G.flush()
        np.save(out_dir / "lean.npy", leans)

    if args.fd:
        qs = [json.loads(l) for l in open(V4_DATA / "test" / "questions_test.jsonl")][:args.fd]
        layers = fd_layers(n_layers, headline(args.model))
        bdirs = {L: b_direction(args.model, L) for L in layers}
        rng = np.random.default_rng(0)
        records = []
        for i, q in enumerate(qs):
            inp = prompt("test", q)
            g, lean0_grad = grad_pass(inp, q)
            with torch.no_grad():
                hk.delta = {}
                lean0 = float(lean_of(R.model(**inp, use_cache=False).logits[0, -1], wid))
                for L in layers:
                    gl = g[0, L].astype(np.float64)
                    tests = []
                    for t in FD_TARGETS:
                        for s in (1, -1):
                            tests.append(("own_grad", s * t, s * t / (gl @ gl) * gl))      # predicted change s*t
                    u, gap = bdirs[L]
                    sign = 1.0 if q["relB"] == "left" else -1.0
                    for k in FD_B_MULT:
                        tests.append(("B", k, sign * k * gap * u))
                    for kind, size, d in tests:
                        hk.delta = {L: torch.tensor(d, dtype=torch.float32, device=dev)[None]}
                        act = float(lean_of(R.model(**inp, use_cache=False).logits[0, -1], wid)) - lean0
                        hk.delta = {}
                        records.append({"q": i, "layer": L, "kind": kind, "size": size,
                                        "pred": float(gl @ d), "actual": act})
            print(f"  fd {i + 1}/{len(qs)} (lean no-grad {lean0:+.3f}, with grad {lean0_grad:+.3f})", flush=True)
        root.mkdir(parents=True, exist_ok=True)
        fd_summary(records, root / "fd_check.json")


if __name__ == "__main__":
    main()
