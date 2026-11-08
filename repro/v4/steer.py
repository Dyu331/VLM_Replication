"""Phase 6c (follow-up, violet): steering for Qwen3-VL and Gemma 4 on the TEST questions (design in
steer_common.py; directions from directions.npz, i.e. from the training rooms).

Per question: one full forward pass (the unpushed reference), one pass over the prompt minus its last token
(kept in the cache), then for each pushed layer one batched step over the last prompt token: the
unpushed row plus 9 directions x 5 strengths, the push added to that layer's output at that token.

    python repro/v4/steer.py --model qwen3vl --limit 8 --tag _smoke
    python repro/v4/steer.py --model qwen3vl
Writes $V4_RESULTS/<model>/steer<tag>.npz and steer<tag>.json.
"""
import argparse
import copy
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
from v4.steer_common import STRENGTHS, deltas, steer_layers, summarize


class Pusher:
    """Forward hooks on the decoder layers: add self.delta[layer] ([batch, hidden]) at the last position."""

    def __init__(self, layers):
        self.delta = {}
        for i, layer in enumerate(layers, start=1):   # hidden-state index i = output of layers[i - 1]
            layer.register_forward_hook(self._hook(i))

    def _hook(self, i):
        def fn(module, args, out):
            d = self.delta.get(i)
            if d is None:
                return None
            h = out[0] if isinstance(out, tuple) else out
            h = h.clone()
            h[:, -1, :] += d.to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["qwen3vl", "gemma4"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--strengths", type=float, nargs="+", default=STRENGTHS)
    args = ap.parse_args()

    root = V4_RESULTS / args.model
    dz = np.load(root / "directions.npz")
    names, dirs, gap, best = list(dz["names"]), dz["dirs"], dz["gap"], int(dz["best_layer"])
    layers = steer_layers(dirs.shape[0], best)
    R = Runner(args.model)
    dev = R.model.device
    push = Pusher(R.model.model.language_model.layers)
    assert len(R.model.model.language_model.layers) == dirs.shape[0] - 1, "layer count mismatch"

    qs = [json.loads(l) for l in open(V4_DATA / "test" / "questions_test.jsonl")][:args.limit]
    rows = [json.loads(l) for l in open(root / "test" / "rows.jsonl")][:args.limit]
    assert all(q["qid"] == r["qid"] for q, r in zip(qs, rows))
    N, K = len(qs), 1 + len(names) * len(args.strengths)
    lp = np.zeros((N, len(layers), K, 4), np.float32)
    base = np.zeros((N, 4), np.float32)
    wid = torch.tensor(R.word_ids, device=dev)
    t0 = time.time()
    print(f"{args.model}: {N} test questions, pushing layers {layers}, {K} rows per step", flush=True)
    with torch.no_grad():
        for i, q in enumerate(qs):
            img = Image.open(V4_DATA / "test" / q["img_A"]).convert("RGB")
            inp = R.inputs(TASK.format(B=q["B"], X=q["X"], Y=q["Y"]), img)
            seq = inp["input_ids"].shape[1]
            full = R.model(**inp).logits[0, -1].float()
            base[i] = torch.log_softmax(full, -1)[wid].cpu().numpy()
            pre = {k: (v[:, :-1] if torch.is_tensor(v) and v.dim() == 2 and v.shape[1] == seq else v)
                   for k, v in inp.items()}
            cache = R.model(**pre, use_cache=True).past_key_values
            cache.batch_repeat_interleave(K)
            last = inp["input_ids"][:, -1:].repeat(K, 1)
            sign = 1.0 if q["relB"] == "left" else -1.0
            for li, L in enumerate(layers):
                push.delta = {L: torch.tensor(deltas(dirs[L], gap[L], sign, args.strengths), device=dev)}
                out = R.model(input_ids=last, past_key_values=copy.deepcopy(cache), use_cache=True)
                push.delta = {}
                lp[i, li] = torch.log_softmax(out.logits[:, -1].float(), -1)[:, wid].cpu().numpy()
            if i % 50 == 0 or i == N - 1:
                print(f"  {i + 1}/{N}  {time.time() - t0:.0f}s", flush=True)
    tag = args.tag
    np.savez(root / f"steer{tag}.npz", lp=lp, base=base, layers=np.array(layers), names=np.array(names),
             qids=np.array([q["qid"] for q in qs]))
    summarize(lp, base, rows, layers, names, best, root / f"steer{tag}.json", args.strengths)


if __name__ == "__main__":
    main()
