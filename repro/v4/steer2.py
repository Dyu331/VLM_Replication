"""Steering redesign, Phase 4 / Part 2 (PLAN_V4_STEER.md): Qwen3-VL and Gemma 4, test questions.

Per question: one full forward pass (the unpushed reference), one pass over the prompt minus its last token (kept
in the cache), then for each pushed layer one batched step over the last prompt token: the unpushed row plus
15 directions x 5 strengths (DIR_NAMES: B, B_perp, A_perp, D_perp, g, 10 random), each push added to that
layer's output at that token, all the same length (k x B's gap), toward each question's B answer.
Scores: the four answer words in float32 from the final hidden state (the lean), and the probability on them
(from the logits) for the "broken" check.

    python repro/v4/steer2.py --model qwen3vl --limit 8 --tag _smoke
    python repro/v4/steer2.py --model qwen3vl
Writes $V4_RESULTS/<model>/steer2/steer<tag>.npz and steer<tag>.json.
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
from v4.steer2_common import STRENGTHS, deltas, lean4, load_dirs, steer_layers, summarize2


class Hooks:
    """Add self.delta[i] ([batch, H]) to the output of decoder layer i at the last position; keep the input of the
    output layer (the final normed hidden state)."""

    def __init__(self, model):
        self.delta, self.head_in = {}, None
        for i, layer in enumerate(model.model.language_model.layers, start=1):
            layer.register_forward_hook(self._post(i))
        model.get_output_embeddings().register_forward_hook(self._head)

    def _head(self, module, args, out):
        self.head_in = args[0]

    def _post(self, i):
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
    args = ap.parse_args()

    root = V4_RESULTS / args.model
    D, gap, head = load_dirs(args.model)
    layers = steer_layers(D.shape[0], head)
    R = Runner(args.model)
    dev = R.model.device
    hk = Hooks(R.model)
    assert len(R.model.model.language_model.layers) == D.shape[0] - 1, "layer count mismatch"
    wid = torch.tensor(R.word_ids, device=dev)
    W = R.model.get_output_embeddings().weight[wid].detach().float()
    cfg = getattr(R.model.config, "text_config", R.model.config)
    cap = getattr(cfg, "final_logit_softcapping", None)

    def scores():
        z = hk.head_in[:, -1].float() @ W.T
        return cap * torch.tanh(z / cap) if cap else z

    qs = [json.loads(l) for l in open(V4_DATA / "test" / "questions_test.jsonl")][:args.limit]
    rows = [json.loads(l) for l in open(root / "test" / "rows.jsonl")][:args.limit]
    assert all(q["qid"] == r["qid"] for q, r in zip(qs, rows))
    N, K = len(qs), 1 + D.shape[1] * len(STRENGTHS)
    Z = np.zeros((N, len(layers), K, 4), np.float32)
    M = np.zeros((N, len(layers), K), np.float32)
    base = np.zeros(N, np.float32)
    t0 = time.time()
    print(f"{args.model}: {N} test questions, pushing layers {layers} (headline L{head}), {K} rows per step", flush=True)
    with torch.no_grad():
        for i, q in enumerate(qs):
            img = Image.open(V4_DATA / "test" / q["img_A"]).convert("RGB")
            inp = R.inputs(TASK.format(B=q["B"], X=q["X"], Y=q["Y"]), img)
            seq = inp["input_ids"].shape[1]
            hk.delta = {}
            R.model(**inp)
            base[i] = float(lean4(scores()[0].cpu().numpy()))
            pre = {k: (v[:, :-1] if torch.is_tensor(v) and v.dim() == 2 and v.shape[1] == seq else v) for k, v in inp.items()}
            cache = R.model(**pre, use_cache=True).past_key_values
            cache.batch_repeat_interleave(K)
            last = inp["input_ids"][:, -1:].repeat(K, 1)
            sign = 1.0 if q["relB"] == "left" else -1.0
            for li, L in enumerate(layers):
                hk.delta = {L: torch.tensor(deltas(D[L], gap[L], sign), device=dev)}
                out = R.model(input_ids=last, past_key_values=copy.deepcopy(cache), use_cache=True)
                hk.delta = {}
                Z[i, li] = scores().cpu().numpy()
                M[i, li] = torch.softmax(out.logits[:, -1].float(), -1)[:, wid].sum(-1).cpu().numpy()
            if i % 50 == 0 or i == N - 1:
                print(f"  {i + 1}/{N}  {time.time() - t0:.0f}s", flush=True)
    out_dir = root / "steer2"
    np.savez(out_dir / f"steer{args.tag}.npz", z=Z, mass=M, base=base, layers=np.array(layers),
             qids=np.array([q["qid"] for q in qs]))
    summarize2(Z, M, base, rows, layers, head, out_dir / f"steer{args.tag}.json")


if __name__ == "__main__":
    main()
