"""Steering redesign, Phase 4 / Part 2 for BAGEL-7B-MoT (as steer2.py; BAGEL's packed cache as in steer_bagel.py).

BAGEL's "token before the answer" is its first generation step. Per question: build the context (image pass,
prompt pass), then for each pushed layer run that first step once for a batch of K copies of the context, adding
the push to that layer's output. Scores: the four answer words in float32 from the final hidden state, and the
probability on them (from the logits).

    python repro/v4/steer2_bagel.py --limit 8 --tag _smoke
    python repro/v4/steer2_bagel.py
Writes $V4_RESULTS/bagel/steer2/steer<tag>.npz and steer<tag>.json.
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
from common.config import BAGEL_PATH
from eval.run_bagel import load  # also puts the BAGEL repo on sys.path
from v4.config_v4 import V4_DATA, V4_RESULTS
from v4.prompts_v4 import TASK, WORDS
from v4.steer2_common import STRENGTHS, deltas, lean4, load_dirs, steer_layers, summarize2
from data.data_utils import pil_img2rgb  # noqa: E402  (BAGEL repo)
from modeling.bagel.qwen2_navit import NaiveCache  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=str(BAGEL_PATH))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    root = V4_RESULTS / "bagel"
    D, gap, head = load_dirs("bagel")
    layers = steer_layers(D.shape[0], head)
    inf = load(args.path)
    lm = inf.model.language_model
    assert len(lm.model.layers) == D.shape[0] - 1, "layer count mismatch"
    dev = torch.device("cuda:0")
    wid = torch.tensor([inf.tokenizer.encode(w)[0] for w in WORDS], device=dev)
    W = lm.lm_head.weight[wid].detach().float()
    bos = inf.new_token_ids["bos_token_id"]
    delta = {}

    def hook(i):
        def fn(module, a, out):
            d = delta.get(i)
            if d is None:
                return None
            h = out[0] if isinstance(out, tuple) else out
            h = h + d.to(h.dtype)                       # [K, hidden]: one packed row per copy
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return fn

    for i, layer in enumerate(lm.model.layers, start=1):
        layer.register_forward_hook(hook(i))

    def first_step(ctx, K):
        """Float32 scores of the four words and the probability on them, for K copies of the context."""
        cache, n, rope = ctx["past_key_values"], int(ctx["kv_lens"][0]), int(ctx["ropes"][0])
        bc = NaiveCache(cache.num_layers)
        for j in range(cache.num_layers):
            bc.key_cache[j] = cache.key_cache[j].repeat(K, 1, 1)
            bc.value_cache[j] = cache.value_cache[j].repeat(K, 1, 1)
        ar = torch.arange(K, device=dev)
        kv_idx = (ar[:, None] * (n + 1) + torch.arange(n, device=dev)[None]).flatten()
        out = lm.forward_inference(
            packed_query_sequence=lm.model.embed_tokens(torch.full((K,), bos, dtype=torch.long, device=dev)),
            query_lens=torch.ones(K, dtype=torch.int, device=dev),
            packed_query_position_ids=torch.full((K,), rope, dtype=torch.long, device=dev),
            packed_query_indexes=ar * (n + 1) + n,
            past_key_values=bc,
            key_values_lens=torch.full((K,), n, dtype=torch.int, device=dev),
            packed_key_value_indexes=kv_idx,
            update_past_key_values=False, is_causal=True, mode="und")
        h = out.packed_query_sequence.reshape(K, -1)
        z = h.float() @ W.T
        mass = torch.softmax(lm.lm_head(h).float(), -1)[:, wid].sum(-1)
        return z.cpu().numpy(), mass.cpu().numpy()

    qs = [json.loads(l) for l in open(V4_DATA / "test" / "questions_test.jsonl")][:args.limit]
    rows = [json.loads(l) for l in open(root / "test" / "rows.jsonl")][:args.limit]
    assert all(q["qid"] == r["qid"] for q, r in zip(qs, rows))
    N, K = len(qs), 1 + D.shape[1] * len(STRENGTHS)
    Z = np.zeros((N, len(layers), K, 4), np.float32)
    M = np.zeros((N, len(layers), K), np.float32)
    base = np.zeros(N, np.float32)
    t0 = time.time()
    print(f"bagel: {N} test questions, pushing layers {layers} (headline L{head}), {K} rows per step", flush=True)
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        for i, q in enumerate(qs):
            img = inf.vae_transform.resize_transform(pil_img2rgb(Image.open(V4_DATA / "test" / q["img_A"]).convert("RGB")))
            ctx = inf.update_context_image(img, inf.init_gen_context(), vae=False)
            ctx = inf.update_context_text(TASK.format(B=q["B"], X=q["X"], Y=q["Y"]), ctx)
            delta.clear()
            z0, _ = first_step(ctx, 1)
            base[i] = float(lean4(z0[0]))
            sign = 1.0 if q["relB"] == "left" else -1.0
            for li, L in enumerate(layers):
                delta[L] = torch.tensor(deltas(D[L], gap[L], sign), device=dev)
                Z[i, li], M[i, li] = first_step(ctx, K)
                delta.clear()
            if i % 50 == 0 or i == N - 1:
                print(f"  {i + 1}/{N}  {time.time() - t0:.0f}s", flush=True)
    out_dir = root / "steer2"
    np.savez(out_dir / f"steer{args.tag}.npz", z=Z, mass=M, base=base, layers=np.array(layers),
             qids=np.array([q["qid"] for q in qs]))
    summarize2(Z, M, base, rows, layers, head, out_dir / f"steer{args.tag}.json")


if __name__ == "__main__":
    main()
