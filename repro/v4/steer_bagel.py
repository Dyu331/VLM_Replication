"""Phase 6c for BAGEL-7B-MoT: the same steering as steer.py, through BAGEL's packed cache.

BAGEL's "token before the answer" is its first generation step (the start token after the image and prompt
passes; extract_bagel.py saved its states). Per question: build the context (image pass, prompt pass), then
for each pushed layer run that first step once for a batch of K copies of the context (BAGEL's attention
takes several packed sequences; the cache is not updated), adding the push to that layer's output.

    python repro/v4/steer_bagel.py --limit 8 --tag _smoke
    python repro/v4/steer_bagel.py
Writes $V4_RESULTS/bagel/steer<tag>.npz and steer<tag>.json.
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
from v4.steer_common import deltas, steer_layers, summarize
from data.data_utils import pil_img2rgb  # noqa: E402  (BAGEL repo)
from modeling.bagel.qwen2_navit import NaiveCache  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=str(BAGEL_PATH))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    root = V4_RESULTS / "bagel"
    dz = np.load(root / "directions.npz")
    names, dirs, gap, best = list(dz["names"]), dz["dirs"], dz["gap"], int(dz["best_layer"])
    layers = steer_layers(dirs.shape[0], best)

    inf = load(args.path)
    lm = inf.model.language_model
    assert len(lm.model.layers) == dirs.shape[0] - 1, "layer count mismatch"
    dev = torch.device("cuda:0")
    wid = torch.tensor([inf.tokenizer.encode(w)[0] for w in WORDS], device=dev)
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
        """Log-probs of the first answer word for K copies of the context (pushes from `delta`)."""
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
        return torch.log_softmax(lm.lm_head(out.packed_query_sequence).float(), -1)[:, wid]

    qs = [json.loads(l) for l in open(V4_DATA / "test" / "questions_test.jsonl")][:args.limit]
    rows = [json.loads(l) for l in open(root / "test" / "rows.jsonl")][:args.limit]
    assert all(q["qid"] == r["qid"] for q, r in zip(qs, rows))
    N, K = len(qs), 1 + len(names) * 4
    lp = np.zeros((N, len(layers), K, 4), np.float32)
    base = np.zeros((N, 4), np.float32)
    t0 = time.time()
    print(f"bagel: {N} test questions, pushing layers {layers}, {K} rows per step", flush=True)
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        for i, q in enumerate(qs):
            img = inf.vae_transform.resize_transform(pil_img2rgb(Image.open(V4_DATA / "test" / q["img_A"]).convert("RGB")))
            ctx = inf.init_gen_context()
            ctx = inf.update_context_image(img, ctx, vae=False)
            ctx = inf.update_context_text(TASK.format(B=q["B"], X=q["X"], Y=q["Y"]), ctx)
            delta.clear()
            base[i] = first_step(ctx, 1)[0].cpu().numpy()      # unpushed reference, batch of one
            sign = 1.0 if q["relB"] == "left" else -1.0
            for li, L in enumerate(layers):
                delta[L] = torch.tensor(deltas(dirs[L], gap[L], sign), device=dev)
                lp[i, li] = first_step(ctx, K).cpu().numpy()
                delta.clear()
            if i % 50 == 0 or i == N - 1:
                print(f"  {i + 1}/{N}  {time.time() - t0:.0f}s", flush=True)
    np.savez(root / f"steer{args.tag}.npz", lp=lp, base=base, layers=np.array(layers), names=np.array(names),
             qids=np.array([q["qid"] for q in qs]))
    summarize(lp, base, rows, layers, names, best, root / f"steer{args.tag}.json")


if __name__ == "__main__":
    main()
