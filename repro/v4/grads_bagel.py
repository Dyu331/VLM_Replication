"""Steering redesign, Phase 2 (PLAN_V4_STEER.md): gradients of the final lean for BAGEL-7B-MoT.

BAGEL answers in three steps: the image pass and the prompt pass fill its cache, then the first generation step
(its start token) gives the answer word, so that step is "the token before the answer" (as in extract_bagel.py and
steer_bagel.py). Here the image pass runs as usual (no gradient). The prompt pass and the first step run with
gradients (the prompt pass by hand, as in the inferencer's update_context_text, so its cache keeps the graph).
Gradients of the lean, LSE(left, Left) - LSE(right, Right), at every layer (0 = input embeddings, i = output of
decoder layer i) at three positions: the first step, the last mention of X, the last mention of Y.

Finite-difference check as in grads.py, pushing the first step.

    python repro/v4/grads_bagel.py --limit 8 --fd 4 --tag _smoke
    python repro/v4/grads_bagel.py
Writes $V4_RESULTS/bagel/steer2<tag>/<split>/{grads.npy, lean.npy} and fd_check.json.
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
from v4.prompts_v4 import TASK, WORDS, find_last
from v4.steer2_common import FD_B_MULT, FD_TARGETS, GRAD_POS, b_direction, fd_layers, fd_summary, headline
from data.data_utils import pil_img2rgb  # noqa: E402  (BAGEL repo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=str(BAGEL_PATH))
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fd", type=int, default=16)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    inf = load(args.path)
    model, lm, tok = inf.model, inf.model.language_model, inf.tokenizer
    model.requires_grad_(False)
    dev = torch.device("cuda:0")
    wid = torch.tensor([tok.encode(w)[0] for w in WORDS], device=dev)
    bos = inf.new_token_ids["bos_token_id"]
    layers = list(lm.model.layers)
    n_layers = len(layers)
    root = V4_RESULTS / "bagel" / f"steer2{args.tag}"
    st = {"grab": None, "outs": {}, "delta": {}}

    def hook(i):
        def fn(module, a, out):
            h = out[0] if isinstance(out, tuple) else out
            if st["grab"] is not None:
                h.retain_grad()
                st["outs"][(st["grab"], i)] = h
                return None
            d = st["delta"].get(i)
            if d is None:
                return None
            h = h + d.to(h.dtype)                       # the first step has one packed row
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return fn

    for i, layer in enumerate(layers, start=1):
        layer.register_forward_hook(hook(i))

    W = lm.lm_head.weight[wid].detach().float()

    def lean_of(h):
        """Lean from the final (normed) hidden state, scored in float32 for the four words."""
        z = h.float() @ W.T
        return torch.logsumexp(z[:2], 0) - torch.logsumexp(z[2:], 0)

    def grad_or_zero(t):
        return t.grad if t.grad is not None else torch.zeros_like(t)   # no gradient = it does not reach the answer

    def image_ctx(split, q):
        img = inf.vae_transform.resize_transform(pil_img2rgb(Image.open(V4_DATA / split / q["img_A"]).convert("RGB")))
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            return inf.update_context_image(img, inf.init_gen_context(), vae=False)

    def first_step(ctx, start_emb):
        cache, n, rope = ctx["past_key_values"], int(ctx["kv_lens"][0]), int(ctx["ropes"][0])
        out = lm.forward_inference(
            packed_query_sequence=start_emb,
            query_lens=torch.ones(1, dtype=torch.int, device=dev),
            packed_query_position_ids=torch.full((1,), rope, dtype=torch.long, device=dev),
            packed_query_indexes=torch.tensor([n], device=dev),
            past_key_values=cache,
            key_values_lens=torch.tensor([n], dtype=torch.int, device=dev),
            packed_key_value_indexes=torch.arange(n, device=dev),
            update_past_key_values=False, is_causal=True, mode="und")
        return lean_of(out.packed_query_sequence.reshape(-1, out.packed_query_sequence.shape[-1])[-1])

    def start_embedding():
        return lm.model.embed_tokens(torch.tensor([bos], device=dev))

    def grad_pass(split, q):
        ctx = image_ctx(split, q)
        text = TASK.format(B=q["B"], X=q["X"], Y=q["Y"])
        gi, kv_lens, ropes = model.prepare_prompts(curr_kvlens=ctx["kv_lens"], curr_rope=ctx["ropes"], prompts=[text],
                                                   tokenizer=tok, new_token_ids=inf.new_token_ids)
        gi = {k: v.to(dev) for k, v in gi.items()}
        ids = tok.encode(text)
        name_pos = [1 + find_last(ids, tok.encode(" " + q[r]), last=True) for r in "XY"]   # rows: [bos, prompt, eos]
        st["outs"] = {}
        with torch.enable_grad(), torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            emb_t = lm.model.embed_tokens(gi["packed_text_ids"]).detach().requires_grad_(True)
            st["grab"] = "text"
            lm.forward_inference(packed_query_sequence=emb_t, query_lens=gi["text_token_lens"],
                                 packed_query_position_ids=gi["packed_text_position_ids"],
                                 packed_query_indexes=gi["packed_text_indexes"], past_key_values=ctx["past_key_values"],
                                 packed_key_value_indexes=gi["packed_key_value_indexes"],
                                 key_values_lens=gi["key_values_lens"], update_past_key_values=True, is_causal=True,
                                 mode="und")
            assert int(gi["text_token_lens"][0]) == len(ids) + 2
            ctx["kv_lens"], ctx["ropes"] = kv_lens, ropes
            emb_s = start_embedding().detach().requires_grad_(True)
            st["grab"] = "step"
            lean = first_step(ctx, emb_s)
            st["grab"] = None
            lean.backward()
        o = st["outs"]
        flat = lambda t: grad_or_zero(t).reshape(-1, t.shape[-1])
        g_ans = [flat(emb_s)[0]] + [flat(o[("step", i)])[-1] for i in range(1, n_layers + 1)]
        g_txt = {p: [flat(emb_t)[p]] + [flat(o[("text", i)])[p] for i in range(1, n_layers + 1)] for p in name_pos}
        g = torch.stack([torch.stack(g_ans)] + [torch.stack(g_txt[p]) for p in name_pos])   # [3, L+1, H]
        st["outs"] = {}
        return g.float().cpu().numpy(), float(lean.detach())

    for split in args.splits:
        qs = [json.loads(l) for l in open(V4_DATA / split / f"questions_{split}.jsonl")][:args.limit]
        rows = [json.loads(l) for l in open(V4_RESULTS / "bagel" / split / "rows.jsonl")][:args.limit]
        assert all(q["qid"] == r["qid"] for q, r in zip(qs, rows))
        out_dir = root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        N, G, leans, t0 = len(qs), None, np.zeros(len(qs), np.float32), time.time()
        for i, q in enumerate(qs):
            g, leans[i] = grad_pass(split, q)
            if G is None:
                G = np.lib.format.open_memmap(out_dir / "grads.npy", "w+", np.float32, (len(GRAD_POS), g.shape[1], N, g.shape[2]))
            G[:, :, i] = g
            if i < 3:
                s = rows[i]["scores"]
                ref = np.logaddexp(s["left"], s["Left"]) - np.logaddexp(s["right"], s["Right"])
                print(f"  check q{i}: lean here {leans[i]:+.3f}, from the saved answer scores {ref:+.3f}", flush=True)
            if i % 100 == 0 or i == N - 1:
                print(f"bagel {split} {i + 1}/{N}  {time.time() - t0:.0f}s  |grad| at the answer step, layers 10/20/last-1: "
                      + " ".join(f"{np.linalg.norm(g[0, L]):.2e}" for L in (10, 20, n_layers - 1)), flush=True)
        G.flush()
        np.save(out_dir / "lean.npy", leans)

    if args.fd:
        qs = [json.loads(l) for l in open(V4_DATA / "test" / "questions_test.jsonl")][:args.fd]
        fl = fd_layers(n_layers, headline("bagel"))
        bdirs = {L: b_direction("bagel", L) for L in fl}
        records = []
        for i, q in enumerate(qs):
            g, _ = grad_pass("test", q)
            ctx = image_ctx("test", q)
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                ctx = inf.update_context_text(TASK.format(B=q["B"], X=q["X"], Y=q["Y"]), ctx)
                st["delta"] = {}
                lean0 = float(first_step(ctx, start_embedding()))
                for L in fl:
                    gl = g[0, L].astype(np.float64)
                    tests = [("own_grad", s * t, s * t / (gl @ gl) * gl) for t in FD_TARGETS for s in (1, -1)]
                    u, gap = bdirs[L]
                    sign = 1.0 if q["relB"] == "left" else -1.0
                    tests += [("B", k, sign * k * gap * u) for k in FD_B_MULT]
                    for kind, size, d in tests:
                        st["delta"] = {L: torch.tensor(d, dtype=torch.float32, device=dev)[None]}
                        act = float(first_step(ctx, start_embedding())) - lean0
                        st["delta"] = {}
                        records.append({"q": i, "layer": L, "kind": kind, "size": size, "pred": float(gl @ d), "actual": act})
            print(f"  fd {i + 1}/{len(qs)}", flush=True)
        root.mkdir(parents=True, exist_ok=True)
        fd_summary(records, root / "fd_check.json")


if __name__ == "__main__":
    main()
