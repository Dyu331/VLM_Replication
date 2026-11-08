"""Phase 3 for BAGEL-7B-MoT: the same outputs as extract.py, through the BAGEL repo's inferencer.

BAGEL fills its cache in two passes before answering: the image ([start, patches in raster order, end])
and then the prompt ([bos, prompt tokens, eos]). Hooks keep every position of every layer in both
passes, and the first generation step (its start token), which our first BAGEL probe also used as
"the token before the answer". Layer 0 = the input to the first decoder layer (text embeddings, or
the connector output + position embedding at the image patches); layers 1..28 = decoder outputs.

    python repro/v4/extract_bagel.py --limit 8 --tag _smoke
    python repro/v4/extract_bagel.py

Writes $V4_RESULTS/bagel<tag>/<split>/{states,text_only,vision}.npy, rows.jsonl and lens.npz
(no DeepStack for BAGEL).
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
from eval.prompts import parse
from eval.run_bagel import MAX_TOKENS, load  # also puts the BAGEL repo on sys.path
from v4.config_v4 import V4_DATA, V4_RESULTS
from v4.prompts_v4 import DIRECT, POSITIONS, ROW_KEYS, TASK, WORDS, cells, find_last, summarize
from data.data_utils import pil_img2rgb  # noqa: E402  (BAGEL repo)


class Recorder:
    """mode 'full': keep each module's whole output from its next call (a prefill pass);
    mode 'first': keep the last row of each module's first call (the first generation step)."""

    def __init__(self, lm):
        self.layers = list(lm.model.layers)
        self.mode, self.out, self.logits = None, None, None
        self.layers[0].register_forward_pre_hook(self._pre, with_kwargs=True)
        for i, layer in enumerate(self.layers, start=1):
            layer.register_forward_hook(self._post(i))
        lm.lm_head.register_forward_hook(self._head)

    def arm(self, mode):
        self.mode, self.out, self.logits = mode, [None] * (len(self.layers) + 1), None

    def _keep(self, i, h):
        if self.mode is None or self.out[i] is not None:
            return
        h = h.reshape(-1, h.shape[-1])
        self.out[i] = h.detach().clone() if self.mode == "full" else h[-1].detach().float().cpu()

    def _pre(self, module, args, kwargs):
        h = args[0] if args else kwargs.get("packed_query_sequence", kwargs.get("hidden_states"))
        self._keep(0, h)

    def _post(self, i):
        def fn(module, args, output):
            self._keep(i, output[0] if isinstance(output, tuple) else output)
        return fn

    def _head(self, module, args, output):
        if self.mode == "first" and self.logits is None:
            self.logits = output.reshape(-1, output.shape[-1])[-1].detach().float().cpu()

    def take(self):
        self.mode = None
        assert all(o is not None for o in self.out), "a hooked layer did not run"
        return self.out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=str(BAGEL_PATH))
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    inf = load(args.path)
    lm = inf.model.language_model
    rec = Recorder(lm)
    tok = inf.tokenizer
    word_ids = [tok.encode(w)[0] for w in WORDS]
    root = V4_RESULTS / ("bagel" + args.tag)
    root.mkdir(parents=True, exist_ok=True)
    np.savez(root / "lens.npz", words=np.array(WORDS), ids=np.array(word_ids),
             unembed=lm.lm_head.weight[word_ids].detach().float().cpu().numpy(),
             norm_weight=lm.model.norm.weight.detach().float().cpu().numpy())

    def answer(ctx):
        rec.arm("first")
        raw = inf.gen_text(ctx, do_sample=False, temperature=0.3, max_length=MAX_TOKENS)
        states = torch.stack(rec.take())                      # [L+1, H] at the start token
        lp = torch.log_softmax(rec.logits, -1)
        return states, raw, parse(raw), {w: float(lp[i]) for w, i in zip(WORDS, word_ids)}

    @torch.no_grad()
    def run(img, text, keep_prefill=False):
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            ctx = inf.init_gen_context()
            seg_img = seg_txt = None
            if img is not None:
                if keep_prefill:
                    rec.arm("full")
                ctx = inf.update_context_image(img, ctx, vae=False)
                if keep_prefill:
                    seg_img = rec.take()
            if keep_prefill:
                rec.arm("full")
            ctx = inf.update_context_text(text, ctx)
            if keep_prefill:
                seg_txt = rec.take()
            return answer(ctx) + (seg_img, seg_txt)

    for split in args.splits:
        qs = [json.loads(l) for l in open(V4_DATA / split / f"questions_{split}.jsonl")][:args.limit]
        out = root / split
        out.mkdir(parents=True, exist_ok=True)
        N, t0 = len(qs), time.time()
        S = T = V = None
        rows = []
        for i, q in enumerate(qs):
            d = V4_DATA / split
            imgA = inf.vae_transform.resize_transform(pil_img2rgb(Image.open(d / q["img_A"]).convert("RGB")))
            semA = np.asarray(Image.open(d / q["sem_A"]))
            text = TASK.format(B=q["B"], X=q["X"], Y=q["Y"])
            st, raw, ans, sc, seg_img, seg_txt = run(imgA, text, keep_prefill=True)
            L, H = st.shape
            if S is None:
                S = np.lib.format.open_memmap(out / "states.npy", "w+", np.float16, (len(POSITIONS), L, N, H))
                T = np.lib.format.open_memmap(out / "text_only.npy", "w+", np.float16, (L, N, H))
                V = np.lib.format.open_memmap(out / "vision.npy", "w+", np.float16, (4, N, H))
            # image pass: [start, patches (rows x cols, raster), end]
            vit = inf.vit_transform(imgA)
            rows_g, cols_g = vit.shape[1] // 14, vit.shape[2] // 14
            assert seg_img[0].shape[0] == rows_g * cols_g + 2, (seg_img[0].shape, rows_g, cols_g)
            obj_cells = {r: cells(semA, q["sid"][r], rows_g, cols_g) for r in "BXY"}
            img_stack = torch.stack([s[1:-1] for s in seg_img])          # [L, patches, H]
            # text pass: [bos, prompt tokens, eos]
            ids = tok.encode(text)
            name_pos = {r: 1 + find_last(ids, tok.encode(" " + q[r]), last=(r != "B")) for r in "BXY"}
            txt_stack = torch.stack(seg_txt)                              # [L, 1 + len(ids) + 1, H]
            assert txt_stack.shape[1] == len(ids) + 2
            vecs = [st.to(txt_stack.device)] + [txt_stack[:, name_pos[r]] for r in "BXY"]
            for r in "BXY":
                vecs.append(img_stack[:, torch.tensor(obj_cells[r], device=img_stack.device)].mean(1))
            S[:, :, i] = torch.stack([v.float() for v in vecs]).cpu().numpy().astype(np.float16)
            emb = img_stack[0]
            V[:, i] = torch.stack([emb.mean(0)] + [emb[torch.tensor(obj_cells[r], device=emb.device)].mean(0)
                                                   for r in "BXY"]).float().cpu().numpy().astype(np.float16)
            del seg_img, seg_txt, img_stack, txt_stack

            st_t, raw_t, ans_t, _, _, _ = run(None, text)
            T[:, i] = st_t.numpy().astype(np.float16)

            direct = DIRECT.format(X=q["X"], Y=q["Y"])
            _, raw_a, ans_a, sc_a, _, _ = run(imgA, direct)
            imgB = inf.vae_transform.resize_transform(pil_img2rgb(Image.open(d / q["img_B"]).convert("RGB")))
            _, raw_b, ans_b, sc_b, _, _ = run(imgB, direct)

            rows.append({k: q[k] for k in ROW_KEYS} | {
                "dist": q["dist"], "answer": ans, "raw": raw, "scores": sc,
                "text_only_answer": ans_t, "direct_A": ans_a, "direct_A_raw": raw_a, "direct_A_scores": sc_a,
                "direct_B": ans_b, "direct_B_raw": raw_b, "direct_B_scores": sc_b,
                "n_tokens": {r: int(len(obj_cells[r])) for r in "BXY"}, "grid": [rows_g, cols_g]})
            if i % 100 == 0 or i == N - 1:
                print(f"bagel {split} {i + 1}/{N}  {time.time() - t0:.0f}s  last: {ans} (A {q['relA']}, "
                      f"B {q['relB']})  direct {ans_a}/{ans_b}", flush=True)
        for arr in (S, T, V):
            arr.flush()
        with open(out / "rows.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        summarize(rows, f"bagel {split}")


if __name__ == "__main__":
    main()
