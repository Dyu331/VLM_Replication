"""Phase 3 (PLAN_V4.md): answers and hidden states for the v4 questions, Qwen3-VL or Gemma 4.

Per question (train and test splits):
  task prompt on A's photo  -> one forward pass: the answer (greedy first token, generate() as a
                               fallback), the left/right scores, and every layer's hidden state at 7
                               positions [theirs: positions; pooling ours]:
                                 0 token before the answer        1-3 last token of B's / X's / Y's name
                                 4-6 mean over the image tokens covering B's / X's / Y's mask
  task prompt, no picture   -> every layer at the token before the answer (text-only control) [theirs]
  direct question on A's photo and on the view from B's spot -> answers [theirs]
Vision features (the language model's input at the image positions, i.e. layer 0 there):
  mean over the whole image [theirs] and per object B, X, Y [violet]; for Qwen3-VL also the three
  DeepStack feature maps, pooled the same way [violet].

    python repro/v4/extract.py --model qwen3vl --limit 8      # smoke test
    python repro/v4/extract.py --model qwen3vl

Writes $V4_RESULTS/<model>/<split>/:
  states.npy     float16 [7, layers+1, N, hidden]   (position, layer, question, unit)
  text_only.npy  float16 [layers+1, N, hidden]
  vision.npy     float16 [4, N, hidden]             (whole image, B, X, Y)
  deepstack.npy  float16 [levels, 4, N, hidden]     (Qwen3-VL only)
  rows.jsonl     one row per question: facts, answers, scores, token counts
and $V4_RESULTS/<model>/lens.npz (final-norm weight, vocabulary rows for left/Left/right/Right).
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
from eval.prompts import parse
from eval.run_vlm import MODELS, load
from v4.config_v4 import V4_DATA, V4_RESULTS

TASK = ("This image shows what you currently see. Imagine you are standing where the {B} is, looking toward "
        "the {X} and the {Y}. From there, is the {X} to the left or to the right of the {Y}? "
        "Answer with exactly one word: left or right.")
DIRECT = "In this image, is the {X} to the left or to the right of the {Y}? Answer with exactly one word: left or right."
POSITIONS = ["answer", "name_B", "name_X", "name_Y", "patch_B", "patch_X", "patch_Y"]
WORDS = ["left", "Left", "right", "Right"]
MIN_CELL = 0.2  # an image token belongs to an object when its mask covers >= 20% of the token's cell [ours]


class Runner:
    def __init__(self, name):
        self.name = name
        self.model, self.processor = load(name, str(MODELS[name][0]))
        self.tok = self.processor.tokenizer
        self.word_ids = [self.tok.encode(w, add_special_tokens=False)[0] for w in WORDS]
        self.vision_out = None
        if name == "qwen3vl":
            self.model.model.visual.register_forward_hook(self._grab_vision)
            self.image_token_id = self.model.config.image_token_id
        else:
            self.image_token_id = getattr(self.model.config, "image_token_id", None) or \
                getattr(self.model.config, "image_token_index")

    def _grab_vision(self, module, args, out):
        self.vision_out = out

    def inputs(self, text, image=None):
        content = ([{"type": "image", "image": image}] if image is not None else []) + [{"type": "text", "text": text}]
        return self.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
                                                  add_generation_prompt=True, return_dict=True,
                                                  return_tensors="pt").to(self.model.device)

    @torch.no_grad()
    def forward(self, inp, hidden=True):
        self.vision_out = None
        out = self.model(**inp, output_hidden_states=hidden)
        logits = out.logits[0, -1].float()
        lp = torch.log_softmax(logits, -1)
        tok = int(logits.argmax())
        raw = self.tok.decode([tok], skip_special_tokens=True)
        ans = parse(raw)
        if ans == "invalid":  # e.g. a leading space or quote token: decode a few tokens
            gen = self.model.generate(**inp, max_new_tokens=5, do_sample=False)
            raw = self.tok.decode(gen[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
            ans = parse(raw)
        scores = {w: float(lp[i]) for w, i in zip(WORDS, self.word_ids)}
        return out, raw, ans, scores

    def image_grid(self, inp):
        """(rows, cols, sequence positions of the image tokens in raster order)."""
        ids = inp["input_ids"][0]
        pos = torch.nonzero(ids == self.image_token_id).flatten().tolist()
        if self.name == "qwen3vl":
            _, h, w = inp["image_grid_thw"][0].tolist()
            merge = self.model.model.visual.spatial_merge_size
            rows, cols = h // merge, w // merge
        else:  # square grid of soft tokens
            rows = cols = int(round(len(pos) ** 0.5))
        assert rows * cols == len(pos), (rows, cols, len(pos))
        return rows, cols, pos

    def name_pos(self, inp, name, last=True):
        """Position of the last token of an occurrence of ' <name>' in the prompt."""
        ids = inp["input_ids"][0].tolist()
        sub = self.tok.encode(" " + name, add_special_tokens=False)
        hits = [i for i in range(len(ids) - len(sub) + 1) if ids[i:i + len(sub)] == sub]
        assert hits, name
        return (hits[-1] if last else hits[0]) + len(sub) - 1


def cells(sem, sid, rows, cols):
    """Indices (raster) of grid cells whose area is >= MIN_CELL covered by the object's mask."""
    m = Image.fromarray((sem == sid).astype(np.float32), mode="F")
    frac = np.asarray(m.resize((cols, rows), Image.BOX))  # mean mask coverage per cell
    idx = np.flatnonzero(frac.ravel() >= MIN_CELL)
    return idx if len(idx) else np.array([int(frac.argmax())])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["qwen3vl", "gemma4"])
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="", help="output folder suffix, e.g. _smoke")
    args = ap.parse_args()

    R = Runner(args.model)
    root = V4_RESULTS / (args.model + args.tag)
    root.mkdir(parents=True, exist_ok=True)
    # logit-lens ingredients [violet]
    lm = R.model.get_output_embeddings().weight
    norm = R.model.model.language_model.norm.weight if hasattr(R.model.model, "language_model") else None
    np.savez(root / "lens.npz", words=np.array(WORDS), ids=np.array(R.word_ids),
             unembed=lm[R.word_ids].detach().float().cpu().numpy(),
             norm_weight=(norm.detach().float().cpu().numpy() if norm is not None else np.zeros(0)))

    for split in args.splits:
        qs = [json.loads(l) for l in open(V4_DATA / split / f"questions_{split}.jsonl")][:args.limit]
        out = root / split
        out.mkdir(parents=True, exist_ok=True)
        N, t0 = len(qs), time.time()
        S = T = V = D = None
        rows = []
        for i, q in enumerate(qs):
            d = V4_DATA / split
            imgA = Image.open(d / q["img_A"]).convert("RGB")
            semA = np.asarray(Image.open(d / q["sem_A"]))
            text = TASK.format(B=q["B"], X=q["X"], Y=q["Y"])

            inp = R.inputs(text, imgA)
            o, raw, ans, sc = R.forward(inp)
            hs = o.hidden_states                      # tuple of layers+1 tensors [1, seq, hidden]
            L, H = len(hs), hs[0].shape[-1]
            if S is None:
                S = np.lib.format.open_memmap(out / "states.npy", "w+", np.float16, (len(POSITIONS), L, N, H))
                T = np.lib.format.open_memmap(out / "text_only.npy", "w+", np.float16, (L, N, H))
                V = np.lib.format.open_memmap(out / "vision.npy", "w+", np.float16, (4, N, H))
            rows_g, cols_g, img_pos = R.image_grid(inp)
            img_pos = torch.tensor(img_pos, device=hs[0].device)
            obj_cells = {r: cells(semA, q["sid"][r], rows_g, cols_g) for r in "BXY"}
            stack = torch.stack([h[0] for h in hs])  # [L, seq, H]
            vecs = [stack[:, -1]] + [stack[:, R.name_pos(inp, q[r], last=(r != "B"))] for r in "BXY"]
            for r in "BXY":
                vecs.append(stack[:, img_pos[torch.tensor(obj_cells[r], device=img_pos.device)]].mean(1))
            S[:, :, i] = torch.stack(vecs).float().cpu().numpy().astype(np.float16)
            emb = stack[0, img_pos]                     # language-model input at the image tokens
            V[:, i] = torch.stack([emb.mean(0)] + [emb[torch.tensor(obj_cells[r], device=emb.device)].mean(0)
                                                   for r in "BXY"]).float().cpu().numpy().astype(np.float16)
            if R.vision_out is not None and getattr(R.vision_out, "deepstack_features", None):
                ds = R.vision_out.deepstack_features
                if D is None:
                    D = np.lib.format.open_memmap(out / "deepstack.npy", "w+", np.float16, (len(ds), 4, N, H))
                for k, f in enumerate(ds):
                    if isinstance(f, (tuple, list)):  # get_image_features splits it per image (one here)
                        f = torch.cat(list(f), 0)
                    f = f.to(emb.device)
                    assert f.shape[0] == len(img_pos), (f.shape, len(img_pos))
                    D[k, :, i] = torch.stack([f.mean(0)] + [f[torch.tensor(obj_cells[r], device=f.device)].mean(0)
                                                            for r in "BXY"]).float().cpu().numpy().astype(np.float16)
            del o, hs, stack

            o_t, raw_t, ans_t, _ = R.forward(R.inputs(text))
            T[:, i] = torch.stack([h[0, -1] for h in o_t.hidden_states]).float().cpu().numpy().astype(np.float16)
            del o_t

            direct = DIRECT.format(X=q["X"], Y=q["Y"])
            _, raw_a, ans_a, sc_a = R.forward(R.inputs(direct, imgA), hidden=False)
            imgB = Image.open(d / q["img_B"]).convert("RGB")
            _, raw_b, ans_b, sc_b = R.forward(R.inputs(direct, imgB), hidden=False)

            rows.append({k: q[k] for k in ("qid", "photo", "room", "split", "B", "X", "Y", "relA", "relB", "flip",
                                           "depth_rank", "image_side", "angle_A", "angle_B")} | {
                "dist": q["dist"], "answer": ans, "raw": raw, "scores": sc,
                "text_only_answer": ans_t, "direct_A": ans_a, "direct_A_raw": raw_a, "direct_A_scores": sc_a,
                "direct_B": ans_b, "direct_B_raw": raw_b, "direct_B_scores": sc_b,
                "n_tokens": {r: int(len(obj_cells[r])) for r in "BXY"}, "grid": [rows_g, cols_g]})
            if i % 100 == 0 or i == N - 1:
                print(f"{args.model} {split} {i + 1}/{N}  {time.time() - t0:.0f}s  last: {ans} (A {q['relA']}, "
                      f"B {q['relB']})  direct {ans_a}/{ans_b}", flush=True)
        for arr in (S, T, V, D):
            if arr is not None:
                arr.flush()
        with open(out / "rows.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        summarize(rows, f"{args.model} {split}")


def summarize(rows, label):
    """The PhD's behaviour table: gives B's / A's answer, says left, direct-A, direct-B, flip split."""
    n = len(rows)
    m = lambda f: sum(f(r) for r in rows) / max(n, 1)
    flip = [r for r in rows if r["flip"]]
    same = [r for r in rows if not r["flip"]]
    acc = lambda rs: sum(r["answer"] == r["relB"] for r in rs) / max(len(rs), 1)
    print(f"\n{label}: {n} questions | gives B's answer {m(lambda r: r['answer'] == r['relB']):.0%} | "
          f"gives A's answer {m(lambda r: r['answer'] == r['relA']):.0%} | says left {m(lambda r: r['answer'] == 'left'):.0%} | "
          f"invalid {m(lambda r: r['answer'] == 'invalid'):.0%} | direct-A {m(lambda r: r['direct_A'] == r['relA']):.0%} | "
          f"direct-B {m(lambda r: r['direct_B'] == r['relB']):.0%} | flip questions {acc(flip):.0%} | "
          f"no-flip questions {acc(same):.0%}", flush=True)
    for key, vals in (("depth_rank", ("nearest", "middle", "farthest")), ("image_side", ("left", "right"))):
        parts = []
        for v in vals:
            rs = [r for r in rows if r[key] == v]
            if rs:
                fr = sum(r["answer"] != r["relA"] for r in rs) / len(rs)
                parts.append(f"{v}: B's answer {acc(rs):.0%}, answers differently from A {fr:.0%} (n={len(rs)})")
        print(f"   by {key}: " + "; ".join(parts), flush=True)


if __name__ == "__main__":
    main()
