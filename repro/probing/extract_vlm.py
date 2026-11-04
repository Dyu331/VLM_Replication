"""Hidden states for the left/right probe (report Section 3), for transformers models (Qwen3-VL, Gemma 4).

For every prompt: one forward pass keeps the last prompt token's hidden state at every layer
(index 0 = input embeddings), then greedy decoding gives the model's own answer, which is the
reader's training label. Prompts are the eval's own (eval/prompts.py), on clean examples only:
  train rooms: s4 (direct, A's photo), s2 (direct, B's photo), s1_identity
  test rooms:  s1, s1_identity, s2, s4

Writes $PP_RESULTS/probe/<model>/<set>/<setting>.npy (float16 [N, layers+1, hidden]) and a
.jsonl with id, scene, answer and truths in the same row order.

    python repro/probing/extract_vlm.py --model qwen3vl
    python repro/probing/extract_vlm.py --model gemma4 --limit 3
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import PP_DATA, PP_RESULTS
from eval.loop import load_clean
from eval.prompts import build, image_name, parse
from eval.run_vlm import MODELS, load

PLAN = {"train": ["s4", "s2", "s1_identity"], "test": ["s1", "s1_identity", "s2", "s4"]}
# --marked: the same readers on outlined photos (X orange, Y blue), trained and tested on them
PLAN_MARKED = {"train": ["s4_marked", "s2_marked", "s1_identity_marked"], "test": ["s1_marked", "s1_identity_marked"]}


@torch.no_grad()
def states_and_answer(model, processor, image_path, text, max_new_tokens):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": Image.open(image_path).convert("RGB")},
        {"type": "text", "text": text}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt").to(model.device)
    out = model(**inputs, output_hidden_states=True)
    hidden = torch.stack([h[0, -1] for h in out.hidden_states]).float().cpu().numpy().astype(np.float16)
    gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    raw = processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return hidden, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["qwen3vl", "gemma4"])
    ap.add_argument("--sets", nargs="+", default=["train", "test"])
    ap.add_argument("--marked", action="store_true", help="outlined photos (PLAN_MARKED)")
    ap.add_argument("--limit", type=int, default=None, help="first N examples per set (smoke test)")
    ap.add_argument("--max-new-tokens", type=int, default=5)
    args = ap.parse_args()

    model, processor = load(args.model, str(MODELS[args.model][0]))
    for set_name in args.sets:
        recs = load_clean(set_name)[:args.limit]
        out_dir = PP_RESULTS / "probe" / args.model / set_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for setting in (PLAN_MARKED if args.marked else PLAN)[set_name]:
            rows, states = [], []
            for rec in recs:
                view, text, truth_view = build(rec, setting)
                img = PP_DATA / set_name / rec["scene"] / image_name(rec, setting, view)
                hidden, raw = states_and_answer(model, processor, img, text, args.max_new_tokens)
                states.append(hidden)
                rows.append({"id": rec["id"], "scene": rec["scene"], "setting": setting, "raw": raw,
                             "answer": parse(raw), "truth": rec[f"rel{truth_view}"],
                             "relA": rec["relA"], "relB": rec["relB"]})
            np.save(out_dir / f"{setting}.npy", np.stack(states))
            with open(out_dir / f"{setting}.jsonl", "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)
            acc = sum(r["answer"] == r["truth"] for r in rows) / len(rows)
            print(f"{args.model} {set_name} {setting}: {len(rows)} prompts, states {np.stack(states).shape}, "
                  f"accuracy {acc:.1%}, invalid {sum(r['answer'] == 'invalid' for r in rows)}", flush=True)


if __name__ == "__main__":
    main()
