"""Qwen3-VL-8B-Instruct answers for settings 1, identity, 2, 4 (report Section 2).

For each (set, setting) writes $PP_RESULTS/qwen3vl/<set>/<setting>.jsonl: one row per clean
example with the raw answer, parsed answer and truth fields.

    python repro/eval/run_qwen.py --sets test
    python repro/eval/run_qwen.py --model /path/to/local/Qwen3-VL-8B-Instruct
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import PP_DATA, PP_RESULTS
from eval.prompts import SETTINGS, build, parse

MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def load_clean(set_name):
    path = PP_DATA / set_name / "examples_labeled.jsonl"
    return [r for r in map(json.loads, open(path)) if r["clean"]]


@torch.no_grad()
def answer(model, processor, image_path, text):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": Image.open(image_path).convert("RGB")},
        {"type": "text", "text": text}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt").to(model.device)
    gen = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    return processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL, help="HF id or local path to the weights")
    ap.add_argument("--sets", nargs="+", default=["test"])
    ap.add_argument("--settings", nargs="+", default=SETTINGS)
    ap.add_argument("--limit", type=int, default=None, help="first N examples only (smoke test)")
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()

    for set_name in args.sets:
        recs = load_clean(set_name)[:args.limit]
        out_dir = PP_RESULTS / "qwen3vl" / set_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for setting in args.settings:
            rows = []
            for rec in recs:
                view, text, truth_view = build(rec, setting)
                img = PP_DATA / set_name / rec["scene"] / f"{rec['id']}_{view}_rgb.png"
                raw = answer(model, processor, img, text)
                rows.append({"id": rec["id"], "scene": rec["scene"], "setting": setting,
                             "raw": raw, "answer": parse(raw), "truth": rec[f"rel{truth_view}"],
                             "relA": rec["relA"], "relB": rec["relB"], "prompt": text})
            with open(out_dir / f"{setting}.jsonl", "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)
            acc = np.mean([r["answer"] == r["truth"] for r in rows])
            print(f"{set_name} {setting}: {len(rows)} prompts, accuracy {acc:.1%}", flush=True)


if __name__ == "__main__":
    main()
