"""VLM answers for settings 1, identity, 2, 4 (report Section 2) and their marked twins, for any
model in MODELS. Marked settings need repro/render/make_marked.py to have run first.

For each (set, setting) writes $PP_RESULTS/<model>/<set>/<setting>.jsonl: one row per clean
example with the raw answer, parsed answer and truth fields.

    python repro/eval/run_vlm.py --model qwen3vl
    python repro/eval/run_vlm.py --model molmo2 --limit 5

BAGEL is not a transformers model; it has its own script.
"""
import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import GEMMA_PATH, MOLMO_PATH, PP_DATA, PP_RESULTS, QWEN_PATH
from eval.prompts import SETTINGS, build, image_name, parse

# name -> (weights, loader). Qwen3-VL is the report's model; Gemma 4 and Molmo2 are extensions.
MODELS = {
    "qwen3vl": (QWEN_PATH, "standard"),
    "gemma4": (GEMMA_PATH, "standard"),
    "molmo2": (MOLMO_PATH, "molmo2"),
}


# ---------- Molmo2 on transformers v5 (patches from VLM_Spatial_Eval/attention/molmo2.py) ----------

@contextmanager
def molmo2_compat():
    """Molmo2's remote code predates transformers v5. While loading:
    - its processor passes optional attributes that ProcessorMixin.__init__ now rejects;
    - it asks for the "default" RoPE init, which v5 no longer registers.
    """
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    from transformers.processing_utils import ProcessorMixin

    original_init = ProcessorMixin.__init__

    def compatible_init(self, *args, **kwargs):
        saved = {k: kwargs.pop(k) for k in getattr(self, "optional_attributes", [])
                 if k != "chat_template" and k in kwargs}
        original_init(self, *args, **kwargs)
        for k, v in saved.items():
            setattr(self, k, v)

    def default_rope(config, device=None, seq_len=None, **_):
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * (getattr(config, "partial_rotary_factor", None) or 1.0))
        inv_freq = 1.0 / (float(config.rope_theta) ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
        return inv_freq.to(torch.float32).to(device), 1.0

    previous_rope = ROPE_INIT_FUNCTIONS.get("default")
    ProcessorMixin.__init__ = compatible_init
    ROPE_INIT_FUNCTIONS["default"] = default_rope
    try:
        yield
    finally:
        ProcessorMixin.__init__ = original_init
        if previous_rope is None:
            ROPE_INIT_FUNCTIONS.pop("default", None)
        else:
            ROPE_INIT_FUNCTIONS["default"] = previous_rope


def load(name, path):
    loader = MODELS[name][1]
    if loader == "molmo2":
        with molmo2_compat():
            processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
            model = AutoModelForImageTextToText.from_pretrained(
                path, trust_remote_code=True, torch_dtype=torch.bfloat16)
        model.to("cuda")
    else:
        processor = AutoProcessor.from_pretrained(path)
        model = AutoModelForImageTextToText.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    return model.eval(), processor


# ---------- inference ----------

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
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--path", default=None, help="override the weights folder")
    ap.add_argument("--sets", nargs="+", default=["test"])
    ap.add_argument("--settings", nargs="+", default=SETTINGS)
    ap.add_argument("--limit", type=int, default=None, help="first N examples only (smoke test)")
    args = ap.parse_args()

    model, processor = load(args.model, args.path or str(MODELS[args.model][0]))
    for set_name in args.sets:
        recs = load_clean(set_name)[:args.limit]
        out_dir = PP_RESULTS / args.model / set_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for setting in args.settings:
            rows = []
            for rec in recs:
                view, text, truth_view = build(rec, setting)
                img = PP_DATA / set_name / rec["scene"] / image_name(rec, setting, view)
                raw = answer(model, processor, img, text)
                rows.append({"id": rec["id"], "scene": rec["scene"], "setting": setting,
                             "raw": raw, "answer": parse(raw), "truth": rec[f"rel{truth_view}"],
                             "relA": rec["relA"], "relB": rec["relB"], "prompt": text})
            with open(out_dir / f"{setting}.jsonl", "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)
            acc = np.mean([r["answer"] == r["truth"] for r in rows])
            bad = sum(r["answer"] == "invalid" for r in rows)
            print(f"{args.model} {set_name} {setting}: {len(rows)} prompts, accuracy {acc:.1%}, "
                  f"invalid {bad}  e.g. {rows[0]['raw']!r}", flush=True)


if __name__ == "__main__":
    main()
