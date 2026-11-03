"""VLM answers for settings 1, identity, 2, 4 (report Section 2) and their marked twins, for any
model in MODELS. Marked settings need repro/render/make_marked.py to have run first.

For each (set, setting) writes $PP_RESULTS/<model>/<set>/<setting>.jsonl: one row per clean
example with the raw answer, parsed answer and truth fields.

    python repro/eval/run_vlm.py --model qwen3vl
    python repro/eval/run_vlm.py --model molmo2 --limit 5

BAGEL is not a transformers-v5 model; it has its own script (run_bagel.py).
"""
import argparse
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import GEMMA_PATH, MOLMO_PATH, QWEN_PATH
from eval.loop import run_settings
from eval.prompts import SETTINGS

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
    - it asks for the "default" RoPE init, which v5 no longer registers;
    - it registers its classes with the Auto factories at import, but newer v5 ships a native
      Molmo2 under the same config name, so registration must be allowed to overwrite
      (registered classes take priority, so the checkpoint's own code is what runs).
    """
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    from transformers.models.auto.auto_factory import _LazyAutoMapping
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

    original_register = _LazyAutoMapping.register

    def overwriting_register(self, key, value, exist_ok=False):
        return original_register(self, key, value, exist_ok=True)

    previous_rope = ROPE_INIT_FUNCTIONS.get("default")
    ProcessorMixin.__init__ = compatible_init
    ROPE_INIT_FUNCTIONS["default"] = default_rope
    _LazyAutoMapping.register = overwriting_register
    try:
        yield
    finally:
        ProcessorMixin.__init__ = original_init
        _LazyAutoMapping.register = original_register
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
                path, trust_remote_code=True, dtype=torch.bfloat16)
        model.to("cuda")
        print(f"molmo2 loaded as {type(model).__module__}.{type(model).__name__}", flush=True)
    else:
        processor = AutoProcessor.from_pretrained(path)
        model = AutoModelForImageTextToText.from_pretrained(
            path, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    return model.eval(), processor


# ---------- inference ----------

@torch.no_grad()
def answer(model, processor, image_path, text, first_token_only=False):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": Image.open(image_path).convert("RGB")},
        {"type": "text", "text": text}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt").to(model.device)
    if first_token_only:
        # Greedy decoding's first token from one forward pass. Used for Molmo2, whose remote
        # prepare_inputs_for_generation needs the cache_position that v5's generate() no longer
        # passes. "left"/"right"/"Left"/"Right" are single tokens, so the parsed answer is the same.
        token = model(**inputs).logits[0, -1].argmax()
        return processor.tokenizer.decode([int(token)], skip_special_tokens=True)
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
    first_token_only = MODELS[args.model][1] == "molmo2"
    run_settings(args.model, lambda img, text: answer(model, processor, img, text, first_token_only),
                 args.sets, args.settings, args.limit)


if __name__ == "__main__":
    main()
