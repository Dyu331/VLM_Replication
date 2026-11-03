"""Hidden states for the left/right probe (report Section 3), for BAGEL-7B-MoT.

The report reads BAGEL "at the start token of its first generation step". BAGEL's inferencer
first fills its cache with the image and the prompt, then generate_text() feeds the start token
and decodes. We hook the token embedding and every decoder layer and keep only their outputs on
that first decoding step (index 0 = embedding, 1..28 = layer outputs, before the final norm).
The answer comes from the same call, so states and labels match.

Same prompts, sets and output format as extract_vlm.py:
  $PP_RESULTS/probe/bagel/<set>/<setting>.{npy,jsonl}

    python repro/probing/extract_bagel.py --limit 3
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import BAGEL_PATH, PP_DATA, PP_RESULTS
from eval.loop import load_clean
from eval.prompts import build, image_name, parse
from eval.run_bagel import MAX_TOKENS, load  # also puts the BAGEL repo on sys.path
from data.data_utils import pil_img2rgb  # noqa: E402  (BAGEL repo)

PLAN = {"train": ["s4", "s2", "s1_identity"], "test": ["s1", "s1_identity", "s2", "s4"]}


class FirstStepRecorder:
    """Keeps each hooked module's output from the first call made while armed."""

    def __init__(self, lm):
        self.modules = [lm.model.embed_tokens] + list(lm.model.layers)
        self.armed, self.out = False, [None] * len(self.modules)
        for i, mod in enumerate(self.modules):
            mod.register_forward_hook(self._hook(i))

    def _hook(self, i):
        def fn(module, inputs, output):
            if self.armed and self.out[i] is None:
                h = output[0] if isinstance(output, tuple) else output
                self.out[i] = h.reshape(-1, h.shape[-1])[-1].float().cpu().numpy()
        return fn

    def arm(self):
        self.armed, self.out = True, [None] * len(self.modules)

    def collect(self):
        self.armed = False
        assert all(o is not None for o in self.out), "a hooked module did not run on the first step"
        return np.stack(self.out).astype(np.float16)


@torch.no_grad()
def states_and_answer(inf, rec, image_path, text):
    # same steps as InterleaveInferencer.interleave_inference(understanding_output=True)
    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        ctx = inf.init_gen_context()
        img = inf.vae_transform.resize_transform(pil_img2rgb(Image.open(image_path).convert("RGB")))
        ctx = inf.update_context_image(img, ctx, vae=False)
        ctx = inf.update_context_text(text, ctx)
        rec.arm()
        raw = inf.gen_text(ctx, do_sample=False, temperature=0.3, max_length=MAX_TOKENS)
    return rec.collect(), raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=str(BAGEL_PATH))
    ap.add_argument("--sets", nargs="+", default=["train", "test"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    inf = load(args.path)
    rec = FirstStepRecorder(inf.model.language_model)
    for set_name in args.sets:
        recs = load_clean(set_name)[:args.limit]
        out_dir = PP_RESULTS / "probe" / "bagel" / set_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for setting in PLAN[set_name]:
            rows, states = [], []
            for r in recs:
                view, text, truth_view = build(r, setting)
                img = PP_DATA / set_name / r["scene"] / image_name(r, setting, view)
                hidden, raw = states_and_answer(inf, rec, img, text)
                states.append(hidden)
                rows.append({"id": r["id"], "scene": r["scene"], "setting": setting, "raw": raw,
                             "answer": parse(raw), "truth": r[f"rel{truth_view}"],
                             "relA": r["relA"], "relB": r["relB"]})
            np.save(out_dir / f"{setting}.npy", np.stack(states))
            with open(out_dir / f"{setting}.jsonl", "w") as f:
                f.writelines(json.dumps(x) + "\n" for x in rows)
            acc = sum(x["answer"] == x["truth"] for x in rows) / len(rows)
            print(f"bagel {set_name} {setting}: {len(rows)} prompts, states {np.stack(states).shape}, "
                  f"accuracy {acc:.1%}, invalid {sum(x['answer'] == 'invalid' for x in rows)}", flush=True)


if __name__ == "__main__":
    main()
