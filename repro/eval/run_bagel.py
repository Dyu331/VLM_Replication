"""BAGEL-7B-MoT answers for all settings (report Section 2), via the BAGEL repo's inferencer.

Loading follows the repo's inference.ipynb; answers use its understanding mode
(image then text, no thinking, greedy). Writes $PP_RESULTS/bagel/<set>/<setting>.jsonl.

    python repro/eval/run_bagel.py --limit 5
"""
import argparse
import os
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import BAGEL_CODE, BAGEL_PATH
from eval.loop import run_settings
from eval.prompts import SETTINGS

# after our own packages: the BAGEL repo also has an `eval` folder
sys.path.append(str(BAGEL_CODE))
from accelerate import init_empty_weights, load_checkpoint_and_dispatch  # noqa: E402
from data.data_utils import add_special_tokens  # noqa: E402
from data.transforms import ImageTransform  # noqa: E402
from inferencer import InterleaveInferencer  # noqa: E402
from modeling.autoencoder import load_ae  # noqa: E402
from modeling.bagel import (  # noqa: E402
    Bagel, BagelConfig, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel)
from modeling.qwen2 import Qwen2Tokenizer  # noqa: E402

MAX_TOKENS = 20  # one-word answer; BAGEL counts its start tokens too


def load(model_path):
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1
    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))
    config = BagelConfig(
        visual_gen=True, visual_und=True, llm_config=llm_config, vit_config=vit_config,
        vae_config=vae_config, vit_max_num_patch_per_side=70, connector_act="gelu_pytorch_tanh",
        latent_patch_size=2, max_latent_size=64)
    with init_empty_weights():
        model = Bagel(Qwen2ForCausalLM(llm_config), SiglipVisionModel(vit_config), config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    # Everything on one GPU: ~29 GB in bf16 fits an A40. (The notebook's infer_auto_device_map
    # sizes the empty model in fp32, ~56 GB, and spills layers to disk, which crashed here.)
    model = load_checkpoint_and_dispatch(
        model, checkpoint=os.path.join(model_path, "ema.safetensors"), device_map={"": "cuda:0"},
        dtype=torch.bfloat16).eval()
    return InterleaveInferencer(
        model=model, vae_model=vae_model, tokenizer=tokenizer,
        vae_transform=ImageTransform(1024, 512, 16), vit_transform=ImageTransform(980, 224, 14),
        new_token_ids=new_token_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=str(BAGEL_PATH), help="BAGEL weights folder")
    ap.add_argument("--sets", nargs="+", default=["test"])
    ap.add_argument("--settings", nargs="+", default=SETTINGS)
    ap.add_argument("--limit", type=int, default=None, help="first N examples only (smoke test)")
    args = ap.parse_args()

    inferencer = load(args.path)

    def answer(image_path, text):
        out = inferencer(image=Image.open(image_path).convert("RGB"), text=text,
                         understanding_output=True, max_think_token_n=MAX_TOKENS, do_sample=False)
        return out["text"]

    run_settings("bagel", answer, args.sets, args.settings, args.limit)


if __name__ == "__main__":
    main()
