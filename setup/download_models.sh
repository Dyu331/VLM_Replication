#!/usr/bin/env bash
# NOT NEEDED: both models already exist in /nfs/turbo/coe-chaijy-unreplicated/pre-trained-weights/
# (see QWEN_PATH / BAGEL_PATH in repro/common/config.py). Kept only as a fallback.
# Downloads model weights into HF_HOME on Turbo (~17 GB Qwen3-VL-8B + ~30 GB BAGEL-7B-MoT).
# Needs huggingface_hub: `pip install -U huggingface_hub` in any env.
set -euo pipefail
source "$(dirname "$0")/paths.sh"

huggingface-cli download Qwen/Qwen3-VL-8B-Instruct
huggingface-cli download ByteDance-Seed/BAGEL-7B-MoT --local-dir "$HF_HOME/BAGEL-7B-MoT"
