#!/usr/bin/env bash
# Creates the BAGEL env on Turbo and clones the BAGEL repo (its code is needed to load the model).
# BAGEL pins transformers 4.49 / torch 2.5.1, which clash with the v5 `vlm` env, hence a separate env.
# Run once, from a login node: `module load python3.11-anaconda; bash setup/create_env_bagel.sh`
set -euo pipefail
source "$(dirname "$0")/paths.sh"

ENV=$CONDA_ENVS/bagel
CODE=$TURBO/code/Bagel
[ -d "$CODE" ] || git clone https://github.com/ByteDance-Seed/Bagel.git "$CODE"
git -C "$CODE" log -1 --format="BAGEL repo at %h (%cd)"

conda create -y -p "$ENV" python=3.10
"$ENV/bin/pip" install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
# the repo's requirements minus torch (installed above) and training/demo-only extras
grep -viE '^(torch|torchvision|#|wandb|gradio|bitsandbytes|xlsxwriter|triton-windows)' "$CODE/requirements.txt" \
  > "$TURBO/code/bagel_requirements.txt"
"$ENV/bin/pip" install -r "$TURBO/code/bagel_requirements.txt"
# prebuilt flash-attn for torch 2.5 / CUDA 12 / python 3.10 (building from source is slow)
"$ENV/bin/pip" install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

"$ENV/bin/python" -c "import torch, transformers, flash_attn; print('torch', torch.__version__, 'transformers', transformers.__version__, 'flash_attn', flash_attn.__version__)"
