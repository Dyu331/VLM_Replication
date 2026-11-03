#!/usr/bin/env bash
# Creates the model env (Qwen3-VL behaviour + hidden states, probe analysis) on Turbo.
# Run once, from a login node: `module load python3.11-anaconda; bash setup/create_env_vlm.sh`
set -euo pipefail
source "$(dirname "$0")/paths.sh"

ENV=$CONDA_ENVS/vlm
conda create -y -p "$ENV" python=3.11
"$ENV/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# transformers v5: Qwen3-VL and Gemma 4 natively; Molmo2 via patches in run_vlm.py (its code needs einops)
"$ENV/bin/pip" install "transformers>=5" accelerate einops pillow numpy pandas scikit-learn tqdm

"$ENV/bin/python" -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"
