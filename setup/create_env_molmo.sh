#!/usr/bin/env bash
# Separate env for Molmo2: its remote code targets transformers 4.x (4.57.1, as in the
# MolmoPoint setup of VLM_Spatial_Eval); under v5 it breaks in several places.
# Run once, from a login node: `module load python3.11-anaconda; bash setup/create_env_molmo.sh`
set -euo pipefail
source "$(dirname "$0")/paths.sh"

ENV=$CONDA_ENVS/molmo
conda create -y -p "$ENV" python=3.11
"$ENV/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cu124
"$ENV/bin/pip" install "transformers==4.57.1" accelerate einops requests pillow numpy

"$ENV/bin/python" -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"
