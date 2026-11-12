#!/bin/bash
# Steering redesign, Phase 2: gradients of the final lean, Qwen3-VL or Gemma 4.
#   sbatch slurm/v4_grads.sbatch --model qwen3vl --limit 8 --fd 8 --tag _smoke
#   sbatch slurm/v4_grads.sbatch --model qwen3vl
#SBATCH --job-name=pp_v4_grads
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
source setup/paths.sh
export HF_HUB_OFFLINE=1

"$CONDA_ENVS/vlm/bin/python" -u repro/v4/grads.py "$@"
