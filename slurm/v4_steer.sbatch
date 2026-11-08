#!/bin/bash
# v4 follow-up: steering on the test questions, Qwen3-VL or Gemma 4 (needs directions.npz first).
#   sbatch slurm/v4_steer.sbatch --model qwen3vl --limit 8 --tag _smoke
#   sbatch slurm/v4_steer.sbatch --model qwen3vl
#SBATCH --job-name=pp_v4_steer
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
source setup/paths.sh
export HF_HUB_OFFLINE=1

"$CONDA_ENVS/vlm/bin/python" -u repro/v4/steer.py "$@"
