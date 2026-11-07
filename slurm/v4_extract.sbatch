#!/bin/bash
# v4 Phase 3: answers + hidden states for Qwen3-VL or Gemma 4, train and test, in one GPU job.
# Submit from the repo root:
#   sbatch slurm/v4_extract.sbatch --model qwen3vl --limit 8 --tag _smoke   # smoke test
#   sbatch slurm/v4_extract.sbatch --model qwen3vl                          # full run (also: gemma4)
#SBATCH --job-name=pp_v4_extract
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

"$CONDA_ENVS/vlm/bin/python" -u repro/v4/extract.py "$@"
