#!/bin/bash
# VLM answers for all settings on the test set, one model per job. Submit from the repo root:
#   sbatch slurm/vlm.sbatch --model qwen3vl --limit 5   # smoke test
#   sbatch slurm/vlm.sbatch --model qwen3vl             # full run (also: gemma4, molmo2)
#SBATCH --job-name=pp_vlm
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
source setup/paths.sh
export HF_HUB_OFFLINE=1  # weights are in the shared lab folder on Turbo

"$CONDA_ENVS/vlm/bin/python" -u repro/eval/run_vlm.py "$@"
