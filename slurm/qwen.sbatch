#!/bin/bash
# Qwen3-VL answers for all settings, in one GPU job.
# Submit from the repo root:
#   sbatch slurm/qwen.sbatch --model <path>       # full run on the test set
#   sbatch slurm/qwen.sbatch --model <path> --limit 5   # smoke test
#SBATCH --job-name=pp_qwen
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
source setup/paths.sh
export HF_HUB_OFFLINE=1  # weights are pre-downloaded on Turbo

"$CONDA_ENVS/vlm/bin/python" -u repro/eval/run_qwen.py "$@"
