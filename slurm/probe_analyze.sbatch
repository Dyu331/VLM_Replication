#!/bin/bash
# Probe analysis (CPU only): readers per layer, checks, chance band, bootstrap. Submit from the repo root:
#   sbatch slurm/probe_analyze.sbatch --model qwen3vl
# If the account cannot use the standard partition, use: sbatch --partition=spgpu --gpus=1 ...
#SBATCH --job-name=pp_probe_analyze
#SBATCH --account=chaijy2
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
source setup/paths.sh
export OMP_NUM_THREADS=8

"$CONDA_ENVS/vlm/bin/python" -u repro/probing/analyze.py "$@"
