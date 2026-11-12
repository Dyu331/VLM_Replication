#!/bin/bash
# Steering redesign (PLAN_V4_STEER.md), CPU part for one model.
#   sbatch slurm/v4_steer2_dirs.sbatch reliability --model qwen3vl      (also gemma4, bagel)
#   sbatch slurm/v4_steer2_dirs.sbatch build --model qwen3vl            (after grads.py)
#SBATCH --job-name=pp_v4_steer2_dirs
#SBATCH --account=chaijy2
#SBATCH --partition=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
source setup/paths.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

"$CONDA_ENVS/vlm/bin/python" -u repro/v4/steer2_dirs.py "$@"
