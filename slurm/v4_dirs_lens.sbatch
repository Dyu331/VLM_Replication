#!/bin/bash
# v4 follow-ups, CPU part for one model: steering directions (training rooms), then logit lens + rotation check.
#   sbatch slurm/v4_dirs_lens.sbatch qwen3vl      (also gemma4, bagel)
#SBATCH --job-name=pp_v4_dirs_lens
#SBATCH --account=chaijy2
#SBATCH --partition=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
source setup/paths.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

"$CONDA_ENVS/vlm/bin/python" -u repro/v4/directions.py --model "$1"
"$CONDA_ENVS/vlm/bin/python" -u repro/v4/lens.py --model "$1"
