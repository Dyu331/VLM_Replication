#!/bin/bash
# v4 Phase 4: readers and controls for one model (CPU only). Submit from the repo root:
#   sbatch slurm/v4_analyze.sbatch --model qwen3vl --layers 0 20     # quick test
#   sbatch slurm/v4_analyze.sbatch --model qwen3vl                   # all layers (also gemma4, bagel)
#SBATCH --job-name=pp_v4_analyze
#SBATCH --account=chaijy2
#SBATCH --partition=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
source setup/paths.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

"$CONDA_ENVS/vlm/bin/python" -u repro/v4/analyze.py "$@"
