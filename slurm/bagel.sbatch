#!/bin/bash
# BAGEL-7B-MoT answers for all settings on the test set. Submit from the repo root:
#   sbatch slurm/bagel.sbatch --limit 5   # smoke test
#   sbatch slurm/bagel.sbatch             # full run
#SBATCH --job-name=pp_bagel
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
source setup/paths.sh
export HF_HUB_OFFLINE=1

"$CONDA_ENVS/bagel/bin/python" -u repro/eval/run_bagel.py "$@"
