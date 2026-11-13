#!/bin/bash
# v4 follow-up: steering for BAGEL-7B-MoT (its own env; needs steer2/dirs.npz first).
#   sbatch slurm/v4_steer2_bagel.sbatch --limit 8 --tag _smoke
#   sbatch slurm/v4_steer2_bagel.sbatch
#SBATCH --job-name=pp_v4_steer2_bagel
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
source setup/paths.sh
export HF_HUB_OFFLINE=1

"$CONDA_ENVS/bagel/bin/python" -u repro/v4/steer2_bagel.py "$@"
