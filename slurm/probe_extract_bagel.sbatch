#!/bin/bash
# BAGEL hidden states (first generation step) + own answers for the probe. Submit from the repo root:
#   sbatch slurm/probe_extract_bagel.sbatch --limit 3   # smoke test
#   sbatch slurm/probe_extract_bagel.sbatch             # full run
#SBATCH --job-name=pp_probe_bagel
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
source setup/paths.sh
export HF_HUB_OFFLINE=1

"$CONDA_ENVS/bagel/bin/python" -u repro/probing/extract_bagel.py "$@"
