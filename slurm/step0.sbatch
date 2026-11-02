#!/bin/bash
# Phase-0 render check on Great Lakes. Submit from the repo root:
#   sbatch --account=<your_slurm_account> slurm/step0.sbatch
#SBATCH --job-name=pp_step0
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
source setup/paths.sh
PY="conda run --no-capture-output -p $CONDA_ENVS/habitat python"

$PY repro/render/step0_check.py --dataset interactive --scene apt_0
$PY repro/render/step0_check.py --dataset interactive --scene v3_sc0_staging_00
$PY repro/render/step0_check.py --dataset baked --scene Baked_sc0_staging_00
