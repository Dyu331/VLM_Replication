#!/bin/bash
# Phase-0 render check on Great Lakes. Submit from the repo root:
#   sbatch slurm/step0.sbatch
#SBATCH --job-name=pp_step0
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
module load python3.11-anaconda  # not strictly needed: we call the Turbo env python directly
source setup/paths.sh
PY="$CONDA_ENVS/habitat/bin/python -u -X faulthandler"  # direct, so conda run cannot grab script flags like --n

# keep going after a crash so one log shows every case
set +e
echo "=== 1. apt_0 (no shipped navmesh) ==="
$PY repro/render/step0_check.py --dataset interactive --scene apt_0
echo "=== 2. v3_sc4_staging_00 (no shipped navmesh) ==="
$PY repro/render/step0_check.py --dataset interactive --scene v3_sc4_staging_00
echo "=== 3. v3_sc0_staging_00 (shipped navmesh) ==="
$PY repro/render/step0_check.py --dataset interactive --scene v3_sc0_staging_00
