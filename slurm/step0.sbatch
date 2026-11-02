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
module load python3.11-anaconda  # provides conda; the env itself lives on Turbo
source setup/paths.sh
PY="conda run --no-capture-output -p $CONDA_ENVS/habitat python -u -X faulthandler"

# keep going after a crash so one log shows every case
set +e
echo "=== 1. apt_0, RGB only ==="
$PY repro/render/step0_check.py --dataset interactive --scene apt_0 --no-semantic
echo "=== 2. apt_0, RGB + semantic ==="
$PY repro/render/step0_check.py --dataset interactive --scene apt_0
echo "=== 3. v3_sc0_staging_00, RGB + semantic ==="
$PY repro/render/step0_check.py --dataset interactive --scene v3_sc0_staging_00
echo "=== 4. Baked_sc0_staging_00, RGB + semantic ==="
$PY repro/render/step0_check.py --dataset baked --scene Baked_sc0_staging_00
