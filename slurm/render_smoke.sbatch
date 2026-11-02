#!/bin/bash
# Smoke test of the cross_line sampler on two rooms. Submit from the repo root:
#   sbatch slurm/render_smoke.sbatch
#SBATCH --job-name=pp_render_smoke
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

set +e
$PY repro/render/render_examples.py --set smoke --scene apt_0 --n 5 --seed 0
$PY repro/render/render_examples.py --set smoke --scene v3_sc0_staging_00 --n 5 --seed 0
