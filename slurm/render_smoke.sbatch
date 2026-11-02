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
module load python3.11-anaconda  # not strictly needed: we call the Turbo env python directly
source setup/paths.sh
PY="$CONDA_ENVS/habitat/bin/python -u -X faulthandler"  # direct, so conda run cannot grab script flags like --n

set +e
$PY repro/render/render_examples.py --set smoke --scene apt_0 --n 5 --seed 0
$PY repro/render/render_examples.py --set smoke --scene v3_sc0_staging_00 --n 5 --seed 0
