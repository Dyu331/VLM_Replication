#!/bin/bash
# Short GPU job for one-off debug scripts. Submit from the repo root:
#   sbatch slurm/debug.sbatch repro/render/debug_articulated.py --scene v3_sc0_staging_00
#SBATCH --job-name=pp_debug
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
module load python3.11-anaconda  # provides conda; the env itself lives on Turbo
source setup/paths.sh

conda run --no-capture-output -p "$CONDA_ENVS/habitat" python -u -X faulthandler "$@"
