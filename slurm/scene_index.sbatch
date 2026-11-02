#!/bin/bash
# Index all 90 usable ReplicaCAD scenes. Submit from the repo root:
#   sbatch slurm/scene_index.sbatch
#SBATCH --job-name=pp_scene_index
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
module load python3.11-anaconda  # not strictly needed: we call the Turbo env python directly
source setup/paths.sh

"$CONDA_ENVS/habitat/bin/python" -u -X faulthandler \
  repro/render/scene_index.py
