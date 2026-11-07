#!/bin/bash
# v4 candidate renders: one GPU, several renderer processes on it. Submit from the repo root:
#   sbatch slurm/v4_render.sbatch smoke                 # 1 train + 1 test room, 60 photos each, raw renders kept
#   sbatch slurm/v4_render.sbatch full <train> <test>   # all rooms; A photos per train / test room
# Then: python repro/v4/balance.py --set train (and test) on the login node.
#SBATCH --job-name=pp_v4_render
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
source setup/paths.sh
PY="$CONDA_ENVS/habitat/bin/python -u -X faulthandler"

if [ "$1" = smoke ]; then
  $PY repro/v4/render.py --set smoke --split train --rooms v3_sc0_staging_01 --photos 60 --debug &
  $PY repro/v4/render.py --set smoke --split test --rooms apt_0 --photos 60 --debug &
  wait
  exit 0
fi

W=4  # processes per split, all on this GPU
for i in $(seq 0 $((W - 1))); do
  $PY repro/v4/render.py --set train --split train --worker "$i" --workers $W --photos "$2" > "logs/pp_v4_render_${SLURM_JOB_ID}_train$i.out" 2>&1 &
  $PY repro/v4/render.py --set test --split test --worker "$i" --workers $W --photos "$3" > "logs/pp_v4_render_${SLURM_JOB_ID}_test$i.out" 2>&1 &
done
wait
grep -h "photos," logs/pp_v4_render_${SLURM_JOB_ID}_*.out
