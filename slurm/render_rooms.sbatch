#!/bin/bash
# Render many rooms in ONE GPU job, running P rooms at a time on the same GPU.
# Rendering is light, so one A40 handles several habitat processes at once.
# Submit from the repo root:
#   sbatch slurm/render_rooms.sbatch <set> <n_per_room> <first_idx> <last_idx> [P]
#   e.g. sbatch slurm/render_rooms.sbatch train 15 10 44 8
#SBATCH --job-name=pp_render_rooms
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
source setup/paths.sh
PY="$CONDA_ENVS/habitat/bin/python -u -X faulthandler"

SET=$1; N=$2; FIRST=$3; LAST=$4; P=${5:-8}
ROOMS_VAR=$([ "$SET" = train ] && echo TRAIN_ROOMS || echo TEST_ROOMS)
SEED=$([ "$SET" = train ] && echo 2 || echo 1)
mkdir -p logs/rooms

for i in $(seq "$FIRST" "$LAST"); do
  SCENE=$($PY -c "import sys; sys.path.insert(0, 'repro'); from common.config import $ROOMS_VAR as R; print(R[$i])")
  # one log per room; a crash in one room doesn't stop the others
  $PY repro/render/render_examples.py --set "$SET" --scene "$SCENE" --n "$N" --seed "$SEED" \
    > "logs/rooms/${SET}_${SCENE}.out" 2>&1 &
  while [ "$(jobs -r | wc -l)" -ge "$P" ]; do sleep 5; done
done
wait
grep -h "examples in" logs/rooms/${SET}_*.out
