#!/bin/bash
# Render more examples for every room of a set in ONE GPU job, 4 rooms at a time.
# Submit from the repo root:
#   sbatch slurm/render_topup.sbatch test 8 3     # <set> <n_per_room> <seed>
#SBATCH --job-name=pp_topup
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
source setup/paths.sh
PY="$CONDA_ENVS/habitat/bin/python -u -X faulthandler"

SET=$1; N=$2; SEED=$3
ROOMS_VAR=$([ "$SET" = train ] && echo TRAIN_ROOMS || echo TEST_ROOMS)
ROOMS=$($PY -c "import sys; sys.path.insert(0, 'repro'); from common.config import $ROOMS_VAR as R; print(' '.join(R))")
mkdir -p logs/rooms

for SCENE in $ROOMS; do
  $PY repro/render/render_examples.py --set "$SET" --scene "$SCENE" --n "$N" --seed "$SEED" \
    > "logs/rooms/${SET}_${SCENE}_s${SEED}.out" 2>&1 &
  while [ "$(jobs -r | wc -l)" -ge 4 ]; do sleep 5; done
done
wait
grep -h "examples in" logs/rooms/${SET}_*_s${SEED}.out
