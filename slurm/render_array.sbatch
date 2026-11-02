#!/bin/bash
# Render cross_line examples, one array task per room. Submit from the repo root:
#   sbatch --array=0-14 slurm/render_array.sbatch test 14    # 15 test rooms  x 14 = 210
#   sbatch --array=0-9  slurm/render_array.sbatch train 30   # 10 train rooms x 30 = 300
# Then, once all tasks finish: python repro/render/label_and_clean.py --set test (and train)
#SBATCH --job-name=pp_render
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
source setup/paths.sh
PY="$CONDA_ENVS/habitat/bin/python -u -X faulthandler"

SET=$1
N=$2
ROOMS_VAR=$([ "$SET" = train ] && echo TRAIN_ROOMS || echo TEST_ROOMS)
SCENE=$($PY -c "import sys; sys.path.insert(0, 'repro'); from common.config import $ROOMS_VAR as R; print(R[$SLURM_ARRAY_TASK_ID])")
echo "set=$SET scene=$SCENE n=$N"

# seed differs per set so a room reused across sets never repeats poses
SEED=$([ "$SET" = train ] && echo 2 || echo 1)
$PY repro/render/render_examples.py --set "$SET" --scene "$SCENE" --n "$N" --seed "$SEED"
