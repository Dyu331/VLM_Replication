#!/bin/bash
# Hidden states + own answers for the probe (Qwen3-VL or Gemma 4), train and test, in one GPU job.
# Submit from the repo root:
#   sbatch slurm/probe_extract.sbatch --model qwen3vl --limit 3   # smoke test
#   sbatch slurm/probe_extract.sbatch --model qwen3vl             # full run (also: gemma4)
#SBATCH --job-name=pp_probe_extract
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
source setup/paths.sh
export HF_HUB_OFFLINE=1

"$CONDA_ENVS/vlm/bin/python" -u repro/probing/extract_vlm.py "$@"
