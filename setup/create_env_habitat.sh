#!/usr/bin/env bash
# Creates the rendering env on Turbo so both Great Lakes and Whistler can use it.
# Run once, from a login node (needs internet). Requires conda/mamba on PATH
# (Great Lakes: `module load python3.11-anaconda` or a personal miniforge).
set -euo pipefail
source "$(dirname "$0")/paths.sh"

ENV=$CONDA_ENVS/habitat
# 0.3.3 is the newest version I'm confident exists on the aihabitat conda channel;
# override with HABITAT_VERSION=0.3.4 if `conda search -c aihabitat habitat-sim` lists it.
HABITAT_VERSION=${HABITAT_VERSION:-0.3.3}

# aihabitat only publishes habitat-sim 0.3.3 conda builds for python 3.9
conda create -y -p "$ENV" python=3.9 cmake=3.27
conda install -y -p "$ENV" "habitat-sim=$HABITAT_VERSION" withbullet headless \
  -c conda-forge -c aihabitat
# git-lfs is needed by datasets_download (no sudo on the clusters)
conda install -y -p "$ENV" git-lfs -c conda-forge
# habitat-sim 0.3.3 pins numpy==1.26.4 and pillow==10.4.0; opencv 4.10 is the last build on numpy 1.x
conda run -p "$ENV" pip install "numpy==1.26.4" "pillow==10.4.0" imageio-ffmpeg \
  "opencv-python-headless==4.10.0.84" "matplotlib<3.10" tqdm

conda run -p "$ENV" python -c "import habitat_sim; print('habitat-sim', habitat_sim.__version__)"
