#!/usr/bin/env bash
# Downloads ReplicaCAD (interactive + baked lighting, ~700 MB) to Turbo. Run once, from a login node.
set -euo pipefail
source "$(dirname "$0")/paths.sh"

# remove partial clones from an earlier failed attempt
rm -rf "$REPLICA_DIR/versioned_data/replica_cad_dataset" "$REPLICA_DIR/versioned_data/replica_cad_baked_lighting" \
  "$REPLICA_DIR/replica_cad" "$REPLICA_DIR/replica_cad_baked_lighting"
conda run -p "$CONDA_ENVS/habitat" git lfs install
conda run -p "$CONDA_ENVS/habitat" python -m habitat_sim.utils.datasets_download \
  --uids replica_cad_dataset replica_cad_baked_lighting \
  --data-path "$REPLICA_DIR"

ls -la "$REPLICA_DIR"
