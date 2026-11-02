"""Paths and rendering constants shared by all scripts. Our own choices, not the report's."""
import os
from pathlib import Path

TURBO = Path(os.environ.get("TURBO", "/nfs/turbo/coe-chaijy-unreplicated/dannyyu"))
REPLICA_DIR = TURBO / "datasets/replica_cad"
PP_DATA = TURBO / "datasets/perspective_pilot"
PP_RESULTS = TURBO / "results/perspective_pilot"

# datasets_download creates these symlinks under REPLICA_DIR
INTERACTIVE_CFG = REPLICA_DIR / "replica_cad/replicaCAD.scene_dataset_config.json"
BAKED_CFG = REPLICA_DIR / "replica_cad_baked_lighting/replicaCAD_baked.scene_dataset_config.json"

# Camera
RES = (512, 512)  # (height, width)
HFOV = 90.0       # degrees
EYE = 1.5         # metres above the floor
