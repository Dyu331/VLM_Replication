"""Paths and rendering constants shared by all scripts. Our own choices, not the report's."""
import os
from pathlib import Path

TURBO = Path(os.environ.get("TURBO", "/nfs/turbo/coe-chaijy-unreplicated/dannyyu"))
REPLICA_DIR = TURBO / "datasets/replica_cad"
PP_DATA = TURBO / "datasets/perspective_pilot"
PP_RESULTS = TURBO / "results/perspective_pilot"

# Shared lab model weights (plain HF folders)
WEIGHTS = Path("/nfs/turbo/coe-chaijy-unreplicated/pre-trained-weights")
QWEN_PATH = WEIGHTS / "Qwen3-VL-8B-Instruct"
BAGEL_PATH = WEIGHTS / "BAGEL-7B-MoT"
GEMMA_PATH = WEIGHTS / "gemma-4-12B-it"  # extension, not in the report
MOLMO_PATH = WEIGHTS / "Molmo2-8B"       # extension, not in the report

# datasets_download creates these symlinks under REPLICA_DIR
INTERACTIVE_CFG = REPLICA_DIR / "replica_cad/replicaCAD.scene_dataset_config.json"
BAKED_CFG = REPLICA_DIR / "replica_cad_baked_lighting/replicaCAD_baked.scene_dataset_config.json"

# Camera
RES = (512, 512)  # (height, width)
HFOV = 90.0       # degrees
EYE = 1.5         # metres above the floor

# Sampling (large cross_line set uses the pilot's stricter values)
MIN_MARGIN = 20.0    # degrees between X and Y bearings, from each observer
MIN_VIS = 1.5        # % of the frame each object must cover, in each view
B_DIST = (1.5, 4.0)  # metres B stands from the X-Y line, on the far side
YAW_JITTER = 15.0    # degrees of random heading offset from the X-Y midpoint

# Clean rule (applied after rendering, from the masks)
MIN_OBS_DIST = 1.5   # metres from each observer to each object's 3D centre (as in the report)
MIN_PX_GAP = 40      # px between the two objects' mask centres
# plus: the two masks must not overlap horizontally

# Room splits (disjoint; v3_sc4_* are not shipped)
TEST_ROOMS = [f"apt_{i}" for i in range(6)] + [
    "v3_sc0_staging_00", "v3_sc0_staging_10", "v3_sc0_staging_20",
    "v3_sc1_staging_00", "v3_sc1_staging_10",
    "v3_sc2_staging_00", "v3_sc2_staging_10",
    "v3_sc3_staging_00", "v3_sc3_staging_10",
]
TRAIN_ROOMS = [f"v3_sc{m}_staging_{k:02d}" for m in range(4) for k in (5, 15)] + [
    "v3_sc0_staging_03", "v3_sc1_staging_03",
]
# First train render gave only 90 clean from these 10 rooms; append the remaining v3 layouts
# (indices 10+ keep the first 10 stable, so already-rendered rooms are not redone)
TRAIN_ROOMS += [f"v3_sc{m}_staging_{k:02d}" for m in range(4) for k in range(21)
                if f"v3_sc{m}_staging_{k:02d}" not in TEST_ROOMS + TRAIN_ROOMS]
