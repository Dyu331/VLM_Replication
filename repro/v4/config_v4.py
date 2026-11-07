"""Settings for the v4 replication ("Three Random Objects, HSSD pool"). See PLAN_V4.md.

Tags: [theirs] = from the PhD's v4 report; [ours] = left open by the report, our choice.
"""
from common.config import PP_DATA, PP_RESULTS, TURBO

HSSD_DIR = TURBO / "datasets/hssd"            # partial copy of hf.co/datasets/hssd/hssd-hab
V4_DATA = PP_DATA / "v4"                      # navmeshes, assets.json, renders, questions
V4_RESULTS = PP_RESULTS / "v4"

# ---------- object pool [theirs: categories and split]; HSSD WordNet keys [ours] ----------
# prompt name -> HSSD semantics/objects.csv wnsynsetkey values
CATEGORIES = {
    # train
    "bench": ["bench.n.01"],
    "cabinet": ["cabinet.n.01"],
    "chair": ["chair.n.01"],
    "couch": ["sofa.n.01"],
    "dresser": ["chest_of_drawers.n.01"],
    "potted plant": ["potted_plant.n.01"],
    "stool": ["stool.n.01"],
    "table": ["dining_table.n.01", "table.n.02"],
    # test
    "bed": ["bed.n.01", "double_bed.n.01"],
    "bookshelf": ["bookcase.n.01"],
    "office chair": ["swivel_chair.n.01"],
    "refrigerator": ["refrigerator.n.01"],
    "trash can": ["ashcan.n.01"],
}
TRAIN_CATS = ["bench", "cabinet", "chair", "couch", "dresser", "potted plant", "stool", "table"]
TEST_CATS = ["bed", "bookshelf", "office chair", "refrigerator", "trash can"]

# plausible real-world size per category, metres: (min, max) of (width = larger floor side, height) [ours]
SIZE_RANGE = {
    "bench": ((0.8, 2.2), (0.35, 1.1)),
    "cabinet": ((0.5, 2.0), (0.6, 2.1)),
    "chair": ((0.4, 0.9), (0.7, 1.2)),
    "couch": ((1.4, 3.0), (0.6, 1.1)),
    "dresser": ((0.6, 2.0), (0.7, 1.6)),
    "potted plant": ((0.3, 1.2), (0.6, 2.0)),
    "stool": ((0.25, 0.7), (0.35, 0.9)),
    "table": ((0.8, 2.4), (0.6, 0.9)),
    "bed": ((1.4, 2.6), (0.4, 1.6)),
    "bookshelf": ((0.5, 2.0), (1.0, 2.3)),
    "office chair": ((0.45, 0.9), (0.8, 1.4)),
    "refrigerator": ((0.5, 1.2), (1.4, 2.2)),
    "trash can": ((0.2, 0.7), (0.3, 1.0)),
}
CANDIDATES_PER_CAT = 12   # downloaded and shown on a contact sheet [ours]
MODELS_PER_CAT = 5        # kept after the visual check [ours]
MAX_GLB_MB = 40           # skip very heavy models [ours]

# ---------- rooms: 34 [theirs: count]; which ones [ours] ----------
TEST_ROOMS_V4 = [f"apt_{i}" for i in range(6)] + [
    "v3_sc0_staging_00", "v3_sc1_staging_00", "v3_sc2_staging_00"]
TRAIN_ROOMS_V4 = [f"v3_sc{m}_staging_{k:02d}" for m in range(4) for k in range(1, 21, 3)][:25]
DOOR_TEMPLATES = ("door",)  # kept when the room is emptied [theirs: "doors kept"]

# ---------- camera [ours; same as the first replication] ----------
RES = 512
HFOV = 90.0
EYE = 1.5          # metres above the floor, for A and for B's spot

# ---------- floor grid [ours] ----------
GRID_STEP = 0.5            # metres between grid lines
GRID_FLOOR = (205, 205, 200)
GRID_LINE = (120, 120, 120)
GRID_SKY = (235, 235, 235)
SHADOW = 0.55              # brightness under an object's footprint (soft contact shadow)

# ---------- placement and validity ----------
OBJ_MIN_GAP = 1.0          # metres between object centres [ours]; B >= 0.8 m from X, Y is [theirs]
A_DIST = (1.5, 7.0)        # A to each object centre, metres [theirs]
MIN_ANGLE_A = 10.0         # degrees between X and Y from A [theirs]
MIN_ANGLE_B = 15.0         # degrees between X and Y from B's spot [theirs]
MIN_B_LINE = 0.5           # B's distance to the X-Y line, metres [theirs]
MIN_B_XY = 0.8             # B's distance to X and to Y, metres [theirs]
MIN_VIS_PCT = 0.3          # % of the frame each object covers, in each view [ours]
EDGE_PX = 2                # masks must not touch the image border (not cut off) [ours]

# ---------- selection [theirs: sizes] ----------
TARGET = {"train": 1520, "test": 1132}
SHORTCUT_MAX = 0.60        # metadata shortcut check bar [theirs]
