"""Pick and download HSSD object models for the 13 v4 categories.

Step 1 (login node, vlm env: needs pandas + huggingface_hub):
    python repro/v4/assets.py candidates
  Downloads semantics/objects.csv if needed, filters each category by WordNet key, single object,
  plausible size and file size, draws CANDIDATES_PER_CAT at random (seed 0), downloads their
  .glb / .collider.glb / .object_config.json, and writes $V4_DATA/assets_candidates.json.

Step 2 (GPU, habitat env): repro/v4/thumbs.py renders a contact sheet of the candidates.

Step 3 (after looking at the sheet):
    python repro/v4/assets.py keep --drop <id> <id> ...
  Keeps the first MODELS_PER_CAT candidates per category that were not dropped and writes
  $V4_DATA/assets.json, the pool render.py uses.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v4.config_v4 import (CANDIDATES_PER_CAT, CATEGORIES, HSSD_DIR, MAX_GLB_MB, MODELS_PER_CAT,
                          SIZE_RANGE, TEST_CATS, V4_DATA)

REPO = "hssd/hssd-hab"


def candidates():
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download

    HSSD_DIR.mkdir(parents=True, exist_ok=True)
    csv = HSSD_DIR / "semantics/objects.csv"
    if not csv.exists():
        hf_hub_download(REPO, "semantics/objects.csv", repo_type="dataset", local_dir=HSSD_DIR)
    df = pd.read_csv(csv)
    sizes = {s.rfilename: s.size for s in HfApi().dataset_info(REPO, files_metadata=True).siblings
             if s.rfilename.startswith("objects/") and s.rfilename.count("/") == 2}

    out = []
    for cat, keys in CATEGORIES.items():
        d = df[df.wnsynsetkey.isin(keys) & (df.hasMultipleObjects.astype(str).str.lower() != "true")]
        d = d[d["aligned.dims"].notna()].copy()
        dims = d["aligned.dims"].str.split(",", expand=True).astype(float)
        d["w"], d["h"], d["dpt"] = dims[0], dims[1], dims[2]
        (w0, w1), (h0, h1) = SIZE_RANGE[cat]
        width = d[["w", "dpt"]].max(axis=1)
        d = d[(width >= w0) & (width <= w1) & (d.h >= h0) & (d.h <= h1)]
        d["glb"] = [f"objects/{i[0]}/{i}.glb" for i in d.id]
        d["mb"] = [sizes.get(g, 0) / 1e6 for g in d.glb]
        d = d[(d.mb > 0) & (d.mb <= MAX_GLB_MB)]
        pick = d.sample(min(CANDIDATES_PER_CAT, len(d)), random_state=0)
        print(f"{cat:13s} {len(d):4d} eligible, {len(pick)} picked, {pick.mb.sum():.0f} MB")
        for _, r in pick.iterrows():
            stem = f"objects/{r.id[0]}/{r.id}"
            for suffix in (".glb", ".collider.glb", ".object_config.json"):
                if f"{stem}{suffix}" in sizes and not (HSSD_DIR / f"{stem}{suffix}").exists():
                    hf_hub_download(REPO, f"{stem}{suffix}", repo_type="dataset", local_dir=HSSD_DIR)
            out.append({"id": r.id, "category": cat, "split": "test" if cat in TEST_CATS else "train",
                        "hssd_name": r["name"], "wnsynsetkey": r.wnsynsetkey,
                        "dims": [r.w, r.h, r.dpt], "mb": round(r.mb, 1),
                        "config": str(HSSD_DIR / f"{stem}.object_config.json")})
    V4_DATA.mkdir(parents=True, exist_ok=True)
    path = V4_DATA / "assets_candidates.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"{len(out)} candidates -> {path}")


def keep(drop):
    cands = json.loads((V4_DATA / "assets_candidates.json").read_text())
    kept, per = [], {}
    for c in cands:
        if c["id"] in drop or per.get(c["category"], 0) >= MODELS_PER_CAT:
            continue
        per[c["category"]] = per.get(c["category"], 0) + 1
        kept.append(c)
    short = [k for k in CATEGORIES if per.get(k, 0) < MODELS_PER_CAT]
    (V4_DATA / "assets.json").write_text(json.dumps(kept, indent=1))
    print(f"kept {len(kept)} models: {per}" + (f"; SHORT: {short}" if short else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["candidates", "keep"])
    ap.add_argument("--drop", nargs="*", default=[])
    a = ap.parse_args()
    candidates() if a.step == "candidates" else keep(set(a.drop))
