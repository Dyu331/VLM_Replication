"""Dump every object in every usable ReplicaCAD scene, to build the X/Y vocabulary from.

Writes $PP_DATA/scene_index/<scene>.json (per-object handle, template, motion type,
world AABB, size) and $PP_DATA/scene_index/templates.tsv (template -> #scenes, #instances,
median size). No rendering, so it is fast; still needs a GPU node for the EGL context.

    python repro/render/scene_index.py                 # all scenes
    python repro/render/scene_index.py --scenes apt_0  # subset
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import PP_DATA
from common.sim import SCENES, all_objects, make_sim, template_name, world_aabb


def index_scene(scene):
    sim = make_sim(scene, semantic=False)
    rows = []
    for obj in all_objects(sim):
        lo, hi = world_aabb(obj)
        rows.append({
            "handle": obj.handle,
            "template": template_name(obj.handle),
            "articulated": not hasattr(obj, "linear_velocity"),
            "motion_type": str(obj.motion_type).split(".")[-1],
            "aabb_min": lo.round(3).tolist(),
            "aabb_max": hi.round(3).tolist(),
            "size": (hi - lo).round(3).tolist(),
        })
    info = {"scene": scene, "navigable_area": round(sim.pathfinder.navigable_area, 2),
            "objects": rows}
    sim.close()
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=SCENES)
    args = ap.parse_args()

    out = PP_DATA / "scene_index"
    out.mkdir(parents=True, exist_ok=True)
    stats = defaultdict(lambda: {"scenes": set(), "count": 0, "sizes": []})
    for scene in args.scenes:
        try:
            info = index_scene(scene)
        except Exception as e:  # keep going; report broken scenes at the end
            print(f"{scene}: FAILED {e}", flush=True)
            continue
        (out / f"{scene}.json").write_text(json.dumps(info, indent=1))
        for r in info["objects"]:
            s = stats[r["template"]]
            s["scenes"].add(scene)
            s["count"] += 1
            s["sizes"].append(r["size"])
        print(f"{scene}: {len(info['objects'])} objects, "
              f"navigable {info['navigable_area']} m^2", flush=True)

    with open(out / "templates.tsv", "w") as f:
        f.write("template\tn_scenes\tn_instances\tmedian_size_xyz\n")
        for t, s in sorted(stats.items(), key=lambda kv: -len(kv[1]["scenes"])):
            med = np.median(np.array(s["sizes"]), axis=0).round(2).tolist()
            f.write(f"{t}\t{len(s['scenes'])}\t{s['count']}\t{med}\n")
    print(f"\n{len(stats)} templates -> {out / 'templates.tsv'}")


if __name__ == "__main__":
    main()
