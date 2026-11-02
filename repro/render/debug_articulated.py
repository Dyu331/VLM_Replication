"""Print per-link bounding boxes of articulated objects, to see why kitchen_counter's AABB is ~8 m.

    python repro/render/debug_articulated.py --scene v3_sc0_staging_00
"""
import argparse
import sys
from pathlib import Path

import habitat_sim
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.sim import make_sim


def bb_of(nodes):
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for node in nodes:
        bb = habitat_sim.geo.get_transformed_bb(node.cumulative_bb, node.absolute_transformation())
        lo, hi = np.minimum(lo, np.array(bb.min)), np.maximum(hi, np.array(bb.max))
    return lo.round(2), hi.round(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="v3_sc0_staging_00")
    ap.add_argument("--detail", default="kitchen_counter", help="print every node of this object")
    args = ap.parse_args()
    sim = make_sim(args.scene, semantic=False)
    aom = sim.get_articulated_object_manager()
    for handle, ao in aom.get_objects_by_handle_substring().items():
        print(f"\n== {handle}  root at {np.array(ao.translation).round(2)}  links={ao.num_links}")
        lo, hi = bb_of(ao.visual_scene_nodes)
        print(f"   all visual nodes: min {lo} max {hi} size {(hi - lo).round(2)}")
        for link_id in [-1] + list(ao.get_link_ids()):
            nodes = ao.get_link_visual_nodes(link_id)
            if not nodes:
                continue
            lo, hi = bb_of(nodes)
            print(f"   link {link_id:3d} {ao.get_link_name(link_id):30s} "
                  f"min {lo} max {hi} size {(hi - lo).round(2)}  ({len(nodes)} nodes)")
            if args.detail and args.detail in handle:
                for node in nodes:
                    bb = node.cumulative_bb
                    t = node.absolute_transformation()
                    print(f"        node local min {np.array(bb.min).round(2)} max {np.array(bb.max).round(2)}"
                          f"  world pos {np.array(t.translation).round(2)}"
                          f"  scale {np.array(t.scaling()).round(2)}  children {len(node.children)}")
    sim.close()


if __name__ == "__main__":
    main()
