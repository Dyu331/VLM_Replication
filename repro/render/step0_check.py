"""Phase-0 check: can we get per-object instance masks from a ReplicaCAD scene?

Loads one scene, gives every rigid and articulated object its own semantic id,
renders RGB + semantic from a navigable point at 4 headings, and reports how many
objects show up in the masks. Run on a GPU node (headless EGL):

    python repro/render/step0_check.py --dataset interactive --scene apt_0
    python repro/render/step0_check.py --dataset baked --scene Baked_sc0_staging_00

If the baked scene reports ~0 objects (furniture merged into the stage), use the
v3 layouts from the interactive dataset instead.
"""
import argparse
import faulthandler
import json
import sys
from pathlib import Path

import cv2
import magnum as mn
import numpy as np
import habitat_sim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import BAKED_CFG, EYE, HFOV, INTERACTIVE_CFG, PP_DATA, RES


def make_sim(cfg_file, scene, semantic=True):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = str(cfg_file)
    sim_cfg.scene_id = scene
    sim_cfg.enable_physics = True  # needed to load articulated objects (fridge, counter, ...)

    sensors = []
    kinds = [("rgb", habitat_sim.SensorType.COLOR)]
    if semantic:
        kinds.append(("semantic", habitat_sim.SensorType.SEMANTIC))
    for uuid, kind in kinds:
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid, spec.sensor_type = uuid, kind
        spec.resolution = list(RES)
        spec.hfov = HFOV
        spec.position = [0.0, EYE, 0.0]
        sensors.append(spec)
    agent_cfg = habitat_sim.agent.AgentConfiguration(sensor_specifications=sensors)
    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def assign_semantic_ids(sim):
    """Give every object its own id (1..N); 0 stays the stage. Returns {id: handle}."""
    objs = list(sim.get_rigid_object_manager().get_objects_by_handle_substring().values())
    objs += list(sim.get_articulated_object_manager().get_objects_by_handle_substring().values())
    id_to_handle = {}
    for i, obj in enumerate(objs, start=1):
        for node in obj.visual_scene_nodes:
            node.semantic_id = i
        id_to_handle[i] = obj.handle
    return id_to_handle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["interactive", "baked"], default="interactive")
    ap.add_argument("--scene", default="apt_0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-semantic", action="store_true", help="RGB only, to isolate crashes")
    args = ap.parse_args()
    faulthandler.enable()

    cfg_file = INTERACTIVE_CFG if args.dataset == "interactive" else BAKED_CFG
    print("creating simulator ...", flush=True)
    sim = make_sim(cfg_file, args.scene, semantic=not args.no_semantic)
    print("simulator ready; assigning semantic ids ...", flush=True)
    id_to_handle = assign_semantic_ids(sim)
    print(f"{len(id_to_handle)} objects in scene {args.scene}")
    print("navmesh loaded:", sim.pathfinder.is_loaded)

    sim.pathfinder.seed(args.seed)
    pos = sim.pathfinder.get_random_navigable_point()
    agent = sim.get_agent(0)
    out = PP_DATA / "step0" / f"{args.dataset}_{args.scene}"
    out.mkdir(parents=True, exist_ok=True)

    seen = set()
    for k, yaw in enumerate([0, 90, 180, 270]):
        state = habitat_sim.AgentState()
        state.position = pos
        rot = mn.Quaternion.rotation(mn.Deg(yaw), mn.Vector3.y_axis())
        state.rotation = np.quaternion(rot.scalar, *rot.vector)
        agent.set_state(state)
        print(f"rendering heading {yaw} ...", flush=True)
        obs = sim.get_sensor_observations()
        cv2.imwrite(str(out / f"rgb_{k}.png"), cv2.cvtColor(obs["rgb"][..., :3], cv2.COLOR_RGB2BGR))
        if args.no_semantic:
            continue
        sem = obs["semantic"].astype(np.int32)
        ids, counts = np.unique(sem, return_counts=True)
        visible = {int(i): int(c) for i, c in zip(ids, counts) if i != 0}
        seen |= visible.keys()
        # colourise the mask for a quick look
        rng = np.random.default_rng(0)
        palette = rng.integers(0, 255, size=(max(id_to_handle, default=0) + 1, 3), dtype=np.uint8)
        palette[0] = 0
        cv2.imwrite(str(out / f"sem_{k}.png"), palette[np.clip(sem, 0, len(palette) - 1)])
        print(f"heading {yaw:3d}: {len(visible)} objects visible")

    report = {"scene": args.scene, "dataset": args.dataset, "n_objects": len(id_to_handle),
              "n_seen": len(seen), "seen": {i: id_to_handle.get(i, "?") for i in sorted(seen)},
              "all_objects": id_to_handle}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nseen {len(seen)} distinct objects; images + report in {out}")
    sim.close()


if __name__ == "__main__":
    main()
