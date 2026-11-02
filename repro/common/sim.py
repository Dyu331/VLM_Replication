"""Simulator helpers shared by the render scripts."""
import re

import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis
import numpy as np

from common.config import EYE, HFOV, INTERACTIVE_CFG, RES

# Usable scenes in replica_cad_dataset v1.6 (v3_sc4_* are listed in the config but not shipped)
SCENES = [f"apt_{i}" for i in range(6)] + [
    f"v3_sc{m}_staging_{k:02d}" for m in range(4) for k in range(21)
]


def make_sim(scene, semantic=True, cfg_file=INTERACTIVE_CFG):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = str(cfg_file)
    sim_cfg.scene_id = scene
    sim_cfg.enable_physics = True  # needed to load articulated objects (fridge, counter, ...)

    kinds = [("rgb", habitat_sim.SensorType.COLOR)]
    if semantic:
        kinds.append(("semantic", habitat_sim.SensorType.SEMANTIC))
    sensors = []
    for uuid, kind in kinds:
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid, spec.sensor_type = uuid, kind
        spec.resolution = list(RES)
        spec.hfov = HFOV
        spec.position = [0.0, EYE, 0.0]
        sensors.append(spec)
    agent_cfg = habitat_sim.agent.AgentConfiguration(sensor_specifications=sensors)
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    if not sim.pathfinder.is_loaded:  # apt_* ship without a navmesh
        ensure_navmesh(sim)
    return sim


def ensure_navmesh(sim):
    """Build a navmesh from the stage + static furniture when the scene has none."""
    settings = habitat_sim.NavMeshSettings()
    settings.set_defaults()
    settings.agent_radius = 0.3  # same radius as ReplicaCAD's shipped navmeshes
    settings.include_static_objects = True
    sim.recompute_navmesh(sim.pathfinder, settings)


def all_objects(sim):
    """Rigid + articulated objects, in a fixed order."""
    objs = list(sim.get_rigid_object_manager().get_objects_by_handle_substring().values())
    objs += list(sim.get_articulated_object_manager().get_objects_by_handle_substring().values())
    return sorted(objs, key=lambda o: o.handle)


def assign_semantic_ids(sim):
    """Give every object its own id (1..N); 0 stays the stage. Returns {id: object}."""
    id_to_obj = {}
    for i, obj in enumerate(all_objects(sim), start=1):
        for node in obj.visual_scene_nodes:
            node.semantic_id = i
        id_to_obj[i] = obj
    return id_to_obj


def template_name(handle):
    """'frl_apartment_sofa_:0000' -> 'frl_apartment_sofa'."""
    return re.sub(r"_?:\d+$", "", handle)


def world_aabb(obj):
    """Axis-aligned world bounding box (min, max) over all visual nodes of an object.

    Skips empty nodes (zero-size local box): kitchen_counter has two per link, placed
    metres away, which otherwise stretch its box to ~8 m.
    """
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for node in obj.visual_scene_nodes:
        local = node.cumulative_bb
        if np.allclose(np.array(local.max) - np.array(local.min), 0):
            continue
        bb = habitat_sim.geo.get_transformed_bb(local, node.absolute_transformation())
        lo = np.minimum(lo, np.array(bb.min))
        hi = np.maximum(hi, np.array(bb.max))
    return lo, hi


def set_pose(sim, position, yaw_deg):
    """Place the agent at a floor position facing yaw_deg (0 = -Z, positive = turn left)."""
    state = habitat_sim.AgentState()
    state.position = np.asarray(position, dtype=np.float32)
    state.rotation = quat_from_angle_axis(np.deg2rad(yaw_deg), np.array([0.0, 1.0, 0.0]))
    sim.get_agent(0).set_state(state)
