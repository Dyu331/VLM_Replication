"""One emptied ReplicaCAD room with HSSD objects in it: placement, rendering, background swap.

    room = Room("apt_0")                      # loads the room, removes movable objects (doors kept),
                                              # rebuilds the navmesh, finds the floor height
    o = room.add(asset, x, z, yaw, sid=1)     # HSSD object standing on the floor, or None if it
                                              # does not fit (footprint off the navmesh)
    rgb, sem, depth = room.render(xz, yaw)    # camera at EYE metres above the floor, no tilt
    img = room.composite(rgb, sem, depth, xz, yaw, keep={1, 2, 3})
    room.remove(o)

Geometry convention (habitat): y is up; yaw 0 looks along -Z; positive yaw turns left.
"""
import math
import sys
from pathlib import Path

import habitat_sim
import magnum as mn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import INTERACTIVE_CFG
from common.sim import all_objects, set_pose, world_aabb
from v4.config_v4 import (BG_GREY, DOOR_TEMPLATES, EYE, FLOOR_TOL, GRID_HALF_WIDTH, GRID_LINE,
                          GRID_STEP, HFOV, RES)

FOCAL = (RES / 2) / math.tan(math.radians(HFOV / 2))
DOOR_SID = 50


# ---------- plain geometry on the floor plane (x, z) ----------

def forward(yaw):
    r = math.radians(yaw)
    return np.array([-math.sin(r), 0.0, -math.cos(r)])


def right(yaw):
    r = math.radians(yaw)
    return np.array([math.cos(r), 0.0, -math.sin(r)])


def yaw_towards(src, dst):
    d = np.asarray(dst, float) - np.asarray(src, float)
    return math.degrees(math.atan2(-d[0], -d[2]))


def bearing(pos, yaw, target):
    """Angle of target from the view direction in degrees; positive = to the right."""
    d = np.asarray(target, float) - np.asarray(pos, float)
    d[1] = 0.0
    return math.degrees(math.atan2(d @ right(yaw), d @ forward(yaw)))


def floor_dist(p, q):
    return float(math.hypot(p[0] - q[0], p[2] - q[2]))


def side_of_line(p, a, b):
    """+1 / -1: which side of the floor line through a and b the point p is on."""
    return float(np.sign((b[0] - a[0]) * (p[2] - a[2]) - (b[2] - a[2]) * (p[0] - a[0])))


def dist_to_line(p, a, b):
    ax, az, bx, bz, px, pz = a[0], a[2], b[0], b[2], p[0], p[2]
    return abs((bx - ax) * (pz - az) - (bz - az) * (px - ax)) / max(math.hypot(bx - ax, bz - az), 1e-9)


def rects_overlap(p, q):
    """Separating-axis test for two convex floor polygons given as (4, 2) arrays of (x, z)."""
    for poly in (p, q):
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]
            n = np.array([-e[1], e[0]])
            a, b = p @ n, q @ n
            if a.max() < b.min() or b.max() < a.min():
                return False
    return True


def grow(poly, margin):
    c = poly.mean(axis=0)
    d = poly - c
    return c + d * (1 + margin / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9))


def inside(pt, poly):
    """Point (x, z) inside convex polygon."""
    s = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        s.append(np.sign((b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])))
    return all(v >= 0 for v in s) or all(v <= 0 for v in s)


# ---------- the room ----------

def _sim(scene):
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_dataset_config_file = str(INTERACTIVE_CFG)
    cfg.scene_id = scene
    cfg.enable_physics = True  # articulated furniture needs it to load; also gives cast_ray
    sensors = []
    for uuid, kind in (("rgb", habitat_sim.SensorType.COLOR),
                       ("semantic", habitat_sim.SensorType.SEMANTIC),
                       ("depth", habitat_sim.SensorType.DEPTH)):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid, spec.sensor_type = uuid, kind
        spec.resolution = [RES, RES]
        spec.hfov = HFOV
        spec.position = [0.0, EYE, 0.0]
        sensors.append(spec)
    agent = habitat_sim.agent.AgentConfiguration(sensor_specifications=sensors)
    return habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent]))


class Room:
    def __init__(self, scene, seed=0):
        self.scene = scene
        self.sim = sim = _sim(scene)
        self.rom = sim.get_rigid_object_manager()
        self.otm = sim.get_object_template_manager()
        self._templates = {}
        # empty the room: keep only doors (made STATIC so the navmesh respects them)
        aom = sim.get_articulated_object_manager()
        self.n_removed = 0
        for obj in all_objects(sim):
            if any(d in obj.handle for d in DOOR_TEMPLATES):
                obj.motion_type = habitat_sim.physics.MotionType.STATIC
                for node in obj.visual_scene_nodes:
                    node.semantic_id = DOOR_SID
                continue
            if hasattr(obj, "linear_velocity"):
                self.rom.remove_object_by_handle(obj.handle)
            else:
                aom.remove_object_by_handle(obj.handle)
            self.n_removed += 1
        settings = habitat_sim.NavMeshSettings()
        settings.set_defaults()
        settings.agent_radius = 0.3
        settings.include_static_objects = True
        sim.recompute_navmesh(sim.pathfinder, settings)
        sim.pathfinder.seed(seed)
        self.pf = sim.pathfinder
        self.nav_y = float(np.median([self.pf.get_random_navigable_point()[1] for _ in range(50)]))
        self.floor_y = self._floor_height()

    def _floor_height(self):
        ys = []
        for _ in range(20):
            p = np.array(self.pf.get_random_navigable_point())
            hit = self._first_hit(p + np.array([0, 1.0, 0]), np.array([0, -1.0, 0]), 3.0)
            if hit is not None:
                ys.append(float(hit[0][1]))
        return float(np.median(ys)) if ys else self.nav_y

    def close(self):
        self.sim.close()

    # ----- objects -----
    def template(self, asset):
        if asset["id"] not in self._templates:
            ids = self.otm.load_configs(asset["config"])
            if not ids:
                raise RuntimeError(f"could not load {asset['config']}")
            self._templates[asset["id"]] = self.otm.get_template_handle_by_id(ids[0])
        return self._templates[asset["id"]]

    def add(self, asset, x, z, yaw, sid, others=(), margin=0.2):
        """Place asset standing on the floor with its footprint centred at (x, z), turned by yaw.

        Returns a dict (object, centre, footprint, height) or None, after removing the object again,
        when the footprint leaves the navmesh (i.e. is within 0.3 m of a wall) or overlaps others.
        """
        obj = self.rom.add_object_by_template_handle(self.template(asset))
        obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        obj.rotation = mn.Quaternion.rotation(mn.Deg(yaw), mn.Vector3(0, 1, 0))
        obj.translation = mn.Vector3(0, 0, 0)
        lo, hi = world_aabb(obj)                        # rotated object at the origin
        shift = np.array([x - (lo[0] + hi[0]) / 2, self.floor_y - lo[1], z - (lo[2] + hi[2]) / 2])
        obj.translation = mn.Vector3(*shift)
        foot = self._footprint(obj)
        ok = all(self.pf.is_navigable(np.array([px, self.nav_y, pz], dtype=np.float32), 0.5)
                 for px, pz in np.vstack([foot, foot.mean(axis=0)]))
        ok = ok and not any(rects_overlap(grow(foot, margin), o["foot"]) for o in others)
        if not ok:
            self.rom.remove_object_by_id(obj.object_id)
            return None
        for node in obj.visual_scene_nodes:
            node.semantic_id = sid
        lo, hi = world_aabb(obj)
        return {"obj": obj, "sid": sid, "foot": foot, "height": float(hi[1] - lo[1]),
                "centre": [float(foot[:, 0].mean()), float(self.floor_y), float(foot[:, 1].mean())],
                "mid": ((lo + hi) / 2).tolist(), "top": float(hi[1])}

    def _footprint(self, obj):
        """Rotated floor rectangle of the object (4 corners, (x, z)), from its local visual box."""
        rot, t = obj.rotation, obj.translation
        obj.rotation, obj.translation = mn.Quaternion(), mn.Vector3(0, 0, 0)
        lo, hi = world_aabb(obj)
        obj.rotation, obj.translation = rot, t
        T = obj.transformation
        pts = [T.transform_point(mn.Vector3(cx, lo[1], cz))
               for cx, cz in ((lo[0], lo[2]), (hi[0], lo[2]), (hi[0], hi[2]), (lo[0], hi[2]))]
        return np.array([[p[0], p[2]] for p in pts])

    def remove(self, o):
        self.rom.remove_object_by_id(o["obj"].object_id)

    def hide(self, o):
        o["_t"] = o["obj"].translation
        o["obj"].translation = mn.Vector3(0, -50, 0)

    def show(self, o):
        o["obj"].translation = o.pop("_t")

    # ----- visibility -----
    def _first_hit(self, origin, direction, max_dist):
        ray = habitat_sim.geo.Ray(mn.Vector3(*origin), mn.Vector3(*direction))
        res = self.sim.cast_ray(ray, max_dist)
        if not res.has_hits():
            return None
        h = res.hits[0]
        return np.array(h.point), h.object_id

    def eye(self, xz):
        return np.array([xz[0], self.floor_y + EYE, xz[1]])

    def line_of_sight(self, xz, o, ours):
        """True when nothing but our objects lies between the eye and points of the object: its middle,
        its top centre and its footprint corners at half height. A ray may pass through an open
        object (under a table top, between leaves) and land behind it; only a wall, door or floor hit
        *before* the point counts as blocked."""
        e = self.eye(xz)
        ids = {p["obj"].object_id for p in ours}
        c = np.array(o["mid"])
        targets = [c, np.array([c[0], o["top"] - 0.05, c[2]])]
        fc = o["foot"].mean(axis=0)
        for px, pz in fc + 0.7 * (o["foot"] - fc):  # footprint corners, pulled in, at half height
            targets.append(np.array([px, self.floor_y + 0.5 * o["height"], pz]))
        for target in targets:
            d = target - e
            n = np.linalg.norm(d)
            hit = self._first_hit(e, d / n, n + 0.5)
            if hit is None or hit[1] in ids:
                continue
            if np.linalg.norm(hit[0] - e) < n - 0.05:  # something else is in the way
                return False
        return True

    # ----- rendering -----
    def render(self, xz, yaw):
        set_pose(self.sim, [xz[0], self.floor_y, xz[1]], yaw)
        obs = self.sim.get_sensor_observations()
        return obs["rgb"][..., :3].copy(), obs["semantic"].astype(np.int32), obs["depth"].astype(np.float32)

    def composite(self, rgb, sem, depth, xz, yaw, keep):
        """Our objects as rendered; everything else grey; visible floor gets the grid lines."""
        h, w = depth.shape
        u = (np.arange(w) + 0.5 - w / 2) / FOCAL
        v = -(np.arange(h) + 0.5 - h / 2) / FOCAL
        U, V = np.meshgrid(u, v)
        f, r = forward(yaw), right(yaw)
        e = self.eye(xz)
        P = e[None, None, :] + depth[..., None] * (U[..., None] * r + V[..., None] * np.array([0, 1, 0]) + f)
        floor = (np.abs(P[..., 1] - self.floor_y) < FLOOR_TOL) & (depth > 0) & (V < 0)
        out = np.empty_like(rgb)
        out[:] = BG_GREY
        cov = np.zeros((h, w))
        for k in (0, 2):
            g = P[..., k] / GRID_STEP
            dist = np.abs(g - np.round(g)) * GRID_STEP                      # metres to nearest line
            pix = np.hypot(*np.gradient(P[..., k]))                         # metres per pixel
            pix = np.clip(np.nan_to_num(pix, nan=0.05), 1e-4, 0.25)
            cov = np.maximum(cov, np.clip((GRID_HALF_WIDTH - dist) / pix + 0.5, 0, 1))
        cov = np.where(floor, cov, 0.0)[..., None]
        out = (out * (1 - cov) + np.array(GRID_LINE) * cov).astype(np.uint8)
        mask = np.isin(sem, list(keep))
        out[mask] = rgb[mask]
        return out


def mask_stats(sem, sid):
    """Visible-pixel statistics of one object in one view."""
    m = sem == sid
    n = int(m.sum())
    if n == 0:
        return {"vis_pct": 0.0}
    ys, xs = np.nonzero(m)
    base_y = int(ys.max())
    return {"vis_pct": round(100.0 * n / m.size, 3), "cx": round(float(xs.mean()), 1),
            "cy": round(float(ys.mean()), 1), "xmin": int(xs.min()), "xmax": int(xs.max()),
            "ymin": int(ys.min()), "ymax": base_y,
            "base_x": round(float(xs[ys >= base_y - 2].mean()), 1)}
