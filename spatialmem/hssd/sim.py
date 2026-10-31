"""HSSD in Habitat-sim (runs inside hpc/habitat.def): load a scene, give every object instance its own semantic ID,
export GT objects (tight world AABBs from the GLB vertices), render RGB / depth / instance at given poses.

Frames: Habitat world is Y-up. Poses are stored camera-to-world with OpenCV camera axes (x right, y down, z forward)
so `spatialmem.hssd.check.backproject` (same convention as the Replica raycaster) applies unchanged.
Instance IDs: 0 = stage (walls / floor / ceiling, one mesh) and empty pixels; GT objects 1..K in sorted-handle order,
where the decomposed parts of one parent template that touch form one GT object (`group_parts`).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

DATA = Path("/data/hssd-hab")
DATASET_CFG = DATA / "hssd-hab.scene_dataset_config.json"
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])
STAGE_ID = 0


@dataclass(frozen=True)
class Camera:
    W: int
    H: int
    hfov_deg: float
    height_m: float
    pitch_deg: float
    depth_scale: float
    depth_max_m: float

    @property
    def fx(self) -> float:
        return (self.W / 2) / math.tan(math.radians(self.hfov_deg) / 2)

    def intrinsics(self) -> dict:
        return {"W": self.W, "H": self.H, "fx": self.fx, "fy": self.fx, "cx": (self.W - 1) / 2, "cy": (self.H - 1) / 2}


def load_camera(path: Path = Path("/workspace/configs/camera.yaml")) -> tuple[Camera, dict]:
    c = yaml.safe_load(path.read_text())
    cam = Camera(c["width"], c["height"], c["hfov_deg"], c["camera_height_m"], c["pitch_deg"], c["depth_scale"],
                 c["depth_max_m"])
    return cam, c


def mat4(m) -> np.ndarray:
    """Magnum Matrix4 -> numpy (row-major math convention); m[c] is column c in Magnum."""
    a = np.array([[m[c][r] for c in range(4)] for r in range(4)], dtype=np.float64)
    assert np.allclose(a[3], [0, 0, 0, 1]), a
    return a


def make_sim(scene: str, cam: Camera, gpu: int = 0):
    import habitat_sim
    sc = habitat_sim.SimulatorConfiguration()
    sc.scene_dataset_config_file = str(DATASET_CFG)
    sc.scene_id = scene
    sc.enable_physics = False
    sc.gpu_device_id = gpu
    sensors = []
    for uuid, st in (("rgb", habitat_sim.SensorType.COLOR), ("depth", habitat_sim.SensorType.DEPTH),
                     ("inst", habitat_sim.SensorType.SEMANTIC)):
        s = habitat_sim.CameraSensorSpec()
        s.uuid, s.sensor_type, s.resolution = uuid, st, [cam.H, cam.W]
        s.hfov = cam.hfov_deg
        s.position = [0.0, 0.0, 0.0]
        s.orientation = [math.radians(cam.pitch_deg), 0.0, 0.0]
        sensors.append(s)
    up = habitat_sim.CameraSensorSpec()   # indoor test for coverage waypoints (traj.is_indoor), never stored
    up.uuid, up.sensor_type, up.resolution, up.hfov = "up", habitat_sim.SensorType.DEPTH, [8, 8], 60.0
    up.position, up.orientation = [0.0, 0.0, 0.0], [math.pi / 2, 0.0, 0.0]
    sensors.append(up)
    ag = habitat_sim.agent.AgentConfiguration(sensor_specifications=sensors)
    return habitat_sim.Simulator(habitat_sim.Configuration(sc, [ag]))


# ---------------------------------------------------------------- objects / GT
PART_TOUCH_M = 0.02   # decomposed parts of one parent template whose AABBs are within this gap = one GT object


@lru_cache(maxsize=1)
def template_index() -> dict[str, Path]:
    """template name -> its object_config.json (objects/<c>/, objects/decomposed/<hash>/, objects/openings/, ...)."""
    return {p.name.split(".")[0]: p for p in (DATA / "objects").rglob("*.object_config.json")}


def template_config(template: str) -> tuple[dict, Path]:
    p = template_index()[template]
    return json.loads(p.read_text()), p.parent


def glb_vertices(path: Path) -> np.ndarray:
    import trimesh
    m = trimesh.load(path, force="scene")
    return np.asarray(m.dump(concatenate=True).vertices, dtype=np.float64)


def categories() -> tuple[dict[str, tuple[str, str]], dict[int, str]]:
    """objects.csv (main_category, wnsynsetkey) per template hash; lexicon id -> name."""
    import csv
    main = {r["id"]: (r["main_category"], r["wnsynsetkey"])
            for r in csv.DictReader(open(DATA / "semantics" / "objects.csv", encoding="utf-8"))}
    lex = {int(c["id"]): c["name"] for c in json.loads((DATA / "semantics" / "hssd-hab_semantic_lexicon.json").read_text())["classes"]}
    return main, lex


def parent_of(template: str) -> str | None:
    return template.split("_part_")[0] if "_part_" in template else None


def instances(sim) -> list[dict]:
    """One record per rigid object instance, sorted by handle, with its tight world AABB = AABB of the GLB vertices
    under the absolute transform of the object's deepest visual scene node. (Habitat re-centres the root node on
    the bounding-box centre and puts the instance's non_uniform_scale on the child visual node, so the root
    transform is NOT the asset frame — task 002 diagnosis.) Checks, in mm:
    `bb_check_mm` = raw GLB AABB vs the union of the visual nodes' own cumulative_bbs; `root_check_mm` = how far the world AABB
    leaves the root's world box (8 transformed corners of the root cumulative_bb); `n_node_transforms` > 1 flags
    visual nodes with differing transforms (multi-node GLB)."""
    main, lex = categories()
    rom = sim.get_rigid_object_manager()
    cache: dict[str, tuple[np.ndarray, dict]] = {}
    out = []
    for h in sorted(rom.get_object_handles()):
        o = rom.get_object_by_handle(h)
        t = h.split("_:")[0].split("/")[-1]
        if t not in cache:
            cfg, d = template_config(t)
            cache[t] = glb_vertices(d / cfg["render_asset"]), cfg
        v_raw, cfg = cache[t]
        nodes = list(o.visual_scene_nodes)
        Ns = [mat4(n.absolute_transformation()) for n in nodes[1:]] or [mat4(nodes[0].absolute_transformation())]
        N = Ns[-1]
        v_world = v_raw @ N[:3, :3].T + N[:3, 3]
        lo, hi = v_world.min(0), v_world.max(0)
        sub = nodes[1:] or nodes[:1]   # one sub-mesh per node, all under the same transform (n_node_transforms == 1)
        blo = np.min([np.array(n.cumulative_bb.min) for n in sub], 0)
        bhi = np.max([np.array(n.cumulative_bb.max) for n in sub], 0)
        check = float(max(np.abs(v_raw.min(0) - blo).max(), np.abs(v_raw.max(0) - bhi).max()) * 1000)
        rb = o.root_scene_node.cumulative_bb
        corners = np.array([[x, y, z] for x in (rb.min[0], rb.max[0]) for y in (rb.min[1], rb.max[1]) for z in (rb.min[2], rb.max[2])])
        M = mat4(o.transformation)
        cw = corners @ M[:3, :3].T + M[:3, 3]
        root_check = float(max(np.clip(cw.min(0) - lo, 0, None).max(), np.clip(hi - cw.max(0), 0, None).max()) * 1000)
        par = parent_of(t)
        mc, syn = main.get(par or t, ("", ""))
        out.append({"handle": h, "template": t, "parent": par, "template_semantic_id": int(o.semantic_id),
                    "category_main": mc, "category_lexicon": lex.get(int(cfg.get("semantic_id", -1)), ""), "wnsynset": syn,
                    "aabb_min": lo.tolist(), "aabb_max": hi.tolist(), "translation": N[:3, 3].tolist(),
                    "n_vertices": int(len(v_raw)), "bb_check_mm": check, "root_check_mm": root_check,
                    "n_node_transforms": len({tuple(np.round(x, 6).ravel()) for x in Ns})})
    return out


def group_parts(recs: list[dict]) -> list[list[int]]:
    """Index groups: a non-part instance alone; parts of one parent whose AABBs touch (gap <= PART_TOUCH_M) together
    (connected components). Groups ordered by their first member (= sorted handle order)."""
    parent = list(range(len(recs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    by_par: dict[str, list[int]] = {}
    for i, r in enumerate(recs):
        if r["parent"]:
            by_par.setdefault(r["parent"], []).append(i)
    for idx in by_par.values():
        lo = np.array([recs[i]["aabb_min"] for i in idx])
        hi = np.array([recs[i]["aabb_max"] for i in idx])
        gap = np.maximum(lo[:, None] - hi[None], lo[None] - hi[:, None]).max(-1)   # > 0 = separated along some axis
        for a, b in zip(*np.nonzero(np.triu(gap <= PART_TOUCH_M, 1))):
            parent[find(idx[a])] = find(idx[b])
    groups: dict[int, list[int]] = {}
    for i in range(len(recs)):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def assign_ids(sim, recs: list[dict]) -> list[dict]:
    """GT objects 1..K (one per group); every member instance renders with its group's ID. Returns GT records."""
    rom = sim.get_rigid_object_manager()
    gts = []
    for gid, g in enumerate(group_parts(recs), start=1):
        for i in g:
            recs[i]["gt_id"] = gid
            rom.get_object_by_handle(recs[i]["handle"]).semantic_id = gid
        m = [recs[i] for i in g]
        gts.append({"id": gid, "members": [r["handle"] for r in m], "parent": m[0]["parent"],
                    "aabb_min": np.min([r["aabb_min"] for r in m], 0).tolist(),
                    "aabb_max": np.max([r["aabb_max"] for r in m], 0).tolist()})
    return gts


# ---------------------------------------------------------------- rendering
def set_pose(sim, position, yaw: float) -> np.ndarray:
    """Agent at `position` (camera height already added) facing yaw (rad about +Y); returns c2w (OpenCV axes)."""
    import habitat_sim
    import magnum as mn
    ag = sim.get_agent(0)
    st = habitat_sim.AgentState()
    st.position = np.asarray(position, dtype=np.float32)
    q = mn.Quaternion.rotation(mn.Rad(yaw), mn.Vector3(0, 1, 0))
    st.rotation = np.quaternion(q.scalar, *q.vector)  # numpy-quaternion (w, x, y, z)
    ag.set_state(st, reset_sensors=True)
    s = ag.get_state().sensor_states["rgb"]
    q = s.rotation
    Rw = quat_to_mat(q.w, q.x, q.y, q.z)
    c2w = np.eye(4)
    c2w[:3, :3], c2w[:3, 3] = Rw, np.asarray(s.position, float)
    return c2w @ GL_TO_CV


def quat_to_mat(w: float, x: float, y: float, z: float) -> np.ndarray:
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def render(sim) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs = sim.get_sensor_observations()
    return obs["rgb"][..., :3], obs["depth"].astype(np.float32), obs["inst"].astype(np.int64)
