"""Replica-dev RGB-D streams in Habitat (task 006 step 1), inside hpc/habitat.def on a PERUN GPU node.

RGB source (configs/replica_dev.yaml rgb_source): `ptex` = Replica's real textures, `<scene>/mesh.ply` + `textures/` +
`habitat/sorted_faces.bin`, which needs habitat-sim 0.2.4 (hpc/habitat_ptex.def; PTex was removed in 0.2.5); or
`mesh_semantic_vertex_colours_flat` = `mesh_semantic.ply`'s geometry with its own vertex colours (= the NICE
`<scene>_mesh.ply` colours) rendered flat, any version (habitat.sif, 0.3.3).

Frames. The render asset is written already rotated into Habitat's Y-up frame, so Habitat loads it unrotated:
    p_hab = R_M2H @ p_mesh,  R_M2H: (x, y, z) -> (x, z, -y)        (mesh / GT frame: up = +z)
Stored poses are c2w in the mesh (GT) frame with OpenCV camera axes: c2w_mesh = R_H2M @ c2w_hab_gl @ GL_TO_CV.
The transform is proved by spatialmem.render.proof_dev (raycast depth from mesh_semantic.ply at the stored poses vs
the Habitat depth), with a wrong-axes negative control.

  python -m spatialmem.render.replica_dev plan --plan-dir /runs/replica_dev_plan            (habitat.sif, 0.3.3)
  python -m spatialmem.render.replica_dev scene --scene apartment_0 --out /runs/replica_dev [--max-frames 100]
  python -m spatialmem.render.replica_dev scenes --out /runs/replica_dev --part 0/2           (habitat_ptex.sif for PTex)
Output (NICE layout): <out>/<scene>/results/{frame%06d.jpg, depth%06d.png}, traj.txt, cam_params.json, meta.json, DONE
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import yaml

CFG = Path("/workspace/configs/replica_dev.yaml")
DATA = Path("/data/replica_v1")
R_M2H = np.array([[1.0, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
R_H2M = R_M2H.T


def write_render_asset(scene: str, out: Path) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray]:
    """mesh_semantic.ply -> triangle PLY in the Habitat frame with vertex colours. Returns (path, verts_hab, tri
    centroids_hab, tri class names)."""
    from plyfile import PlyData, PlyElement
    ply = PlyData.read(str(DATA / scene / "habitat" / "mesh_semantic.ply"))
    v, f = ply["vertex"], ply["face"]
    xyz = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float64)
    hab = xyz @ R_M2H[:3, :3].T
    quads = np.vstack(f["vertex_indices"]).astype(np.int64)
    tris = np.concatenate([quads[:, [0, k, k + 1]] for k in range(1, quads.shape[1] - 1)])
    fobj = np.tile(np.asarray(f["object_id"], np.int64), quads.shape[1] - 2)
    info = json.loads((DATA / scene / "habitat" / "info_semantic.json").read_text())
    cls = {int(c["id"]): c["name"] for c in info["classes"]}
    oc = {int(o["id"]): cls.get(int(o["class_id"]), "undefined") for o in info["objects"]}
    tri_cls = np.array([oc.get(int(o), "undefined") for o in fobj])
    vert = np.empty(len(hab), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vert["x"], vert["y"], vert["z"] = hab[:, 0], hab[:, 1], hab[:, 2]
    vert["red"], vert["green"], vert["blue"] = v["red"], v["green"], v["blue"]
    face = np.empty(len(tris), dtype=[("vertex_indices", "i4", (3,))])
    face["vertex_indices"] = tris
    path = out / "mesh_vc.ply"
    out.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vert, "vertex"), PlyElement.describe(face, "face")], text=False).write(str(path))
    return path, hab, hab[tris].mean(1), tri_cls


def make_sim(asset: Path, cam: dict, pitch_deg: float, gpu: int = 0):
    import habitat_sim
    sc = habitat_sim.SimulatorConfiguration()
    sc.scene_id = str(asset)
    sc.enable_physics = False
    sc.gpu_device_id = gpu
    sc.override_scene_light_defaults = True
    sc.scene_light_setup = habitat_sim.gfx.NO_LIGHT_KEY      # flat: pixel colour = interpolated vertex colour
    hfov = math.degrees(2 * math.atan(cam["w"] / (2 * cam["fx"])))
    sensors = []
    for uuid, st in (("rgb", habitat_sim.SensorType.COLOR), ("depth", habitat_sim.SensorType.DEPTH)):
        s = habitat_sim.CameraSensorSpec()
        s.uuid, s.sensor_type, s.resolution, s.hfov = uuid, st, [cam["h"], cam["w"]], hfov
        s.position, s.orientation = [0.0, 0.0, 0.0], [math.radians(pitch_deg), 0.0, 0.0]
        sensors.append(s)
    ag = habitat_sim.agent.AgentConfiguration(sensor_specifications=sensors)
    return habitat_sim.Simulator(habitat_sim.Configuration(sc, [ag])), hfov


def keyframes(pf, islands_pts: list[np.ndarray], tours: list[list[int]], spin: float) -> list[tuple[np.ndarray, float, bool]]:
    """(floor point, unwrapped yaw, jump-before) keys: a `spin` rad look-around at every waypoint, then turn-then-walk
    along the geodesic path to the next one (heading = segment direction, forward = -Z, yaw about +Y)."""
    from spatialmem.hssd.traj import geodesic
    keys: list[tuple[np.ndarray, float, bool]] = []
    yaw = 0.0

    def turn_to(h: float) -> float:
        return yaw + (h - yaw + math.pi) % (2 * math.pi) - math.pi
    for pts, order in zip(islands_pts, tours):
        for k, i in enumerate(order):
            keys.append((pts[i], yaw, k == 0 and bool(keys)))
            if spin > 0:
                yaw += spin
                keys.append((pts[i], yaw, False))
            if k + 1 == len(order):
                continue
            _, path = geodesic(pf, pts[i], pts[order[k + 1]])
            for a, b in zip(path[:-1], path[1:]):
                if math.hypot(b[0] - a[0], b[2] - a[2]) < 1e-6:
                    continue
                yaw = turn_to(math.atan2(-(b[0] - a[0]), -(b[2] - a[2])))
                keys.append((a, yaw, False))
                keys.append((b, yaw, False))
    return keys


def spin_angle(pf, islands_pts: list[np.ndarray], tours: list[list[int]], n: int, max_rot: float, share: float) -> float:
    """Look-around per waypoint: 360 deg if the scene's rotation fits in `share` of the n frames at `max_rot` per frame,
    else the angle that makes it fit (path turns are counted first)."""
    keys = keyframes(pf, islands_pts, tours, 0.0)
    turns = float(np.abs(np.diff([k[1] for k in keys])).sum()) if len(keys) > 1 else 0.0
    n_wp = sum(len(t) for t in tours)
    return float(np.clip((share * (n - 1) * max_rot - turns) / max(n_wp, 1), 0.0, 2 * math.pi))


def sample(keys: list[tuple[np.ndarray, float, bool]], n: int, max_rot: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """n frames along the keys: segment i gets max(d_i / s, |dyaw_i| / max_rot) frame intervals (island jumps 0), with
    the translation step s solved so the total is n - 1. Hence every frame turns by <= max_rot."""
    P = np.array([k[0] for k in keys], float)
    Y = np.array([k[1] for k in keys], float)
    J = np.array([k[2] for k in keys])
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    r = np.abs(np.diff(Y)) / max_rot
    d[J[1:]], r[J[1:]] = 0.0, 0.0
    if r.sum() >= n - 1:
        raise ValueError(f"rotation alone needs {r.sum():.0f} frames > {n - 1} at {math.degrees(max_rot):.1f} deg/frame")
    lo, hi = 1e-6, max(d.sum(), 1e-6) * 1e3
    for _ in range(200):                       # bisection on the step: total intervals decrease with s
        s = (lo * hi) ** 0.5
        lo, hi = (s, hi) if np.maximum(d / s, r).sum() > n - 1 else (lo, s)
    seg = np.maximum(d / hi, r)
    tau = np.r_[0.0, np.cumsum(seg)]
    t = np.linspace(0, tau[-1], n)
    j = np.clip(np.searchsorted(tau, t, side="right") - 1, 0, len(tau) - 2)
    w = np.where(tau[j + 1] > tau[j], (t - tau[j]) / np.maximum(tau[j + 1] - tau[j], 1e-12), 0.0)
    pos = P[j] + (P[j + 1] - P[j]) * w[:, None]
    yaw = Y[j] + (Y[j + 1] - Y[j]) * w
    return pos, yaw, {"step_m": float(hi), "path_m": float(d.sum()), "rot_rad": float(np.abs(np.diff(Y))[~J[1:]].sum()),
                      "rot_frames": float(r.sum()), "n_jumps": int(J.sum())}


def recompute_navmesh(sim, radius: float, height: float) -> None:
    """Works on habitat-sim 0.2.4 (PTex image) and 0.3.3 (no include_static_objects field / argument needed)."""
    import habitat_sim
    ns = habitat_sim.NavMeshSettings()
    ns.set_defaults()
    ns.agent_radius, ns.agent_height = radius, height
    assert sim.recompute_navmesh(sim.pathfinder, ns), "navmesh recompute failed"


def plan_scene(scene: str, plan_dir: Path) -> dict:
    """Camera path in the Habitat frame (floor points + yaw per frame) from a navmesh on the mesh_semantic geometry.
    Runs on habitat-sim 0.3.3 (habitat.sif): 0.2.4 cannot build these navmeshes ("Could not build Detour navmesh",
    PERUN 137135). Writes <plan_dir>/<scene>.json; the PTex render (0.2.4) only replays it."""
    from scipy.spatial import cKDTree
    from spatialmem.hssd.traj import MIN_ISLAND_M2, tour, waypoints

    c = yaml.safe_load(CFG.read_text())
    cam, tr = c["camera"], c["trajectory"]
    f = plan_dir / f"{scene}.json"
    if f.exists():
        return json.loads(f.read_text())
    asset, _, cen, tri_cls = write_render_asset(scene, plan_dir / "_assets" / scene)
    sim, _ = make_sim(asset, cam, tr["pitch_deg"])
    try:
        radius = tr["agent_radius_m"]
        recompute_navmesh(sim, radius, tr["camera_height_m"])
    except AssertionError:   # "where the navmesh allows": hotel_0 has no navmesh at 0.5 m (PERUN 137141)
        radius = tr["agent_radius_fallback_m"]
        recompute_navmesh(sim, radius, tr["camera_height_m"])
    pf = sim.pathfinder
    pf.seed(tr["seed"])
    tree = cKDTree(cen)
    floor_ok = np.isin(tri_cls, tr["floor_classes"])

    def keep(p: np.ndarray) -> bool:   # floor test only: Replica FRL ceilings are partial, the 002 indoor test fails there
        return bool(floor_ok[tree.query(p)[1]])
    islands = [i for i in range(pf.num_islands) if pf.island_area(i) >= MIN_ISLAND_M2]
    pts_l, tours, per = [], [], {}
    for isl in islands:
        pts = waypoints(pf, tr["grid_m"], isl, keep)
        order = tour(pf, pts)
        per[int(isl)] = {"area_m2": float(pf.island_area(isl)), "waypoints": int(len(pts)), "toured": len(order)}
        if order:
            pts_l.append(pts)
            tours.append(order)
    max_rot = math.radians(tr["max_rot_deg_per_frame"])
    spin = spin_angle(pf, pts_l, tours, c["frames"], max_rot, tr["rot_frame_share"])
    pos, yaw, stats = sample(keyframes(pf, pts_l, tours, spin), c["frames"], max_rot)
    stats["spin_deg"], stats["agent_radius_m"] = math.degrees(spin), radius
    plan = {"scene": scene, "pos": pos.tolist(), "yaw": yaw.tolist(), "path": stats, "islands": per,
            "n_islands": int(pf.num_islands), "config": c}
    sim.close()
    plan_dir.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(plan))
    print(f"{scene}: planned {len(pos)} frames, {sum(v['waypoints'] for v in per.values())} waypoints, spin "
          f"{stats['spin_deg']:.0f} deg, radius {radius} m, step {stats['step_m'] * 100:.1f} cm, path {stats['path_m']:.1f} m, jumps {stats['n_jumps']}", flush=True)
    return plan


def run_scene(scene: str, out_root: Path, max_frames: int | None, plan_dir: Path) -> dict:
    from PIL import Image
    from spatialmem.hssd.sim import set_pose

    c = yaml.safe_load(CFG.read_text())
    cam, tr = c["camera"], c["trajectory"]
    out = out_root / scene
    if (out / "DONE").exists():
        print(f"{scene}: done already", flush=True)
        return json.loads((out / "meta.json").read_text())
    plan = plan_scene(scene, plan_dir)
    assert plan["config"] == c, "plan made under a different configs/replica_dev.yaml"
    (out / "results").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if c["rgb_source"] == "ptex":   # habitat-sim 0.2.4: <scene>/mesh.ply -> FRL_PTEX_MESH, up (0,0,1) front (0,1,0) = R_M2H
        asset = DATA / scene / "mesh.ply"   # (needs textures/ + habitat/sorted_faces.bin)
    else:
        asset = write_render_asset(scene, plan_dir / "_assets" / scene)[0]
    sim, hfov = make_sim(asset, cam, tr["pitch_deg"])
    t_load = time.time() - t0
    pos, yaw, stats, per = np.array(plan["pos"]), np.array(plan["yaw"]), plan["path"], plan["islands"]
    n = c["frames"] if not max_frames else min(max_frames, c["frames"])
    pos, yaw = pos[:n], yaw[:n]
    c2ws, ms_render, ms_total, dvalid = [], [], [], []
    for i in range(n):
        ta = time.time()
        c2w_hab = set_pose(sim, pos[i] + [0.0, tr["camera_height_m"], 0.0], float(yaw[i]))
        tb = time.time()
        obs = sim.get_sensor_observations()
        rgb, depth = obs["rgb"][..., :3], np.asarray(obs["depth"], np.float32)
        tc = time.time()
        depth = np.where(np.isfinite(depth) & (depth <= cam["depth_max_m"]), depth, 0.0)
        dvalid.append(float((depth > 0).mean()))
        Image.fromarray(rgb).save(out / "results" / f"frame{i:06d}.jpg", quality=100)
        Image.fromarray(np.round(depth * cam["scale"]).astype(np.uint16)).save(out / "results" / f"depth{i:06d}.png")
        c2ws.append(R_H2M @ c2w_hab)
        ms_render.append((tc - tb) * 1000)
        ms_total.append((time.time() - ta) * 1000)
    C = np.array(c2ws)
    np.savetxt(out / "traj.txt", C.reshape(len(C), 16))
    (out / "cam_params.json").write_text(json.dumps({"camera": {k: cam[k] for k in ("w", "h", "fx", "fy", "cx", "cy", "scale")}}, indent=4))
    fwd = C[:, :3, 2]
    step = np.linalg.norm(np.diff(C[:, :3, 3], axis=0), axis=1)
    ang = np.degrees(np.arccos(np.clip((fwd[1:] * fwd[:-1]).sum(1), -1, 1)))
    ang = ang[step < 1.0]          # island jumps (camera teleports between navmesh islands) are cuts, not motion
    meta = {"scene": scene, "n_frames": n, "hfov_deg": hfov, "rgb_source": c["rgb_source"], "habitat_sim": __import__("habitat_sim").__version__,
            "pose": "c2w, OpenCV axes, mesh_semantic.ply frame (up +z), row-major 4x4 per line", "R_M2H": R_M2H.tolist(),
            "islands": per, "n_islands": plan["n_islands"], "path": stats, "load_s": t_load,
            "ms_render_median": float(np.median(ms_render)), "ms_frame_median": float(np.median(ms_total)),
            "depth_valid_min": float(min(dvalid)), "depth_valid_median": float(np.median(dvalid)),
            "step_cm_median": float(np.median(step) * 100), "step_cm_p95": float(np.percentile(step, 95) * 100),
            "rot_deg_median": float(np.median(ang)), "rot_deg_p95": float(np.percentile(ang, 95)), "rot_deg_max": float(ang.max()),
            "cam_height_mesh_z_median": float(np.median(C[:, 2, 3])), "config": c}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    sim.close()
    (out / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    print(f"{scene}: frames={n} islands={plan['n_islands']} waypoints={sum(v['waypoints'] for v in per.values())} "
          f"path {stats['path_m']:.1f} m rot {math.degrees(stats['rot_rad']):.0f} deg jumps {stats['n_jumps']} | step med "
          f"{meta['step_cm_median']:.1f} cm rot med {meta['rot_deg_median']:.2f} max {meta['rot_deg_max']:.2f} deg spin {stats['spin_deg']:.0f} | depth valid min {meta['depth_valid_min']:.3f} "
          f"| render {meta['ms_render_median']:.1f} ms frame {meta['ms_frame_median']:.1f} ms load {t_load:.0f} s", flush=True)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "scene", "scenes"])
    ap.add_argument("--scene")
    ap.add_argument("--out", type=Path, default=Path("/runs/replica_dev"))
    ap.add_argument("--plan-dir", type=Path, default=Path("/runs/replica_dev_plan"))
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--part", default="0/1", help="i/n: every n-th scene starting at i")
    a = ap.parse_args()
    i, n = (int(x) for x in a.part.split("/"))
    scenes = [a.scene] if a.scene else yaml.safe_load(CFG.read_text())["scenes"][i::n]
    for s in scenes:
        if a.cmd == "plan":
            plan_scene(s, a.plan_dir)
        else:
            run_scene(s, a.out, a.max_frames, a.plan_dir)


if __name__ == "__main__":
    main()
