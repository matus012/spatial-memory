"""Deterministic coverage path on the navmesh (task 002 step 6).

Islands = every navmesh island of >= MIN_ISLAND_M2. Waypoints = a `grid_m` grid over each island (grid points
snapped to it, deduplicated), kept only if INDOOR: an upward 8x8 depth sensor at camera height hits geometry
within INDOOR_CEILING_M on >= half its pixels (drops yards and roofs). Tour = per island (island index order),
greedy nearest-neighbour by geodesic distance from the waypoint nearest the island's waypoint centroid; the
camera jumps between islands. Frames = a `spin_yaws`-view spin at every waypoint + one frame every `path_step_m`
along the geodesic path to the next waypoint, facing along the path. No randomness.

G6 fix (task 002, the one diagnosed fix): the first version toured only the largest island, which in the low-coverage
scenes was a yard, a roof or one of several rooms / floors (report 002, G6 diagnosis).
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

MIN_ISLAND_M2 = 1.0
INDOOR_CEILING_M = 6.0


def recompute_navmesh(sim, radius: float, height: float) -> None:
    import habitat_sim
    ns = habitat_sim.NavMeshSettings()
    ns.set_defaults()
    ns.agent_radius, ns.agent_height = radius, height
    ns.include_static_objects = True
    assert sim.recompute_navmesh(sim.pathfinder, ns), "navmesh recompute failed"


def indoor_test(sim, cam_height: float) -> Callable[[np.ndarray], bool]:
    from spatialmem.hssd.sim import set_pose

    def f(p: np.ndarray) -> bool:
        set_pose(sim, np.asarray(p) + [0.0, cam_height, 0.0], 0.0)
        d = np.asarray(sim.get_sensor_observations()["up"], float)
        ok = np.isfinite(d) & (d > 0) & (d <= INDOOR_CEILING_M)
        return bool(ok.mean() >= 0.5)
    return f


def waypoints(pf, grid: float, isl: int, keep: Callable[[np.ndarray], bool]) -> np.ndarray:
    lo, hi = (np.asarray(b, float) for b in pf.get_bounds())
    ref = np.asarray(pf.snap_point(pf.get_random_navigable_point(island_index=isl), island_index=isl), float)
    pts: list[np.ndarray] = []
    for x in np.arange(lo[0] + grid / 2, hi[0], grid):
        for z in np.arange(lo[2] + grid / 2, hi[2], grid):
            q = np.asarray(pf.snap_point([x, ref[1], z], island_index=isl), float)
            if not np.all(np.isfinite(q)) or math.hypot(q[0] - x, q[2] - z) > grid / 2:
                continue
            if all(np.linalg.norm(q - p) > grid / 2 for p in pts) and keep(q):
                pts.append(q)
    return np.array(pts).reshape(-1, 3)


def geodesic(pf, a, b) -> tuple[float, np.ndarray]:
    import habitat_sim
    sp = habitat_sim.ShortestPath()
    sp.requested_start, sp.requested_end = a, b
    if not pf.find_path(sp):
        return math.inf, np.zeros((0, 3))
    return float(sp.geodesic_distance), np.asarray(sp.points, float)


def tour(pf, pts: np.ndarray) -> list[int]:
    left = list(range(len(pts)))
    if not left:
        return []
    cur = min(left, key=lambda i: (np.linalg.norm(pts[i] - pts.mean(0)), i))
    order = [cur]
    left.remove(cur)
    while left:
        d = [(geodesic(pf, pts[cur], pts[j])[0], j) for j in left]
        dist, nxt = min(d)
        if not math.isfinite(dist):
            break  # unreachable remainder (should not happen on one island)
        order.append(nxt)
        left.remove(nxt)
        cur = nxt
    return order


def resample(path: np.ndarray, step: float) -> list[tuple[np.ndarray, float]]:
    """Points every `step` m along a polyline with the heading (yaw about +Y, forward = -Z) of their segment."""
    out = []
    for a, b in zip(path[:-1], path[1:]):
        seg = b - a
        L = float(np.hypot(seg[0], seg[2]))
        if L < 1e-6:
            continue
        yaw = math.atan2(-seg[0], -seg[2])
        for t in np.arange(step, L, step):
            out.append((a + seg * (t / L), yaw))
    return out


def coverage_poses(pf, cfg: dict, keep: Callable[[np.ndarray], bool]) -> tuple[list[tuple[np.ndarray, float]], dict]:
    """(floor point, yaw) per frame + path stats."""
    poses: list[tuple[np.ndarray, float]] = []
    islands = [i for i in range(pf.num_islands) if pf.island_area(i) >= MIN_ISLAND_M2]
    per = {}
    for isl in islands:
        pts = waypoints(pf, cfg["grid_m"], isl, keep)
        order = tour(pf, pts)
        for k, i in enumerate(order):
            for s in range(cfg["spin_yaws"]):
                poses.append((pts[i], 2 * math.pi * s / cfg["spin_yaws"]))
            if k + 1 < len(order):
                _, path = geodesic(pf, pts[i], pts[order[k + 1]])
                poses += resample(path, cfg["path_step_m"])
        per[isl] = {"area_m2": float(pf.island_area(isl)), "indoor_waypoints": int(len(pts)), "toured": len(order)}
    return poses, {"n_islands": int(pf.num_islands), "islands_used": per,
                   "n_waypoints": sum(v["indoor_waypoints"] for v in per.values()), "n_frames": len(poses)}
