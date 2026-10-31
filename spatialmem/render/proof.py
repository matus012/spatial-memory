"""Transform / renderer proof on Replica (task 001 gates G5, G6).

G5  render depth from habitat/mesh_semantic.ply at the NICE-SLAM poses; compare with the dataset depth
    (png / 6553.5) on valid pixels (both > 0): median |diff| <= 1.0 cm AND >= 95 % within 2.0 cm.
G6  back-project the *rendered* instance IDs with the *dataset* depth + pose (every 2nd pixel in u and v,
    all frames — uniform, no per-object cap); per object (>= 500 points) one-sided Chamfer = mean distance
    points -> that object's mesh vertices (the gated number, conservative); points -> that object's mesh
    surface reported alongside. Median over objects <= 2.0 cm.

Run: uv run python -m spatialmem.render.proof [--scenes room0 office0] [--step 20] [--out file.json]
Replica-dev (task 006): same gates on data/replica_dev/<scene> (our Habitat stream); negative controls
--pose-fix opengl (wrong camera axes) and --pose-fix yup (Habitat world frame taken as the mesh frame).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import cKDTree

from spatialmem.gt.replica import DATA, DEV_SCENES, object_vertex_sets, read_semantic_mesh
from spatialmem.render.raycast import Intrinsics, MeshRaycaster, backproject, triangulate
from spatialmem.render.replica_dev import R_M2H

# nice-slam configs/Replica/replica.yaml
REPLICA_INTR = Intrinsics(W=1200, H=680, fx=600.0, fy=600.0, cx=599.5, cy=339.5)
DEPTH_SCALE = 6553.5
BIN = 5e-5                     # 0.05 mm histogram bins for the pooled median
G6_STRIDE = 2                  # pixel subsampling for G6 back-projection


def scene_dir(scene: str) -> Path:
    """NICE stream; Replica-dev scenes (task 006): our Habitat stream in the same layout."""
    return DATA / "replica_dev" / scene if scene in DEV_SCENES else DATA / "nice" / "Replica" / scene


def intrinsics(scene: str) -> tuple[Intrinsics, float]:
    """cam_params.json if shipped, asserted equal to the yaml constants."""
    p = scene_dir(scene) / "cam_params.json" if scene in DEV_SCENES else DATA / "nice" / "Replica" / "cam_params.json"
    if p.exists():
        c = json.loads(p.read_text())["camera"]
        got = Intrinsics(int(c["w"]), int(c["h"]), float(c["fx"]), float(c["fy"]), float(c["cx"]), float(c["cy"]))
        assert got == REPLICA_INTR, f"cam_params.json {got} != yaml {REPLICA_INTR}"
        assert float(c["scale"]) == DEPTH_SCALE, c["scale"]
    return REPLICA_INTR, DEPTH_SCALE


def poses(scene: str) -> np.ndarray:
    return np.loadtxt(scene_dir(scene) / "traj.txt").reshape(-1, 4, 4)


def dataset_depth(scene: str, i: int, scale: float = DEPTH_SCALE) -> np.ndarray:
    return np.asarray(Image.open(scene_dir(scene) / "results" / f"depth{i:06d}.png"), dtype=np.float64) / scale


def dataset_rgb(scene: str, i: int) -> np.ndarray:
    return np.asarray(Image.open(scene_dir(scene) / "results" / f"frame{i:06d}.jpg"))


def semantic_raycaster(scene: str) -> tuple[MeshRaycaster, np.ndarray, dict[int, np.ndarray]]:
    verts, faces, face_obj = read_semantic_mesh(scene)
    tris, src = triangulate(faces)
    return MeshRaycaster(verts, tris, face_obj[src]), verts, object_vertex_sets(faces, face_obj)


def prove_scene(scene: str, step: int = 20, n_frames: int | None = None, pose_fix: str = "none") -> dict:
    intr, scale = intrinsics(scene)
    T = poses(scene)
    frames = list(range(0, len(T), step))[:n_frames]
    rc, verts, vsets = semantic_raycaster(scene)
    hist = np.zeros(int(1.0 / BIN) + 1, dtype=np.int64)   # |diff| up to 1 m, last bin = overflow
    n_valid = n_within2 = n_ds = 0
    per_frame = []
    pts: dict[int, list[np.ndarray]] = {}
    for i in frames:
        c2w = T[i].copy()
        if pose_fix == "opengl":      # diagnostic only: treat traj as OpenGL camera axes
            c2w[:3, 1:3] *= -1
        elif pose_fix == "yup":       # negative control (dev): skip the Habitat Y-up -> mesh Z-up conversion
            c2w = R_M2H @ c2w
        d_ren, inst = rc.render(intr, c2w)
        d_ds = dataset_depth(scene, i, scale)
        valid = (d_ds > 0) & (d_ren > 0)
        diff = np.abs(d_ren[valid] - d_ds[valid])
        np.add.at(hist, np.minimum((diff / BIN).astype(np.int64), len(hist) - 1), 1)
        n_valid += int(valid.sum())
        n_ds += int((d_ds > 0).sum())
        n_within2 += int((diff <= 0.02).sum())
        per_frame.append({"frame": i, "median_cm": float(np.median(diff) * 100) if diff.size else None,
                          "within2": float(np.mean(diff <= 0.02)) if diff.size else None,
                          "valid": float(valid.mean())})
        # G6: rendered instance ids + dataset depth, uniform pixel grid
        sub = (slice(None, None, G6_STRIDE), slice(None, None, G6_STRIDE))
        P = backproject(d_ds, intr, c2w)[sub]
        m = (d_ds[sub] > 0) & (inst[sub] >= 0)
        ids_f, P_f = inst[sub][m], P[m]
        order = np.argsort(ids_f, kind="stable")
        ids_f, P_f = ids_f[order], P_f[order]
        cut = np.flatnonzero(np.r_[True, ids_f[1:] != ids_f[:-1], True]) if len(ids_f) else np.zeros(1, int)
        for a, b in zip(cut[:-1], cut[1:]):
            pts.setdefault(int(ids_f[a]), []).append(P_f[a:b].astype(np.float32))
    cdf = np.cumsum(hist)
    med_bin = int(np.searchsorted(cdf, (n_valid + 1) / 2))
    median_m = (med_bin + 0.5) * BIN
    overflow = med_bin >= len(hist) - 1                    # median in the last bin = |diff| >= 1 m
    chamfer = {}
    for oid, chunks in pts.items():
        p = np.concatenate(chunks)
        if len(p) < 500 or oid not in vsets:
            continue
        d_v, _ = cKDTree(verts[vsets[oid]]).query(p, k=1)
        surf = o3d.t.geometry.RaycastingScene()
        surf.add_triangles(o3d.core.Tensor(rc.vertices), o3d.core.Tensor(rc.triangles[rc.tri_labels == oid]))
        d_s = surf.compute_distance(o3d.core.Tensor(p)).numpy()
        chamfer[oid] = {"n_points": int(len(p)), "mean_cm": float(d_v.mean() * 100),
                        "median_cm": float(np.median(d_v) * 100), "surface_mean_cm": float(d_s.mean() * 100)}
    ch = np.array([v["mean_cm"] for v in chamfer.values()])
    return {
        "scene": scene, "frames": len(frames), "step": step, "pose_fix": pose_fix,
        "G5_median_abs_diff_cm": float(median_m * 100), "G5_median_overflow": overflow, "G5_within_2cm": n_within2 / max(n_valid, 1),
        "valid_fraction_of_dataset_pixels": n_valid / max(n_ds, 1),
        "G6_objects": len(chamfer), "G6_median_chamfer_cm": float(np.median(ch)) if ch.size else None,
        "G6_p90_chamfer_cm": float(np.percentile(ch, 90)) if ch.size else None,
        "G6_median_surface_chamfer_cm": float(np.median([v["surface_mean_cm"] for v in chamfer.values()]))
        if chamfer else None,
        "per_frame": per_frame, "per_object": chamfer,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=["room0", "office0"])
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--n-frames", type=int, default=None)
    ap.add_argument("--pose-fix", default="none", choices=["none", "opengl", "yup"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out, ok = [], True
    for s in a.scenes:
        r = prove_scene(s, a.step, a.n_frames, a.pose_fix)
        g5 = r["G5_median_abs_diff_cm"] <= 1.0 and r["G5_within_2cm"] >= 0.95 and r["frames"] >= 100
        g6 = r["G6_median_chamfer_cm"] is not None and r["G6_median_chamfer_cm"] <= 2.0
        ok &= g5 and g6
        print(f"{s}: frames={r['frames']} G5 median={'>= 100 cm (overflow bin)' if r['G5_median_overflow'] else format(r['G5_median_abs_diff_cm'], '.3f') + ' cm'} "
              f"within2cm={r['G5_within_2cm']:.4f} valid={r['valid_fraction_of_dataset_pixels']:.4f} -> "
              f"{'PASS' if g5 else 'FAIL'} | G6 objects={r['G6_objects']} median chamfer="
              f"{r['G6_median_chamfer_cm']} cm p90={r['G6_p90_chamfer_cm']} (surface "
              f"{r['G6_median_surface_chamfer_cm']}) -> {'PASS' if g6 else 'FAIL'}")
        out.append(r)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
