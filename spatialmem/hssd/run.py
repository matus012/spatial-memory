"""Task 002 driver, inside hpc/habitat.def on a PERUN GPU node.

  python -m spatialmem.hssd.run probe                                   # G1: version, GL renderer
  python -m spatialmem.hssd.run scene --scene S --out /runs/hssd [--max-frames 100] [--no-images]
      -> <out>/<S>/gt.json (objects, relations v2, meta), stream (rgb/*.jpg, depth/*.png, inst/*.png, poses.txt,
         meta.json), check.json (G3 / G4 / G6), timing
  python -m spatialmem.hssd.run scenes --list dev|eval|all --out /runs/hssd    # loop, idempotent per scene
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from spatialmem.hssd import labels
from spatialmem.hssd.check import StreamCheck
from spatialmem.hssd.sim import STAGE_ID, assign_ids, instances, load_camera, make_sim, render, set_pose
from spatialmem.hssd.traj import coverage_poses, indoor_test, recompute_navmesh

SCENES = Path("/workspace/configs/hssd_scenes.yaml")


def gl_info() -> dict:
    try:
        from magnum import gl
        c = gl.Context.current
        return {"renderer": c.renderer_string, "vendor": c.vendor_string, "version": c.version_string}
    except Exception as e:  # noqa: BLE001 - diagnostic only
        return {"error": repr(e)}


def probe() -> None:
    import habitat_sim
    cam, _ = load_camera()
    scene = yaml.safe_load(SCENES.read_text())["dev"][0]
    sim = make_sim(scene, cam)
    rgb, depth, inst = render(sim)
    info = {"habitat_sim": habitat_sim.__version__, "built_with_bullet": habitat_sim.built_with_bullet,
            "cuda_enabled": getattr(habitat_sim, "cuda_enabled", None), "gl": gl_info(), "scene": scene,
            "rgb_mean": float(rgb.mean()), "depth_finite": float(np.isfinite(depth).mean())}
    print("PROBE", json.dumps(info), flush=True)
    sim.close()


def write_png16(path: Path, a: np.ndarray) -> None:
    from PIL import Image
    Image.fromarray(a.astype(np.uint16)).save(path)


def run_scene(scene: str, out_root: Path, max_frames: int | None, images: bool) -> dict:
    from PIL import Image
    from spatialmem.eval.contract import PredObject
    from spatialmem.gt.relations import compute_relations

    out = out_root / scene
    if (out / "check.json").exists():
        print(f"{scene}: done already", flush=True)
        return json.loads((out / "check.json").read_text())
    for d in ("rgb", "depth", "inst"):
        (out / d).mkdir(parents=True, exist_ok=True)
    cam, cfg = load_camera()
    t0 = time.time()
    sim = make_sim(scene, cam)
    t_load = time.time() - t0
    recs = instances(sim)
    gts = assign_ids(sim, recs)
    by_h = {r["handle"]: r for r in recs}
    for r in recs:
        r["label"] = labels.label(r)
    for g in gts:
        g["label"] = labels.group_label([by_h[h] for h in g["members"]])
    gt_objs = [PredObject(g["id"], g["label"], np.array(g["aabb_min"]), np.array(g["aabb_max"])) for g in gts
               if g["label"] not in labels.EXCLUDED]
    rels = compute_relations(gt_objs, up=1)
    gt = {"scene": scene, "up_axis": 1, "stage_id": STAGE_ID, "label_rule": labels.RULE,
          "objects": [{"id": o.id, "label": o.label, "aabb_min": o.aabb_min.tolist(), "aabb_max": o.aabb_max.tolist()}
                      for o in gt_objs],
          "relations": [{"subj": r.subj, "pred": r.pred, "obj": r.obj} for r in rels],
          "gt_groups": gts, "instances": recs, "relation_rules": "docs/RELATIONS.md v2 (2026-10-27)",
          "n_instances": len(recs), "n_groups": len(gts), "n_part_groups": sum(1 for g in gts if g["parent"])}
    (out / "gt.json").write_text(json.dumps(gt, indent=1))

    recompute_navmesh(sim, cfg["agent_radius_m"], cam.height_m)
    poses, path_stats = coverage_poses(sim.pathfinder, cfg, indoor_test(sim, cam.height_m))
    if max_frames:
        poses = poses[:max_frames]
    intr = cam.intrinsics()
    chk = StreamCheck(gts, intr, cover_ids=[o.id for o in gt_objs])
    c2ws, ms_render, ms_total, g2 = [], [], [], []
    for i, (p, yaw) in enumerate(poses):
        ta = time.time()
        c2w = set_pose(sim, np.asarray(p) + [0.0, cam.height_m, 0.0], yaw)
        tb = time.time()
        rgb, depth, inst = render(sim)
        tc = time.time()
        depth = np.where(np.isfinite(depth) & (depth <= cam.depth_max_m), depth, 0.0)
        chk.add(depth, inst, c2w)
        g2.append((float(rgb.std()), float((depth > 0).mean()), int((inst > 0).sum())))
        if images:
            Image.fromarray(rgb).save(out / "rgb" / f"{i:06d}.jpg", quality=95)
            write_png16(out / "depth" / f"{i:06d}.png", np.round(depth * cam.depth_scale))
            write_png16(out / "inst" / f"{i:06d}.png", inst)
        c2ws.append(c2w)
        ms_render.append((tc - tb) * 1000)
        ms_total.append((time.time() - ta) * 1000)
    np.savetxt(out / "poses.txt", np.array(c2ws).reshape(len(c2ws), 16))
    g2a = np.array(g2)
    res = chk.result()
    res.update({"scene": scene, "path": path_stats, "load_s": t_load, "gl": gl_info(),
                "ms_render_median": float(np.median(ms_render)), "ms_frame_median": float(np.median(ms_total)),
                "G2_rgb_nonempty": int((g2a[:, 0] > 0).sum()), "G2_depth_valid_min": float(g2a[:, 1].min()),
                "G2_depth_valid_median": float(np.median(g2a[:, 1])), "G2_inst_nonempty": int((g2a[:, 2] > 0).sum()),
                "G2_frames": len(g2a), "n_gt_objects": len(gt_objs), "n_relations": len(rels),
                "bb_check_mm_max": float(max(r["bb_check_mm"] for r in recs)),
                "bb_check_mm_gt1": int(sum(r["bb_check_mm"] > 1 for r in recs)),
                "root_check_mm_max": float(max(r["root_check_mm"] for r in recs)),
                "root_check_mm_gt1": int(sum(r["root_check_mm"] > 1 for r in recs)),
                "multi_node_instances": int(sum(r["n_node_transforms"] > 1 for r in recs))})
    meta = {"scene": scene, "intrinsics": intr, "depth_scale": cam.depth_scale, "pose": "c2w, OpenCV axes, row-major 4x4 per line",
            "inst": "uint16 png, 0 = stage/empty, else gt.json gt_groups id", "camera": cfg, "n_frames": len(c2ws)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    (out / "check.json").write_text(json.dumps(res, indent=1))
    sim.close()
    print(f"{scene}: frames={res['frames']} gt={len(gt_objs)} rel={len(rels)} G3 dup={res['G3_duplicate_gt_ids']} "
          f"unknown={len(res['G3_unknown_rendered_ids'])} G4 med={res['G4_median_inside']} ge0.9={res['G4_frac_objects_ge_0.90']} "
          f"G6={res['G6_coverage']} render {res['ms_render_median']:.1f} ms frame {res['ms_frame_median']:.1f} ms "
          f"bbchk>1mm={res['bb_check_mm_gt1']} rootchk>1mm={res['root_check_mm_gt1']} multinode={res['multi_node_instances']}", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "scene", "scenes"])
    ap.add_argument("--scene")
    ap.add_argument("--list", default="dev")
    ap.add_argument("--out", type=Path, default=Path("/runs/hssd"))
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--part", default="0/1", help="i/n: every n-th scene of the list starting at i")
    a = ap.parse_args()
    if a.cmd == "probe":
        probe()
    elif a.cmd == "scene":
        run_scene(a.scene, a.out, a.max_frames, not a.no_images)
    else:
        d = yaml.safe_load(SCENES.read_text())
        lst = d["dev"] + d["eval"] if a.list == "all" else d[a.list]
        i, n = (int(x) for x in a.part.split("/"))
        for s in lst[i::n]:
            run_scene(s, a.out, a.max_frames, not a.no_images)
    sys.exit(0)


if __name__ == "__main__":
    main()
