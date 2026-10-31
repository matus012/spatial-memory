""""Ours" v1 object mapper (task 004): SAM 2.1 masks -> lift -> SigLIP 2 descriptor -> association -> Prediction.

  uv run python -m spatialmem.ours.run --config configs/ours_v1.yaml --dataset hssd --scenes 102816627 \
      --tag ours_smoke [--set key=value ...] [--cache-dir DIR | --cache-dir off]

Writes data/baselines/<tag>/<dataset>/<scene>.json (+ .sem.npz for Replica), <scene>.stats.json,
data/baselines/<tag>/runtime.csv and data/baselines/<tag>/run_meta.json.

Detection cache (<cache-dir>/<dataset>/<scene>/detections.npz, default data/cache/ours): every processed frame's
detections after mask filter + lift + DBSCAN (points, AABB, mask area, blend AND crop descriptors) + the text
embeddings. Key = sha256 over every config field upstream of association (UPSTREAM), the model sha256s and the stream
identity (frame count + poses file sha256). Hit -> no images, no models, association only; mismatch -> recompute and
overwrite. Fields free on a hit: descriptor, bbox_margin, geo_radius, lam, tau, tau_g, tau_c, merge_every,
merge_use_cos, top_k, min_detections.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from spatialmem.baselines.ovo import UP, vocabulary
from spatialmem.eval.boxes import from_points
from spatialmem.eval.contract import STRUCTURAL, PredObject, Prediction
from spatialmem.gt.relations import compute_relations
from spatialmem.gt.replica import ROOT
from spatialmem.ours.association import Detection, ObjectMap
from spatialmem.ours.lift import lift
from spatialmem.render import proof
from spatialmem.render.raycast import Intrinsics, backproject

log = logging.getLogger("ours")
OUT = ROOT / "data" / "baselines"
HSSD_STREAMS = ROOT / "data" / "hssd_streams"


# ------------------------------------------------------------------ streams
@dataclass
class Stream:
    n: int
    intr: Intrinsics
    rgb: Callable[[int], np.ndarray]
    depth: Callable[[int], np.ndarray]   # metres
    poses: np.ndarray        # (n, 4, 4) c2w, OpenCV axes
    poses_file: Path


def hssd_stream(scene: str, root: Path = HSSD_STREAMS) -> Stream:
    D = root / scene
    m = json.loads((D / "meta.json").read_text())
    i = m["intrinsics"]
    P = np.loadtxt(D / "poses.txt").reshape(-1, 4, 4)
    assert len(P) == m["n_frames"], (len(P), m["n_frames"])
    return Stream(m["n_frames"], Intrinsics(int(i["W"]), int(i["H"]), i["fx"], i["fy"], i["cx"], i["cy"]),
                  lambda k: np.asarray(Image.open(D / "rgb" / f"{k:06d}.jpg").convert("RGB")),
                  lambda k: np.asarray(Image.open(D / "depth" / f"{k:06d}.png"), np.float64) / m["depth_scale"], P,
                  D / "poses.txt")


def replica_stream(scene: str) -> Stream:
    intr, scale = proof.intrinsics(scene)
    P = proof.poses(scene)
    return Stream(len(P), intr, lambda k: proof.dataset_rgb(scene, k), lambda k: proof.dataset_depth(scene, k, scale), P,
                  proof.scene_dir(scene) / "traj.txt")


# ------------------------------------------------------------------ prediction
def to_prediction(scene: str, dataset: str, objs: list, text: np.ndarray, classes: list[str],
                  lab: np.ndarray | None = None) -> Prediction:
    """Label = argmax cos(descriptor, text) (CONTRACT 2026-10-27e) unless class indices `lab` are given (v2 label rule);
    relations = RELATIONS.md v2 on non-structural."""
    if lab is None:
        lab = (np.stack([o.descriptor() for o in objs]) @ text.T).argmax(1) if objs else np.zeros(0, int)
    pobjs = []
    for k, (o, li) in enumerate(zip(objs, lab)):
        lo, hi = from_points(o.points)
        pobjs.append(PredObject(id=k, label=classes[li], aabb_min=lo, aabb_max=hi))
    pred = Prediction(scene=scene, objects=pobjs,
                      relations=compute_relations([o for o in pobjs if o.label not in STRUCTURAL], up=UP[dataset]))
    if dataset == "replica":
        pred.sem_points = np.concatenate([o.points for o in objs]).astype(np.float32) if objs else np.zeros((0, 3), np.float32)
        pred.sem_labels = np.concatenate([np.full(len(o.points), li, np.int64) for o, li in zip(objs, lab)]) if objs else np.zeros(0, np.int64)
    return pred


# ------------------------------------------------------------------ bookkeeping
def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def load_config(path: Path, sets: list[str]) -> dict:
    cfg = yaml.safe_load(path.read_text())
    for s in sets:
        k, v = s.split("=", 1)
        assert k in cfg, f"unknown config key {k}"
        cfg[k] = type(cfg[k])(yaml.safe_load(v))
    return cfg


class SmiPoll(threading.Thread):
    """nvidia-smi memory.used poll (whole GPU, includes other processes), max in MiB."""

    def __init__(self, every: float = 1.0):
        super().__init__(daemon=True)
        self.every, self.peak, self.stop = every, 0, threading.Event()

    def run(self) -> None:
        while not self.stop.is_set():
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True).stdout
            self.peak = max(self.peak, int(out.split()[0]))
            self.stop.wait(self.every)


# ------------------------------------------------------------------ detection cache
UPSTREAM = ("stride", "min_mask_frac", "points_per_batch", "voxel", "dbscan_eps", "dbscan_min_points", "crop_pad_px",
            "sam_cfg", "seed")
CACHE = ROOT / "data" / "cache" / "ours"


def cache_key(cfg: dict, model_sha: dict, st: Stream) -> str:
    d = {"cfg": {k: cfg[k] for k in UPSTREAM}, "models": model_sha, "n_frames": st.n, "poses_sha256": sha256(st.poses_file)}
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


def save_cache(path: Path, key: str, frames: list[list[tuple]], n_masks: list[int], text: np.ndarray,
               classes: list[str]) -> None:
    dets = [d for f in frames for d in f]
    pts = [d[0] for d in dets]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, key=np.array(key), frame_ndet=np.array([len(x) for x in frames]), frame_nmasks=np.array(n_masks),
                 points=np.concatenate(pts) if pts else np.zeros((0, 3)), pt_off=np.cumsum([0] + [len(p) for p in pts]),
                 aabb=np.array([[p.min(0), p.max(0)] for p in pts]).reshape(-1, 2, 3), area=np.array([d[1] for d in dets]),
                 feat_blend=np.stack([d[2]["blend"] for d in dets]), feat_crop=np.stack([d[2]["crop"] for d in dets]),
                 text=text, classes=np.array(classes))
    tmp.replace(path)
    (path.parent / "key.json").write_text(json.dumps({"key": key}))


def cache_valid(path: Path, key: str) -> bool:
    """True if the cache file exists and its key matches (reads only the key member, not the arrays)."""
    if not path.exists():
        return False
    with np.load(path) as z:
        ok = str(z["key"]) == key
    if not ok:
        log.info("cache key mismatch at %s: recompute", path)
    return ok


def load_cache(path: Path, key: str) -> dict | None:
    """The cached arrays if the file exists and its key matches, else None (a stale cache is never reused)."""
    if not cache_valid(path, key):
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def replay(c: dict) -> Iterator[tuple[int, list[tuple]]]:
    """Per processed frame: (n_masks, [(points, area, {"blend": f, "crop": f})])."""
    d = 0
    for nm, nd in zip(c["frame_nmasks"], c["frame_ndet"]):
        yield int(nm), [(c["points"][c["pt_off"][i]:c["pt_off"][i + 1]], int(c["area"][i]),
                         {"blend": c["feat_blend"][i], "crop": c["feat_crop"][i]}) for i in range(d, d + nd)]
        d += nd


def detect(st: Stream, cfg: dict, masker, siglip, t: collections.Counter) -> Iterator[tuple[int, list[tuple]]]:
    import torch
    for k in range(0, st.n, cfg["stride"]):
        t0 = time.perf_counter()
        rgb, depth = st.rgb(k), st.depth(k)
        t1 = time.perf_counter()
        masks = masker(rgb)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        world, valid = backproject(depth, st.intr, st.poses[k]), depth > 0
        pts = [lift(world, valid, m, cfg["voxel"], cfg["dbscan_eps"], cfg["dbscan_min_points"]) for m in masks]
        keep = [i for i, p in enumerate(pts) if len(p)]
        t3 = time.perf_counter()
        feats = siglip.masks(rgb, [masks[i] for i in keep], cfg["crop_pad_px"]) if keep else {}
        t4 = time.perf_counter()
        t.update(io=t1 - t0, masks=t2 - t1, lift=t3 - t2, descriptor=t4 - t3)
        yield len(masks), [(pts[i], int(masks[i].sum()), {"blend": feats["blend"][j], "crop": feats["crop"][j]})
                           for j, i in enumerate(keep)]


# ------------------------------------------------------------------ main loop
def map_scene(frames: Iterator[tuple[int, list[tuple]]], n_used: int, cfg: dict, t: collections.Counter,
              record: list | None = None, record_masks: list | None = None) -> tuple[list, dict]:
    """Association over per-frame detections (from `detect` or `replay`); optionally records them for the cache."""
    M = ObjectMap(cfg)
    n_masks = n_dets = 0
    for f_i, (nm, dets) in enumerate(frames):
        t0 = time.perf_counter()
        M.integrate([Detection(p, f[cfg["descriptor"]], a) for p, a, f in dets])
        if (f_i + 1) % cfg["merge_every"] == 0:
            M.merge_pass()
        t["association"] += time.perf_counter() - t0
        n_masks += nm
        n_dets += len(dets)
        if record is not None:
            record.append(dets)
            record_masks.append(nm)
        if f_i % 20 == 0:
            log.info("frame %d/%d: masks %d dets %d objects %d", f_i + 1, n_used, nm, len(dets), len(M.objects))
    t0 = time.perf_counter()
    merged = M.merge_pass()
    t["association"] += time.perf_counter() - t0
    final = M.final()
    return final, {"frames_used": n_used, "masks": n_masks, "detections": n_dets,
                   "objects_before_min_det": len(M.objects), "objects_after_min_det": len(final),
                   "final_merge_pass_merged": merged, "stage_s": {k: round(v, 2) for k, v in t.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--dataset", required=True, choices=["replica", "hssd"])
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--set", nargs="*", default=[], dest="sets")
    ap.add_argument("--cache-dir", default=str(CACHE), help="detection cache root (<dir>/<dataset>/<scene>/), or 'off'")
    a = ap.parse_args()
    import torch
    cfg = load_config(a.config, a.sets)
    out = OUT / a.tag
    (out / a.dataset).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / "run.log")])
    log.setLevel(logging.INFO)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    model_sha = {"sam2": sha256(ROOT / cfg["sam_ckpt"]), "siglip": sha256(ROOT / cfg["siglip"] / "model.safetensors")}
    classes = vocabulary(a.dataset)
    streams = {s: hssd_stream(s) if a.dataset == "hssd" else replica_stream(s) for s in a.scenes}
    cpath = {s: None if a.cache_dir == "off" else Path(a.cache_dir) / a.dataset / s / "detections.npz" for s in a.scenes}
    keys = {s: cache_key(cfg, model_sha, st) for s, st in streams.items()}
    # keys only up front; a scene's arrays are loaded when it is processed (8 Replica caches ~20 GB > lpc RAM)
    status = {s: "off" if cpath[s] is None else ("hit" if cache_valid(cpath[s], keys[s]) else "miss") for s in a.scenes}
    log.info("cache %s", status)
    masker = siglip = text = dev = None
    if any(v != "hit" for v in status.values()):
        log.info("torch %s cuda_available %s device %s", torch.__version__, torch.cuda.is_available(),
                 torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
        assert torch.cuda.is_available(), "CPU fallback: STOP"
        from spatialmem.ours.models import Masker, Siglip
        masker = Masker(cfg["sam_cfg"], str(ROOT / cfg["sam_ckpt"]), cfg["points_per_batch"], cfg["min_mask_frac"])
        siglip = Siglip(str(ROOT / cfg["siglip"]))
        dev = {"sam2": str(next(masker.model.parameters()).device), "siglip": str(next(siglip.model.parameters()).device)}
        log.info("model devices %s", dev)
        assert all(v.startswith("cuda") for v in dev.values()), dev
        text = siglip.text(classes)
    else:
        log.info("all scenes are cache hits: no images, no models, association only (CPU)")
    resolved = yaml.safe_dump(cfg, sort_keys=True)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT).stdout.strip())
    meta = {"tag": a.tag, "dataset": a.dataset, "scenes": a.scenes, "config_file": str(a.config), "sets": a.sets,
            "config": cfg, "config_sha256": hashlib.sha256(resolved.encode()).hexdigest(),
            "config_file_sha256": sha256(a.config), "git_head": git, "git_dirty": dirty, "devices": dev,
            "torch": torch.__version__, "vocab_size": len(classes), "cache": status, "cache_keys": keys,
            "cache_dir": a.cache_dir,
            "models": {"sam2": {"path": cfg["sam_ckpt"], "sha256": model_sha["sam2"]},
                       "siglip": {"path": cfg["siglip"], "sha256": model_sha["siglip"]}}}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1))
    rt = out / "runtime.csv"
    if not rt.exists():
        rt.write_text("dataset,scene,frames,frames_used,wall_s,s_per_frame,s_per_frame_scope,peak_vram_gb,smi_peak_gb,cache\n")
    for scene in a.scenes:
        st, c = streams[scene], (load_cache(cpath[scene], keys[scene]) if status[scene] == "hit" else None)
        n_used = len(range(0, st.n, cfg["stride"]))
        t: collections.Counter = collections.Counter()
        gpu = c is None
        if gpu:
            torch.cuda.reset_peak_memory_stats()
            poll = SmiPoll()
            poll.start()
        t0 = time.perf_counter()
        if gpu:
            rec, rec_m = ([], []) if status[scene] == "miss" else (None, None)
            objs, stats = map_scene(detect(st, cfg, masker, siglip, t), n_used, cfg, t, rec, rec_m)
            scene_text = text
        else:
            assert [str(x) for x in c["classes"]] == classes, "cached vocabulary != vocabulary"
            objs, stats = map_scene(replay(c), n_used, cfg, t)
            scene_text = c["text"]
        wall = time.perf_counter() - t0
        peak = smi = 0.0
        if gpu:
            poll.stop.set()
            poll.join()
            peak, smi = torch.cuda.max_memory_allocated() / 2**30, poll.peak / 1024
        if status[scene] == "miss":
            save_cache(cpath[scene], keys[scene], rec, rec_m, text, classes)
        pred = to_prediction(scene, a.dataset, objs, scene_text, classes)
        pred.to_json(out / a.dataset / f"{scene}.json")
        if a.dataset == "replica":   # diagnostic sidecar (not CONTRACT): object index of every sem point
            np.save(out / a.dataset / f"{scene}.sem_obj.npy",
                    np.concatenate([np.full(len(o.points), k, np.int32) for k, o in enumerate(objs)]) if objs else np.zeros(0, np.int32))
        scope ="full" if gpu else "association_only"
        stats.update(scene=scene, frames=st.n, cache=status[scene], wall_s=round(wall, 1),
                     s_per_frame=round(wall / n_used, 3), s_per_frame_scope=scope, peak_vram_gb=round(peak, 3),
                     smi_peak_gb=round(smi, 3), relations=len(pred.relations),
                     label_hist=collections.Counter(o.label for o in pred.objects).most_common())
        (out / a.dataset / f"{scene}.stats.json").write_text(json.dumps(stats, indent=1))
        with open(rt, "a") as f:
            f.write(f"{a.dataset},{scene},{st.n},{n_used},{wall:.1f},{stats['s_per_frame']},{scope},{peak:.3f},{smi:.3f},"
                    f"{status[scene]}\n")
        log.info("scene %s done: %s", scene, json.dumps(stats))
        assert peak <= 7.5, f"peak VRAM {peak:.2f} GB > 7.5"


if __name__ == "__main__":
    main()
