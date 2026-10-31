""""Ours" v2 (task 006): v1 + encoder knob + label rule (mean | vote), with caches so every tuning config is an
association-only CPU re-run.

  uv run python -m spatialmem.ours.v2 --config configs/ours_v2.yaml --dataset hssd --scenes 102816627 --tag T \
      [--set key=value ...] [--build-only]

Per scene, under data/cache/ours/<dataset>/<scene>/ (all keyed off the v1 detection-cache key `run.cache_key`):
- detections.npz  v1 detection cache (points, areas, SigLIP 2 base blend/crop descriptors), unchanged format;
- masks.npz       the kept masks (detection order), tight box + bit-packed pixels; built with SAM 2.1-s + the v1
                  detect loop. If detections.npz already exists, the regenerated detections are asserted equal to it;
- desc_<encoder>.npz  blend + crop descriptor per detection (detection order) + that encoder's text embeddings.
Phases: A (SAM + base, scenes lacking masks) -> B (encoder, scenes lacking desc) -> association (CPU, one scene's
arrays at a time). A run whose caches all hit loads no images and no models.
Writes data/baselines/<tag>/<dataset>/<scene>.json (+ .sem.npz, .sem_obj.npy for Replica / Replica-dev),
<scene>.stats.json, runtime.csv (one row per scene x phase), run_meta.json.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from spatialmem.baselines.ovo import vocabulary
from spatialmem.eval.contract import Prediction
from spatialmem.gt.replica import ROOT
from spatialmem.ours import run
from spatialmem.ours.encoders import ENCODERS, weights
from spatialmem.render import proof
from spatialmem.render.raycast import Intrinsics

log = logging.getLogger("ours")
REPLICA_DEV = ROOT / "data" / "replica_dev"
VOCAB = {"hssd": "hssd", "replica": "replica", "replica_dev": "replica"}


# ------------------------------------------------------------------ Replica-dev stream (NICE layout)
def replica_dev_stream(scene: str, root: Path = REPLICA_DEV) -> run.Stream:
    """<root>/<scene>/results/frame%06d.jpg + depth%06d.png (/ scale), <root>/<scene>/traj.txt (c2w OpenCV);
    intrinsics from <root>/<scene>/cam_params.json, else <root>/cam_params.json, else the NICE Replica camera."""
    D, intr, scale = root / scene, proof.REPLICA_INTR, proof.DEPTH_SCALE
    cp = next((f for f in (D / "cam_params.json", root / "cam_params.json") if f.exists()), None)
    if cp is not None:
        c = json.loads(cp.read_text())["camera"]
        intr = Intrinsics(int(c["w"]), int(c["h"]), float(c["fx"]), float(c["fy"]), float(c["cx"]), float(c["cy"]))
        scale = float(c["scale"])
    P = np.loadtxt(D / "traj.txt").reshape(-1, 4, 4)
    return run.Stream(len(P), intr, lambda k: np.asarray(Image.open(D / "results" / f"frame{k:06d}.jpg").convert("RGB")),
                      lambda k: np.asarray(Image.open(D / "results" / f"depth{k:06d}.png"), np.float64) / scale, P,
                      D / "traj.txt")


def stream(dataset: str, scene: str) -> run.Stream:
    return {"hssd": run.hssd_stream, "replica": run.replica_stream, "replica_dev": replica_dev_stream}[dataset](scene)


# ------------------------------------------------------------------ label rule
def labels(objs: list, text: np.ndarray, rule: str) -> np.ndarray:
    """mean = v1 (argmax of the mean top-k descriptor). vote = per top-k view (by mask area) argmax label, majority
    over views; a tie between labels -> the mean label."""
    if not objs:
        return np.zeros(0, int)
    mean = (np.stack([o.descriptor() for o in objs]) @ text.T).argmax(1)
    if rule == "mean":
        return mean
    assert rule == "vote", rule
    out = mean.copy()
    for i, o in enumerate(objs):
        cnt = np.bincount((np.stack([f for _, f in o.views]) @ text.T).argmax(1))
        top = np.flatnonzero(cnt == cnt.max())
        if len(top) == 1:
            out[i] = top[0]
    return out


# ------------------------------------------------------------------ mask cache
class MaskRecorder:
    """Wraps the v1 descriptor model: records every mask passed to `masks` (= the kept masks, detection order)."""

    def __init__(self, enc):
        self.enc, self.box, self.bits = enc, [], []

    def masks(self, rgb: np.ndarray, masks: list[np.ndarray], pad: int) -> dict[str, np.ndarray]:
        for m in masks:
            ys, xs = np.nonzero(m)
            y0, x0 = ys.min(), xs.min()
            h, w = ys.max() - y0 + 1, xs.max() - x0 + 1
            self.box.append((y0, x0, h, w))
            self.bits.append(np.packbits(m[y0:y0 + h, x0:x0 + w]))
        return self.enc.masks(rgb, masks, pad)


def save_masks(path: Path, key: str, hw: tuple[int, int], box: list, bits: list) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, key=np.array(key), hw=np.array(hw), box=np.array(box, np.int32).reshape(-1, 4),
                 bits=np.concatenate(bits) if bits else np.zeros(0, np.uint8),
                 off=np.cumsum([0] + [len(b) for b in bits]).astype(np.int64))
    tmp.replace(path)


def mask(z: dict, i: int) -> np.ndarray:
    y0, x0, h, w = (int(v) for v in z["box"][i])
    m = np.zeros(tuple(int(v) for v in z["hw"]), bool)
    m[y0:y0 + h, x0:x0 + w] = np.unpackbits(z["bits"][z["off"][i]:z["off"][i + 1]], count=h * w).reshape(h, w)
    return m


def key_of(path: Path) -> str | None:
    if not path.exists():
        return None
    with np.load(path) as z:
        return str(z["key"])


def load(path: Path, members: tuple[str, ...] | None = None) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in (members or z.files)}


# ------------------------------------------------------------------ descriptor cache
def desc_key(det_key: str, encoder: str, enc_sha: str) -> str:
    return hashlib.sha256(json.dumps({"detections": det_key, "encoder": encoder, "sha256": enc_sha},
                                     sort_keys=True).encode()).hexdigest()


def save_desc(path: Path, key: str, blend: np.ndarray, crop: np.ndarray, text: np.ndarray, classes: list[str]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, key=np.array(key), feat_blend=blend, feat_crop=crop, text=text, classes=np.array(classes))
    tmp.replace(path)


def descriptors(st: run.Stream, cfg: dict, frame_ndet: np.ndarray, mz: dict, enc, t: collections.Counter
                ) -> tuple[np.ndarray, np.ndarray]:
    """Per processed frame (batched per frame, as v1): the encoder's blend + crop descriptors of the cached masks."""
    import torch
    blend, crop, d = [], [], 0
    for f, nd in enumerate(frame_ndet):
        if nd:
            t0 = time.perf_counter()
            rgb = st.rgb(f * cfg["stride"])
            ms = [mask(mz, i) for i in range(d, d + nd)]
            t1 = time.perf_counter()
            out = enc.masks(rgb, ms, cfg["crop_pad_px"])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t.update(io=t1 - t0, descriptor=time.perf_counter() - t1)
            blend.append(out["blend"])
            crop.append(out["crop"])
            d += nd
        if f % 50 == 0:
            log.info("desc frame %d/%d", f + 1, len(frame_ndet))
    assert d == len(mz["box"]), (d, len(mz["box"]))
    return np.concatenate(blend), np.concatenate(crop)


# ------------------------------------------------------------------ association
def associate(scene: str, ds: str, det: dict, dz: dict, cfg: dict, classes: list[str], n_used: int,
              t: collections.Counter) -> tuple[list, Prediction, dict]:
    """v1 association over the cached detections with the encoder's descriptors swapped in, then the label rule.
    `det` (a loaded detections.npz) is modified in place."""
    assert len(dz["feat_crop"]) == len(det["area"]), scene
    det["feat_blend"], det["feat_crop"] = dz["feat_blend"], dz["feat_crop"]
    objs, stats = run.map_scene(run.replay(det), n_used, cfg, t)
    pred = run.to_prediction(scene, ds, objs, dz["text"], classes, labels(objs, dz["text"], cfg["label"]))
    return objs, pred, stats


# ------------------------------------------------------------------ main
def gpu_check(models: dict) -> dict:
    import torch
    log.info("torch %s cuda_available %s device %s", torch.__version__, torch.cuda.is_available(),
             torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
    assert torch.cuda.is_available(), "CPU fallback: STOP"
    dev = {k: str(next(m.parameters()).device) for k, m in models.items()}
    log.info("model devices %s", dev)
    assert all(v.startswith("cuda") for v in dev.values()), dev
    return dev


def smi_now() -> int:
    return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.split()[0])


def free() -> None:
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--dataset", required=True, choices=["replica", "hssd", "replica_dev"])
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--set", nargs="*", default=[], dest="sets")
    ap.add_argument("--build-only", action="store_true", help="build missing caches, no association / predictions")
    ap.add_argument("--meta", default="run_meta.json", help="run-meta file name under the tag dir (PERUN: one per process)")
    a = ap.parse_args()
    import torch
    cfg = run.load_config(a.config, a.sets)
    assert cfg["encoder"] in ENCODERS and cfg["label"] in ("mean", "vote"), (cfg["encoder"], cfg["label"])
    out = run.OUT / a.tag
    (out / a.dataset).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / "run.log")])
    log.setLevel(logging.INFO)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    enc = cfg["encoder"]
    sha = {"sam2": run.sha256(ROOT / cfg["sam_ckpt"]), "siglip": run.sha256(ROOT / cfg["siglip"] / "model.safetensors")}
    enc_sha = run.sha256(weights(enc))
    classes = vocabulary(VOCAB[a.dataset])
    streams = {s: stream(a.dataset, s) for s in a.scenes}
    cdir = {s: run.CACHE / a.dataset / s for s in a.scenes}
    keys = {s: run.cache_key(cfg, sha, st) for s, st in streams.items()}
    dkeys = {s: desc_key(keys[s], enc, enc_sha) for s in a.scenes}
    dname = f"desc_{enc}.npz"
    need_a = [s for s in a.scenes if key_of(cdir[s] / "masks.npz") != keys[s]]
    need_b = [s for s in a.scenes if key_of(cdir[s] / dname) != dkeys[s]]
    status = {s: "hit" if s not in need_a + need_b else f"miss:{'A' if s in need_a else ''}{'B' if s in need_b else ''}"
              for s in a.scenes}
    log.info("cache %s", status)
    rt = out / "runtime.csv"
    if not rt.exists():
        rt.write_text("dataset,scene,phase,encoder,frames,frames_used,wall_s,s_per_frame,peak_alloc_gb,"
                      "peak_reserved_gb,smi_peak_gb,stage_s\n")
    timing: dict[str, dict] = {s: {} for s in a.scenes}

    def row(scene: str, phase: str, wall: float, t: collections.Counter, gpu: bool, smi: float = 0.0) -> None:
        n_used = len(range(0, streams[scene].n, cfg["stride"]))
        al = torch.cuda.max_memory_allocated() / 2**30 if gpu else 0.0
        rs = torch.cuda.max_memory_reserved() / 2**30 if gpu else 0.0
        st = {k: round(v, 2) for k, v in t.items()}
        timing[scene][phase] = {"wall_s": round(wall, 1), "s_per_frame": round(wall / n_used, 3),
                                "peak_alloc_gb": round(al, 3), "peak_reserved_gb": round(rs, 3),
                                "smi_peak_gb": round(smi, 3), "stage_s": st}
        with open(rt, "a") as f:
            f.write(f"{a.dataset},{scene},{phase},{enc if phase != 'A' else 'siglip2_base'},{streams[scene].n},{n_used},"
                    f"{wall:.1f},{wall / n_used:.3f},{al:.3f},{rs:.3f},{smi:.3f},\"{json.dumps(st)}\"\n")
        # as v1: allocated, not reserved (the caching allocator on a large GPU keeps blocks an 8 GB card would free;
        # the lpc fit with SAM is measured by spatialmem.ours.vram_probe)
        assert al <= 7.5, f"peak allocated VRAM {al:.2f} GB > 7.5"

    devices = {}
    smi_idle = smi_now() if need_a or need_b else 0
    log.info("nvidia-smi idle %d MiB", smi_idle)
    # ---- phase A: SAM 2.1-s + SigLIP 2 base (the v1 detect loop), record the kept masks
    if need_a:
        from spatialmem.ours.models import Masker, Siglip
        masker = Masker(cfg["sam_cfg"], str(ROOT / cfg["sam_ckpt"]), cfg["points_per_batch"], cfg["min_mask_frac"])
        base = Siglip(str(ROOT / cfg["siglip"]))
        devices["A"] = gpu_check({"sam2": masker.model, "siglip": base.model})
        text = base.text(classes)
        for s in need_a:
            st, t = streams[s], collections.Counter()
            torch.cuda.reset_peak_memory_stats()
            poll = run.SmiPoll()
            poll.start()
            t0 = time.perf_counter()
            rec, frames, nm = MaskRecorder(base), [], []
            for f_i, (n, dets) in enumerate(run.detect(st, cfg, masker, rec, t)):
                frames.append(dets)
                nm.append(n)
                if f_i % 20 == 0:
                    log.info("A %s frame %d: masks %d dets %d", s, f_i + 1, n, len(dets))
            wall = time.perf_counter() - t0
            poll.stop.set()
            poll.join()
            row(s, "A", wall, t, True, poll.peak / 1024)
            cdir[s].mkdir(parents=True, exist_ok=True)
            det = cdir[s] / "detections.npz"
            if key_of(det) == keys[s]:      # v1 cache exists: the regenerated detections must equal it exactly
                chk = cdir[s] / "detections.v2check.npz"
                run.save_cache(chk, keys[s], frames, nm, text, classes)
                old, new = load(det), load(chk)
                bad = [k for k in old if not np.array_equal(old[k], new[k])]
                assert not bad and set(old) == set(new), f"{s}: regenerated detections != v1 cache: {bad}"
                log.info("%s: regenerated detections == v1 cache (%s)", s, ", ".join(sorted(old)))
                del old, new
                chk.unlink()
                (cdir[s] / "key.json").write_text(json.dumps({"key": keys[s]}))   # save_cache rewrote it; same key
            else:
                run.save_cache(det, keys[s], frames, nm, text, classes)
            save_masks(cdir[s] / "masks.npz", keys[s], (st.intr.H, st.intr.W), rec.box, rec.bits)
            assert len(rec.box) == sum(len(f) for f in frames)
            del frames, rec
        del masker, base
        free()
    # ---- phase B: the configured encoder over the cached masks
    if need_b:
        from spatialmem.ours.encoders import load as load_enc
        model = load_enc(enc)
        devices["B"] = gpu_check({enc: model.model})
        text = model.text(classes)
        for s in need_b:
            torch.cuda.reset_peak_memory_stats()
            poll = run.SmiPoll()
            poll.start()
            t, t0 = collections.Counter(), time.perf_counter()
            det = load(cdir[s] / "detections.npz", ("key", "frame_ndet"))
            assert str(det["key"]) == keys[s]
            mz = load(cdir[s] / "masks.npz")
            blend, crop = descriptors(streams[s], cfg, det["frame_ndet"], mz, model, t)
            wall = time.perf_counter() - t0
            poll.stop.set()
            poll.join()
            row(s, "B", wall, t, True, poll.peak / 1024)
            if enc == "siglip2_base":       # masks cache proof: base descriptors from cached masks == v1 cache
                v1 = load(cdir[s] / "detections.npz", ("feat_blend", "feat_crop", "text"))
                assert np.array_equal(v1["feat_blend"], blend) and np.array_equal(v1["feat_crop"], crop), s
                assert np.array_equal(v1["text"], text), s
                log.info("%s: siglip2_base descriptors from masks.npz == v1 cache", s)
            save_desc(cdir[s] / dname, dkeys[s], blend, crop, text, classes)
            del mz, blend, crop
        del model
        free()
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip() \
        or "snapshot " + (ROOT / "REV").read_text().strip()            # PERUN: git archive snapshot (push_repo.sh)
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT).stdout.strip())
    meta = {"tag": a.tag, "dataset": a.dataset, "scenes": a.scenes, "config_file": str(a.config), "sets": a.sets,
            "config": cfg, "config_sha256": hashlib.sha256(yaml.safe_dump(cfg, sort_keys=True).encode()).hexdigest(),
            "config_file_sha256": run.sha256(a.config), "git_head": git, "git_dirty": dirty, "devices": devices,
            "torch": torch.__version__, "vocab_size": len(classes), "cache": status, "cache_keys": keys,
            "desc_keys": dkeys, "build_only": a.build_only,
            "models": {"sam2": {"path": cfg["sam_ckpt"], "sha256": sha["sam2"]},
                       "siglip2_base_detections": {"path": cfg["siglip"], "sha256": sha["siglip"]},
                       "encoder": {"name": enc, "path": str(weights(enc).relative_to(ROOT)), "sha256": enc_sha}},
            "smi_idle_mib": smi_idle, "timing": timing}
    (out / a.meta).write_text(json.dumps(meta, indent=1))
    if a.build_only:
        return
    # ---- association: CPU, one scene's arrays at a time
    for s in a.scenes:
        st, t = streams[s], collections.Counter()
        c = load(cdir[s] / "detections.npz")
        dz = load(cdir[s] / dname)
        assert str(c["key"]) == keys[s] and str(dz["key"]) == dkeys[s], s
        assert [str(x) for x in dz["classes"]] == classes, "cached vocabulary != vocabulary"
        ds = "hssd" if a.dataset == "hssd" else "replica"
        t0 = time.perf_counter()
        objs, pred, stats = associate(s, ds, c, dz, cfg, classes, len(range(0, st.n, cfg["stride"])), t)
        wall = time.perf_counter() - t0
        row(s, "assoc", wall, t, False)
        pred.to_json(out / a.dataset / f"{s}.json")
        if ds == "replica":
            np.save(out / a.dataset / f"{s}.sem_obj.npy",
                    np.concatenate([np.full(len(o.points), k, np.int32) for k, o in enumerate(objs)]) if objs else np.zeros(0, np.int32))
        hist = collections.Counter(o.label for o in pred.objects)
        stats.update(scene=s, frames=st.n, cache=status[s], encoder=enc, label_rule=cfg["label"],
                     relations=len(pred.relations), objects=len(pred.objects),
                     object__outside_share=round(hist.get("object__outside", 0) / max(len(pred.objects), 1), 4),
                     label_hist=hist.most_common(), timing=timing[s])
        (out / a.dataset / f"{s}.stats.json").write_text(json.dumps(stats, indent=1))
        log.info("scene %s done: objects %d, assoc %.1f s", s, len(pred.objects), wall)
        del c, dz, objs


if __name__ == "__main__":
    main()
