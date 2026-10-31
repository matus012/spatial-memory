"""ConceptGraphs baseline (task 003): CG's native map -> CONTRACT `Prediction`.

Three steps, two environments:
- `export`  (inside hpc/cg.def): CG's `full_pcd_none_<suffix>_post.pkl.gz` -> objects.npz (points, counts, clip_ft,
  num_detections). The pickle holds omegaconf / open3d-era objects, so it is only read inside the image.
- `text`    (inside hpc/cg.def): CG's own CLIP (open_clip ViT-H-14 laion2b_s32b_b79k) text embeddings of
  "a photo of a {c}." over the scene-set vocabulary -> text_<dataset>.npz (classes, emb).
- `predict` (lpc, project venv): objects.npz + text npz -> Prediction JSON per scene (docs/CONTRACT.md 2026-10-27e:
  label = argmax cosine of the object's own clip_ft vs the text embeddings; relations = compute_relations).

Run: uv run python -m spatialmem.baselines.cg predict <replica|hssd> <runs_dir> <text.npz> <out_dir>
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from spatialmem.eval.boxes import from_points
from spatialmem.eval.contract import STRUCTURAL, Prediction, PredObject
from spatialmem.gt.relations import compute_relations

ROOT = Path(__file__).resolve().parents[2]
UP = {"replica": 2, "hssd": 1}
PROMPT = "a photo of a {}."


def vocabulary(dataset: str) -> list[str]:
    if dataset == "replica":   # = spatialmem.gt.replica.protocol().names (asserted in `predict`; no plyfile in the image)
        import yaml
        info = yaml.safe_load((ROOT / "third_party" / "ovo_replica_eval_info.yaml").read_text(encoding="utf-8"))
        return list(info["class_names_reduced"])
    from spatialmem.hssd.vocab import vocabulary as hssd_vocabulary
    return hssd_vocabulary()


def prompt(c: str) -> str:
    return PROMPT.format(c.replace("_", " ").replace("-", " "))


def export(pkl_gz: Path, out: Path) -> None:
    import gzip
    import pickle
    with gzip.open(pkl_gz, "rb") as f:
        objs = pickle.load(f)["objects"]
    pts = [np.asarray(o["pcd_np"], dtype=np.float32) for o in objs]
    np.savez_compressed(
        out, points=np.concatenate(pts) if pts else np.zeros((0, 3), np.float32),
        counts=np.array([len(p) for p in pts], dtype=np.int64),
        clip_ft=np.stack([np.asarray(o["clip_ft"], dtype=np.float32).reshape(-1) for o in objs]),
        num_detections=np.array([o["num_detections"] for o in objs], dtype=np.int64))
    print(f"exported {len(objs)} objects, {sum(map(len, pts))} points -> {out}")


def text(dataset: str, out: Path) -> None:
    import open_clip
    import torch
    classes = vocabulary(dataset)
    prompts = [prompt(c) for c in classes]   # CONTRACT 2026-10-27e (2): `_` / `-` in {c} -> spaces
    model, _, _ = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
    model = model.to("cuda").eval()
    tok = open_clip.get_tokenizer("ViT-H-14")
    with torch.no_grad():
        emb = model.encode_text(tok(prompts).to("cuda")).float()
    emb = torch.nn.functional.normalize(emb, dim=-1).cpu().numpy()
    np.savez(out, classes=np.array(classes), prompts=np.array(prompts), emb=emb, prompt=np.array(PROMPT))
    print(f"text {dataset}: {len(classes)} classes, emb {emb.shape} on cuda -> {out}; e.g. {prompts[:3]}")


def label_objects(clip_ft: np.ndarray, emb: np.ndarray) -> np.ndarray:
    """Index of the argmax cosine class per object."""
    f = clip_ft / np.linalg.norm(clip_ft, axis=1, keepdims=True)
    e = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return np.argmax(f @ e.T, axis=1)


def prediction(scene: str, objects_npz: Path, text_npz: Path, up: int, semantic: bool) -> Prediction:
    o, t = np.load(objects_npz), np.load(text_npz)
    classes = [str(c) for c in t["classes"]]
    lab = label_objects(o["clip_ft"], t["emb"]) if len(o["counts"]) else np.zeros(0, int)
    pts = np.split(o["points"].astype(np.float64), np.cumsum(o["counts"])[:-1])
    objs = [PredObject(i, classes[lab[i]], *from_points(p), score=float(o["num_detections"][i]))
            for i, p in enumerate(pts)]
    rels = compute_relations([x for x in objs if x.label not in STRUCTURAL], up=up)
    sem_p = sem_l = None
    if semantic:
        sem_p = o["points"].astype(np.float64)
        sem_l = np.repeat(lab, o["counts"]).astype(np.int64)
    return Prediction(scene, objs, rels, sem_p, sem_l)


def gt_boxes(dataset: str, scene: str) -> np.ndarray:
    p = ROOT / "gt" / f"{scene}.json" if dataset == "replica" else ROOT / "data" / "hssd_runs" / "all" / scene / "gt.json"
    d = json.loads(p.read_text())
    return np.array([[o["aabb_min"], o["aabb_max"]] for o in d["objects"]])


def frame_check(pred: Prediction, gt: np.ndarray) -> tuple[float, float]:
    """(median distance of object centroids to the nearest GT box [m], fraction of centroids inside the GT extent)."""
    b = pred.boxes()
    c = b.mean(axis=1)
    gap = np.maximum(gt[None, :, 0] - c[:, None], c[:, None] - gt[None, :, 1])
    d = np.linalg.norm(np.clip(gap, 0, None), axis=-1).min(axis=1)
    lo, hi = gt[:, 0].min(axis=0), gt[:, 1].max(axis=0)
    return float(np.median(d)), float(np.mean(np.all((c >= lo) & (c <= hi), axis=1)))


def predict(dataset: str, runs: Path, text_npz: Path, out: Path) -> None:
    classes = [str(c) for c in np.load(text_npz)["classes"]]
    if dataset == "replica":
        from spatialmem.gt.replica import protocol
        assert classes == list(protocol().names), "text npz class list != Replica protocol names"
    else:
        from spatialmem.hssd.vocab import vocabulary as hssd_vocabulary
        assert classes == hssd_vocabulary(), "text npz class list != HSSD canonical vocabulary"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for npz in sorted(runs.glob("*/objects.npz")):
        s = npz.parent.name
        pred = prediction(s, npz, text_npz, UP[dataset], semantic=dataset == "replica")
        pred.to_json(out / f"{s}.json")
        Prediction.from_json(out / f"{s}.json")   # schema round-trip
        med, inside = frame_check(pred, gt_boxes(dataset, s))
        rows.append({"scene": s, "objects": len(pred.objects), "relations": len(pred.relations),
                     "median_centroid_to_gt_box_m": round(med, 4), "frac_centroids_in_gt_extent": round(inside, 4)})
        print(rows[-1])
    with open(out / "_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def runtime(runs: Path, out: Path) -> None:
    """runtime.csv from each scene's time_det.tsv / time_map.tsv (phase, start, end, peak MiB or NA, concurrent scenes,
    job[, note]) and DONE ("<frames> <frames used> <job>"), as written by hpc/slurm/cg_run_scene.sh."""
    rows = []
    for done in sorted(runs.glob("*/*/DONE")):
        w = done.parent
        nf, used, _ = done.read_text().split()
        ph = {n: (w / f"time_{n}.tsv").read_text().split("\t") for n in ("det", "map")}
        wall = {n: int(v[2]) - int(v[1]) for n, v in ph.items()}
        mib = [v[3] for v in ph.values()]
        rows.append({"dataset": w.parent.name, "scene": w.name, "frames": int(nf), "frames_used": int(used),
                     "det_wall_s": wall["det"], "map_wall_s": wall["map"],
                     "s_per_frame": round((wall["det"] + wall["map"]) / int(used), 3),
                     "peak_vram_gb": "NA" if "NA" in mib else round(max(map(int, mib)) / 1024, 2),
                     "det_concurrent": int(ph["det"][4]), "map_concurrent": int(ph["map"][4]),
                     "job_id": "/".join(dict.fromkeys(v[5].strip() for v in ph.values())),
                     "note": " ".join(v[6].strip() for v in ph.values() if len(v) > 6)})
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"runtime: {len(rows)} scenes -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("export"); a.add_argument("pkl_gz", type=Path); a.add_argument("out", type=Path)
    a = sp.add_parser("text"); a.add_argument("dataset", choices=list(UP)); a.add_argument("out", type=Path)
    a = sp.add_parser("predict"); a.add_argument("dataset", choices=list(UP))
    a.add_argument("runs", type=Path); a.add_argument("text_npz", type=Path); a.add_argument("out", type=Path)
    a = sp.add_parser("runtime"); a.add_argument("runs", type=Path); a.add_argument("out", type=Path)
    args = ap.parse_args()
    if args.cmd == "export":
        export(args.pkl_gz, args.out)
    elif args.cmd == "text":
        text(args.dataset, args.out)
    elif args.cmd == "runtime":
        runtime(args.runs, args.out)
    else:
        predict(args.dataset, args.runs, args.text_npz, args.out)


if __name__ == "__main__":
    main()
