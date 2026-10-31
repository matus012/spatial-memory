"""Replica GT: objects, AABBs, geometric relations (docs/RELATIONS.md), per-vertex semantic labels.

Sources (data/MANIFEST.md):
- data/replica_v1/<room_0>/habitat/{mesh_semantic.ply, info_semantic.json}  — object instances
- data/nice/Replica/<room0>/{results/, traj.txt}, data/nice/Replica/<room0>_mesh.ply — sequences, protocol mesh
- data/ovo_replica_gt/<room0>.txt  — protocol per-vertex labels (OVO @ e0d16f7), original Replica class ids
- third_party/ovo_replica_eval_info.yaml — 51-class protocol

Run: uv run python -m spatialmem.gt.replica --all   (writes gt/<scene>.json)
     uv run python -m spatialmem.gt.replica --dev   (Replica-dev: also gt/<scene>.sem.npz, per-vertex GT from mesh_semantic)
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from plyfile import PlyData
from scipy.spatial import cKDTree

from spatialmem.eval.contract import STRUCTURAL, GroundTruth, PredObject
from spatialmem.gt.relations import compute_relations

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SCENES = ("office0", "office1", "office2", "office3", "office4", "room0", "room1", "room2")
# Replica-dev (task 006): the v1 scenes outside the protocol's 8, named by their v1 directory (configs/replica_dev.yaml)
DEV_SCENES = tuple(yaml.safe_load((ROOT / "configs" / "replica_dev.yaml").read_text(encoding="utf-8"))["scenes"])


def v1_name(scene: str) -> str:
    """room0 -> room_0 (Replica v1 directory naming); dev scenes already carry it (apartment_0)."""
    return scene if scene[-2] == "_" else scene[:-1] + "_" + scene[-1]


@dataclass(frozen=True)
class Protocol:
    names: tuple[str, ...]          # 51 reduced names, index = reduced id
    to_reduced: dict[int, int]      # original Replica class id -> reduced id; unmapped/ignored -> -1
    num_classes: int

    def reduce(self, orig: np.ndarray) -> np.ndarray:
        lut = np.full(max(max(self.to_reduced), int(orig.max())) + 3, -1, dtype=np.int64)
        for k, v in self.to_reduced.items():
            if k >= 0:
                lut[k] = v
        out = np.where(orig >= 0, lut[np.clip(orig, 0, None)], -1)
        return out


@lru_cache(maxsize=1)
def protocol() -> Protocol:
    info = yaml.safe_load((ROOT / "third_party" / "ovo_replica_eval_info.yaml").read_text(encoding="utf-8"))
    n = int(info["num_classes"])
    ignore = set(info["ignore"])
    m = {int(k): (int(v) if int(v) not in ignore else -1) for k, v in info["map_to_reduced"].items()}
    names = tuple(info["class_names_reduced"])
    assert len(names) == n == 51
    return Protocol(names, m, n)


# ---------------------------------------------------------------- loading
def read_semantic_mesh(scene: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """vertices (V, 3) float64, faces (F, k) int64 (quads in Replica), face object ids (F,)."""
    ply = PlyData.read(str(DATA / "replica_v1" / v1_name(scene) / "habitat" / "mesh_semantic.ply"))
    v = ply["vertex"]
    verts = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float64)
    f = ply["face"]
    vi = f["vertex_indices"]
    lens = np.fromiter((len(x) for x in vi), dtype=np.int64, count=len(vi))
    if (lens == lens[0]).all():
        faces = np.vstack(vi).astype(np.int64)
    else:  # variable polygons: keep as object array
        faces = vi
    return verts, faces, np.asarray(f["object_id"], dtype=np.int64)


def read_info(scene: str) -> dict:
    return json.loads((DATA / "replica_v1" / v1_name(scene) / "habitat" / "info_semantic.json").read_text())


def read_protocol_mesh(scene: str) -> np.ndarray:
    ply = PlyData.read(str(DATA / "nice" / "Replica" / f"{scene}_mesh.ply"))
    v = ply["vertex"]
    return np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float64)


def read_protocol_labels(scene: str) -> np.ndarray:
    """Original Replica class id per protocol-mesh vertex (OVO GT txt)."""
    return np.loadtxt(DATA / "ovo_replica_gt" / f"{scene}.txt", dtype=np.int64)


# ---------------------------------------------------------------- extraction
def object_classes(info: dict) -> tuple[dict[int, int], dict[int, str]]:
    obj_cls = {int(o["id"]): int(o["class_id"]) for o in info["objects"]}
    cls_name = {int(c["id"]): str(c["name"]) for c in info["classes"]}
    return obj_cls, cls_name


def object_vertex_sets(faces: np.ndarray, face_obj: np.ndarray) -> dict[int, np.ndarray]:
    """object id -> unique vertex indices of its faces."""
    order = np.argsort(face_obj, kind="stable")
    fo = face_obj[order]
    starts = np.flatnonzero(np.r_[True, fo[1:] != fo[:-1]])
    ends = np.r_[starts[1:], len(fo)]
    out = {}
    for s, e in zip(starts, ends):
        idx = order[s:e]
        fv = faces[idx] if isinstance(faces, np.ndarray) and faces.dtype != object else np.concatenate(list(faces[idx]))
        out[int(fo[s])] = np.unique(np.asarray(fv).ravel())
    return out


def up_axis(verts: np.ndarray, vsets: dict[int, np.ndarray], obj_cls: dict[int, int], cls_name: dict[int, str]) -> int:
    """Axis along which ceiling vertices sit highest above floor vertices relative to their spread (|median
    separation| / (std floor + std ceiling)); asserts the sign is +. The spread normalisation (task 006) is needed
    where the ceiling object covers only part of the floor plan (Replica FRL apartments: horizontal offset of the
    medians > storey height); the protocol 8 give the same axis either way (gt/ regenerated byte-identical)."""
    def pts(name):
        ids = [o for o, c in obj_cls.items() if cls_name.get(c) == name and o in vsets]
        return verts[np.concatenate([vsets[o] for o in ids])] if ids else None
    fl, ce = pts("floor"), pts("ceiling")
    if fl is None or ce is None:
        raise ValueError("need floor and ceiling objects to determine the up axis")
    sep = np.median(ce, 0) - np.median(fl, 0)
    ax = int(np.argmax(np.abs(sep) / (fl.std(0) + ce.std(0) + 1e-9)))
    if sep[ax] <= 0:
        raise ValueError(f"ceiling below floor along axis {ax}: frame is not +up ({sep})")
    return ax


def vertex_classes(n_vertices: int, faces: np.ndarray, face_obj: np.ndarray, obj_cls: dict[int, int]) -> tuple[np.ndarray, int]:
    """Original Replica class id per mesh vertex = the most frequent class among its incident faces (ties ->
    smallest id); faces of objects without a class entry count as -1; vertices without a face get -1.
    Returns (labels, number of vertices whose incident faces disagree)."""
    if not (isinstance(faces, np.ndarray) and faces.dtype != object):
        raise ValueError("variable-length polygons: not needed for Replica (quads)")
    uo = np.unique(face_obj)
    cls_face = np.array([obj_cls.get(int(o), -1) for o in uo])[np.searchsorted(uo, face_obj)]
    vi = faces.ravel()
    ci = np.repeat(cls_face, faces.shape[1])
    pairs, cnt = np.unique(np.stack([vi, ci], 1), axis=0, return_counts=True)
    order = np.lexsort((pairs[:, 1], -cnt, pairs[:, 0]))      # by vertex, then count desc, then class asc
    pairs = pairs[order]
    first = np.r_[True, pairs[1:, 0] != pairs[:-1, 0]]
    out = np.full(n_vertices, -1, dtype=np.int64)
    out[pairs[first, 0]] = pairs[first, 1]
    n_conflict = int((np.bincount(pairs[:, 0], minlength=n_vertices) > 1).sum())
    return out, n_conflict


def dev_vertex_gt(scene: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Replica-dev per-vertex GT: mesh_semantic vertices + reduced 51-class labels (-1 = ignored / unmapped)."""
    verts, faces, face_obj = read_semantic_mesh(scene)
    obj_cls, _ = object_classes(read_info(scene))
    orig, n_conflict = vertex_classes(len(verts), faces, face_obj, obj_cls)
    return verts, protocol().reduce(orig), n_conflict


def derivation_agreement(scene: str) -> dict:
    """Sanity proof of the dev derivation on a protocol scene: mesh_semantic-derived per-vertex labels vs the
    protocol's OVO txt, index by index (NICE mesh vertices == mesh_semantic vertices, asserted)."""
    verts, red, n_conflict = dev_vertex_gt(scene)
    pv = read_protocol_mesh(scene)
    assert pv.shape == verts.shape and np.array_equal(pv, verts), scene
    gt = protocol().reduce(read_protocol_labels(scene))
    lab = gt >= 0
    return {"n_vertices": int(len(verts)), "conflict_vertices": n_conflict,
            "agree_reduced_on_protocol_labelled": float(np.mean(red[lab] == gt[lab])),
            "agree_reduced_all": float(np.mean(red == gt)),
            "labelled_protocol": int(lab.sum()), "labelled_derived": int((red >= 0).sum())}


def build_gt(scene: str, with_vertices: bool = True) -> tuple[GroundTruth, dict]:
    proto = protocol()
    verts, faces, face_obj = read_semantic_mesh(scene)
    info = read_info(scene)
    obj_cls, cls_name = object_classes(info)
    vsets = object_vertex_sets(faces, face_obj)
    up = up_axis(verts, vsets, obj_cls, cls_name)

    objects, meta, skipped = [], [], {"ignored_or_unmapped": 0, "structural": 0, "missing_class": 0}
    name_mismatch = []
    for oid, vidx in sorted(vsets.items()):
        if oid not in obj_cls:
            skipped["missing_class"] += 1
            continue
        cid = obj_cls[oid]
        red = proto.to_reduced.get(cid, -1)
        name = cls_name.get(cid, "undefined")
        p = verts[vidx]
        rec = {"id": oid, "class_id": cid, "class_name": name, "n_vertices": int(len(vidx)),
               "aabb_min": p.min(0).tolist(), "aabb_max": p.max(0).tolist(),
               "label": proto.names[red] if red >= 0 else None}
        meta.append(rec)
        if red < 0:
            skipped["ignored_or_unmapped"] += 1  # protocol ignore (e.g. Replica "undefined") or absent from map_to_reduced
            continue
        if name != proto.names[red]:
            name_mismatch.append((oid, cid, name, proto.names[red]))
        if proto.names[red] in STRUCTURAL:
            skipped["structural"] += 1
            continue
        objects.append(PredObject(oid, proto.names[red], p.min(0), p.max(0)))
    relations = compute_relations(objects, up=up)

    gt = GroundTruth(scene=scene, objects=objects, relations=relations)
    if with_vertices and scene in DEV_SCENES:
        gt.vertices, gt.vertex_labels, _ = dev_vertex_gt(scene)
    elif with_vertices:
        gt.vertices = read_protocol_mesh(scene)
        gt.vertex_labels = proto.reduce(read_protocol_labels(scene))
        assert len(gt.vertices) == len(gt.vertex_labels), (len(gt.vertices), len(gt.vertex_labels))
    extra = {"up_axis": up, "n_semantic_objects": len(vsets), "skipped": skipped,
             "class_name_mismatch": name_mismatch, "all_objects": meta}
    return gt, extra


def write_gt(scene: str, out_dir: Path = ROOT / "gt") -> Path:
    gt, extra = build_gt(scene, with_vertices=False)
    out_dir.mkdir(exist_ok=True)
    d = gt.to_json()
    d.update(extra)
    d["protocol"] = "OVO Replica 51-class @ tberriel/OVO e0d16f7 (third_party/ovo_replica_eval_info.yaml)"
    d["relation_rules"] = "docs/RELATIONS.md (2026-10-27)"
    if scene in DEV_SCENES:   # per-vertex GT derived from mesh_semantic (configs/replica_dev.yaml gt_rule)
        verts, red, n_conflict = dev_vertex_gt(scene)
        np.savez_compressed(out_dir / f"{scene}.sem.npz", vertices=verts, labels=red)
        d["vertex_gt"] = {"source": "mesh_semantic.ply face object_id -> info_semantic class, majority of incident faces",
                          "n_vertices": int(len(verts)), "labelled": int((red >= 0).sum()), "conflict_vertices": n_conflict,
                          "classes_present": sorted(protocol().names[i] for i in np.unique(red[red >= 0]))}
    path = out_dir / f"{scene}.json"
    path.write_text(json.dumps(d, indent=1))
    return path


def load_gt(scene: str, with_vertices: bool = True) -> GroundTruth:
    """GT from gt/<scene>.json (objects + relations) plus protocol vertices/labels from data/ (dev: gt/<scene>.sem.npz)."""
    d = json.loads((ROOT / "gt" / f"{scene}.json").read_text())
    base = GroundTruth.from_json(d)
    gt = GroundTruth(scene=base.scene, objects=base.objects, relations=base.relations)
    if with_vertices and scene in DEV_SCENES:
        z = np.load(ROOT / "gt" / f"{scene}.sem.npz")
        gt.vertices, gt.vertex_labels = z["vertices"], z["labels"]
    elif with_vertices:
        gt.vertices = read_protocol_mesh(scene)
        gt.vertex_labels = protocol().reduce(read_protocol_labels(scene))
        assert len(gt.vertices) == len(gt.vertex_labels), (scene, len(gt.vertices), len(gt.vertex_labels))
    return gt


def protocol_crosscheck(scene: str) -> dict:
    """Independent check of the OVO per-vertex GT: (a) ids outside map_to_reduced (OVO would index row -1
    for them), (b) frame + label agreement of each protocol vertex with the nearest semantic-mesh face."""
    proto = protocol()
    pv, pl = read_protocol_mesh(scene), read_protocol_labels(scene)
    verts, faces, face_obj = read_semantic_mesh(scene)
    obj_cls, _ = object_classes(read_info(scene))
    ids, cnt = np.unique(pl, return_counts=True)
    unmapped = {int(i): int(c) for i, c in zip(ids, cnt) if int(i) not in proto.to_reduced}
    cen = verts[faces].mean(axis=1) if isinstance(faces, np.ndarray) and faces.dtype != object else         np.stack([verts[f].mean(0) for f in faces])
    dist, nn = cKDTree(cen).query(pv, k=1)
    sem_cls = np.array([obj_cls.get(int(o), 0) for o in face_obj])[nn]
    lab = pl >= 0
    return {"n_vertices": int(len(pv)), "unmapped_ids": unmapped,
            "nn_dist_cm_median": float(np.median(dist) * 100), "nn_dist_cm_p99": float(np.percentile(dist, 99) * 100),
            "label_agreement": float(np.mean(sem_cls[lab] == pl[lab])),
            "label_agreement_reduced": float(np.mean(proto.reduce(sem_cls[lab]) == proto.reduce(pl[lab])))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--scenes", nargs="*", default=[])
    ap.add_argument("--dev", action="store_true", help="the Replica-dev scenes (configs/replica_dev.yaml)")
    ap.add_argument("--crosscheck", action="store_true", help="also check OVO GT txt vs mesh_semantic")
    ap.add_argument("--agreement", nargs="*", default=None,
                    help="protocol scenes: dev per-vertex derivation vs OVO txt (sanity proof), then exit")
    a = ap.parse_args()
    if a.agreement is not None:
        for s in a.agreement:
            print(f"{s}: derivation agreement {json.dumps(derivation_agreement(s))}")
        return
    for s in (SCENES if a.all else DEV_SCENES if a.dev else a.scenes):
        if a.crosscheck:
            print(f"{s}: crosscheck {json.dumps(protocol_crosscheck(s))}")
        p = write_gt(s)
        d = json.loads(p.read_text())
        rel = {k: sum(r["pred"] == k for r in d["relations"]) for k in ("on", "in", "next_to")}
        print(f"{s}: up={d['up_axis']} objects={len(d['objects'])} (of {d['n_semantic_objects']}, "
              f"skipped {d['skipped']}, name mismatches {len(d['class_name_mismatch'])}) relations={rel} "
              f"{'vertex_gt ' + json.dumps({k: v for k, v in d['vertex_gt'].items() if k != 'source'}) + ' ' if 'vertex_gt' in d else ''}"
              f"-> {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
