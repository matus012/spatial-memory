"""P1a demo: one HSSD scene -> 3D map + scene-graph viewer (one offline HTML file) and the 007 figure.

Inputs (under the data dir, `<repo>/data` or $SPATIALMEM_DATA; this module only):
  baselines/cg/hssd/<scene>.json                  ConceptGraphs map as a CONTRACT Prediction
  baselines/cg/runs/hssd/<scene>/objects.npz      CG per-object points (points, counts, ...): object i = pred.objects[i]
  hssd_runs/all/<scene>/gt.json                   HSSD ground truth
Output: <data>/demo/<scene>.html (git-ignored, embeds dataset points) [+ reports/figs/007_fig1_house_map_graph.png with --fig].

Run: uv run python -m spatialmem.demo [--scene 102816627] [--fig]
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import random
from pathlib import Path

import numpy as np

from spatialmem._demo_html import TEMPLATE
from spatialmem.eval.contract import STRUCTURAL, GroundTruth, Prediction
from spatialmem.eval.objects import ObjectScore, score_objects, score_relations
from spatialmem.hssd.gt import load_hssd_gt
from spatialmem.hssd.vocab import canonical

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = "102816627"
TAU = 0.25
POINT_BUDGET = 400_000   # ~2.7 MB as base64 int16 xyz
SEED = 114


def data_dir() -> Path:
    return Path(os.environ.get("SPATIALMEM_DATA") or REPO / "data")


def load(scene: str, data: Path | None = None) -> tuple[Prediction, GroundTruth, ObjectScore, Path]:
    """(map prediction with canonical labels, GT, object score @ TAU, objects.npz path)."""
    data = data or data_dir()
    pred = Prediction.from_json(data / "baselines" / "cg" / "hssd" / f"{scene}.json")
    for o in pred.objects:
        o.label = canonical(o.label)
    gt, _ = load_hssd_gt(data / "hssd_runs" / "all" / scene / "gt.json")
    return pred, gt, score_objects(pred, gt, TAU), data / "baselines" / "cg" / "runs" / "hssd" / scene / "objects.npz"


def statuses(pred: Prediction, gt: GroundTruth, sc: ObjectScore) -> tuple[dict[int, tuple[str, int | None, float]], dict[int, tuple[str, int | None, float]]]:
    """Per object id: (m matched | x extra | u missed | i structural-ignored, matched partner id, IoU)."""
    p2g = {p: (g, iou) for p, g, iou in sc.matches}
    g2p = {g: (p, iou) for p, g, iou in sc.matches}
    ps = {o.id: ("i", None, 0.0) if o.label in STRUCTURAL else ("m", *p2g[o.id]) if o.id in p2g else ("x", None, 0.0) for o in pred.objects}
    gs = {o.id: ("i", None, 0.0) if o.label in STRUCTURAL else ("m", *g2p[o.id]) if o.id in g2p else ("u", None, 0.0) for o in gt.objects}
    return ps, gs


def subsample(npz: Path, n_objects: int, budget: int = POINT_BUDGET) -> tuple[np.ndarray, np.ndarray]:
    """(points (P, 3) float32 grouped by object, offsets (n_objects + 1,)): <= budget points, every object keeps >= 25."""
    z = np.load(npz)
    pts, counts = z["points"], z["counts"]
    assert len(counts) == n_objects, f"{npz}: {len(counts)} objects != prediction's {n_objects}"
    frac = min(1.0, budget / max(int(counts.sum()), 1))
    rng = np.random.default_rng(SEED)
    out, off, start = [], [0], 0
    for c in counts:
        k = min(int(c), max(int(c * frac), 25))
        out.append(pts[start + np.sort(rng.choice(int(c), k, replace=False))] if k < c else pts[start:start + int(c)])
        off.append(off[-1] + k)
        start += int(c)
    return (np.concatenate(out) if out else np.zeros((0, 3), np.float32)), np.array(off)


def encode_points(pts: np.ndarray) -> dict:
    lo, hi = pts.min(0).astype(np.float64), pts.max(0).astype(np.float64)
    sc = np.maximum(hi - lo, 1e-6) / 65535.0
    q = (np.round((pts - lo) / sc) - 32768).astype(np.int16)
    return {"b64": base64.b64encode(q.astype("<i2").tobytes()).decode(), "lo": lo.tolist(), "sc": sc.tolist()}


def _nan(v: float) -> float | None:
    return None if math.isnan(v) else float(v)


def payload(scene: str, data: Path | None = None) -> dict:
    pred, gt, sc, npz = load(scene, data)
    assert [o.id for o in pred.objects] == list(range(len(pred.objects))), "object ids must be indices into objects.npz"
    ps, gs = statuses(pred, gt, sc)
    pts, off = subsample(npz, len(pred.objects))
    obj = lambda o, st: {"id": o.id, "label": o.label, "min": o.aabb_min.round(4).tolist(), "max": o.aabb_max.round(4).tolist(),
                         "st": st[o.id][0], "mate": st[o.id][1], "iou": st[o.id][2]}
    rel = score_relations(pred, gt, sc)
    return {"scene": scene, "tau": TAU, "pred": [obj(o, ps) for o in pred.objects], "gt": [obj(o, gs) for o in gt.objects],
            "prel": [[r.subj, r.obj, r.pred] for r in pred.relations], "grel": [[r.subj, r.obj, r.pred] for r in gt.relations],
            "score": {"n_pred": sc.n_pred, "n_gt": sc.n_gt, "tp": sc.tp, "p": sc.precision, "r": sc.recall, "f1": sc.f1, "label_acc": _nan(sc.label_acc)},
            "rel": {"p": rel.precision, "r": rel.recall, "f1": rel.f1},
            "n_pred_total": len(pred.objects), "n_pred": sc.n_pred, "n_gt": len(gt.objects),
            "n_points_total": int(np.load(npz)["counts"].sum()), "pts": {**encode_points(pts), "off": off.tolist()}}


def build(scene: str, data: Path | None = None) -> Path:
    data = data or data_dir()
    blob = json.dumps(payload(scene, data), separators=(",", ":")).replace("</", "<\\/")
    out = data / "demo" / f"{scene}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(TEMPLATE.replace("__DATA__", blob), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=DEFAULT_SCENE, help="HSSD dev scene (configs/hssd_scenes.yaml)")
    ap.add_argument("--fig", action="store_true", help="also write reports/figs/007_fig1_house_map_graph.png")
    a = ap.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    out = build(a.scene)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    if a.fig:
        from spatialmem.viz import house_map_graph
        print("wrote", house_map_graph(a.scene, data_dir()))


if __name__ == "__main__":
    main()
