"""HSSD ground truth for scoring: gt.json (spatialmem.hssd.run) -> GroundTruth with canonical labels
(docs/CONTRACT.md 2026-10-27e). Relations are taken as exported (computed on raw labels; equal on canonical ones)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spatialmem.eval.contract import GroundTruth, PredObject, PredRelation
from spatialmem.hssd.vocab import canonical


def load_hssd_gt(gt_json: Path) -> tuple[GroundTruth, int]:
    """(GT, up axis)."""
    d = json.loads(Path(gt_json).read_text())
    objs = [PredObject(o["id"], canonical(o["label"]), np.array(o["aabb_min"]), np.array(o["aabb_max"])) for o in d["objects"]]
    rels = [PredRelation(r["subj"], r["pred"], r["obj"]) for r in d["relations"]]
    return GroundTruth(scene=d["scene"], objects=objs, relations=rels), int(d["up_axis"])
