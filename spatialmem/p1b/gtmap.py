"""The map a planner reads (P1B_PROTOCOL §1): objects (id, label, centre, size) + relations, structural labels removed.
GT maps and real mapper outputs (spatialmem.eval.contract.Prediction) share `from_prediction`."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from spatialmem.eval.contract import STRUCTURAL, Prediction
from spatialmem.hssd.gt import load_hssd_gt
from spatialmem.hssd.vocab import ROOT

RUNS = ROOT / "data" / "hssd_runs" / "all"


@dataclass
class MapObj:
    id: int
    label: str
    center: np.ndarray
    size: np.ndarray


@dataclass
class Map:
    scene: str
    objects: list[MapObj]
    relations: list[tuple[int, str, int]] = field(default_factory=list)

    def with_label(self, label: str) -> list[MapObj]:
        return [o for o in self.objects if o.label == label]

    def obj(self, i: int) -> MapObj:
        return next(o for o in self.objects if o.id == i)

    def related(self, a: int, pred: str) -> list[int]:
        """Ids b with (a pred b); next_to is unordered (either exported direction)."""
        out = [o for s, p, o in self.relations if s == a and p == pred]
        if pred == "next_to":
            out += [s for s, p, o in self.relations if o == a and p == pred]
        return out


def from_prediction(p: Prediction) -> Map:
    """Drop STRUCTURAL objects and every relation that touches one (or a missing id)."""
    objs = [MapObj(o.id, o.label, (o.aabb_min + o.aabb_max) / 2, o.aabb_max - o.aabb_min)
            for o in p.objects if o.label not in STRUCTURAL]
    keep = {o.id for o in objs}
    rels = [(r.subj, r.pred, r.obj) for r in p.relations if r.subj in keep and r.obj in keep]
    return Map(p.scene, sorted(objs, key=lambda o: o.id), sorted(rels))


def eval_scenes() -> list[str]:
    return yaml.safe_load((ROOT / "configs" / "hssd_scenes.yaml").read_text())["eval"]


def load(scene: str) -> Map:
    gt, up = load_hssd_gt(RUNS / scene / "gt.json")
    assert up == 1, f"{scene}: up axis {up}"
    return from_prediction(gt)


def vocabulary(maps: list[Map]) -> list[str]:
    """Eval-30 label vocabulary: union over the GT maps, sorted."""
    return sorted({o.label for m in maps for o in m.objects})
