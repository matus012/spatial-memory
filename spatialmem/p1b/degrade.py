"""Map degradation axes (P1B_SWEEP §2). Applied to the GT Prediction (canonical labels, AABBs) of one house.

`level` is the string printed in the spec table: "10" "25" "50" "75" (percent; drop / label / extra / relations, extra also
"100") or "0.10" "0.25" "0.50" "1.00" (position sigma, m); "0" / "0.00" = no degradation. It is used verbatim in the RNG stream
`random.Random(f"009:{axis}:{level}:{seed}:{scene}")`.
"""
from __future__ import annotations

import random

import numpy as np

from spatialmem.eval.contract import PredObject, PredRelation, Prediction
from spatialmem.eval.objects import without_structural
from spatialmem.gt.relations import compute_relations

AXES = ("drop", "label", "position", "extra", "relations")
LEVELS = {"drop": ("10", "25", "50", "75"), "label": ("10", "25", "50", "75"), "position": ("0.10", "0.25", "0.50", "1.00"),
          "extra": ("10", "25", "50", "100"), "relations": ("25", "50", "75", "100")}
SEEDS = (0, 1, 2)
UP = 1
REAL = ("cg", "ovo", "ours_v2")


def degraded_names() -> list[str]:
    return [f"{a}_{lv}_s{s}" for a in AXES for lv in LEVELS[a] for s in SEEDS]


def parse_name(name: str) -> tuple[str, str, int]:
    """'position_0.25_s1' -> ('position', '0.25', 1)."""
    axis, lv, s = name.rsplit("_", 2)
    return axis, lv, int(s[1:])


def _copy(o: PredObject, label: str | None = None, shift: np.ndarray | None = None) -> PredObject:
    d = np.zeros(3) if shift is None else shift
    return PredObject(o.id, o.label if label is None else label, o.aabb_min + d, o.aabb_max + d, o.score)


def _rkey(r: PredRelation) -> tuple[int, str, int]:
    return (r.subj, r.pred, r.obj)


def degrade(gt: Prediction, axis: str, level: str, seed: int, scene: str, vocab: list[str]) -> Prediction:
    """Degraded copy of `gt` (STRUCTURAL objects and their relations dropped first; N, M count what is left)."""
    g = without_structural(gt)
    objs = sorted(g.objects, key=lambda o: o.id)
    rels = sorted(g.relations, key=_rkey)
    n, m = len(objs), len(rels)
    rng = random.Random(f"009:{axis}:{level}:{seed}:{scene}")
    x = float(level)
    if axis == "drop":
        gone = {o.id for o in rng.sample(objs, round(x / 100 * n))}
        out = [_copy(o) for o in objs if o.id not in gone]
        rl = [r for r in rels if r.subj not in gone and r.obj not in gone]
    elif axis == "label":
        pick = rng.sample(objs, round(x / 100 * n))
        new = {o.id: rng.choice(sorted(set(vocab) - {o.label})) for o in pick}   # draws in sample order
        out = [_copy(o, new.get(o.id)) for o in objs]
        rl = list(rels)
    elif axis == "position":
        out = []
        for o in objs:
            out.append(_copy(o, shift=np.array([rng.gauss(0, x), rng.gauss(0, x), rng.gauss(0, x)])))
        rl = compute_relations(out, up=UP)
    elif axis == "extra":
        lo = np.min([o.aabb_min for o in objs], axis=0)
        hi = np.max([o.aabb_max for o in objs], axis=0)
        h = [k for k in range(3) if k != UP]
        out = [_copy(o) for o in objs]
        top = max(o.id for o in objs)
        for k in range(round(x / 100 * n)):
            label = rng.choice(objs).label
            size = rng.choice([o for o in objs if o.label == label])
            size = size.aabb_max - size.aabb_min
            y = rng.choice(objs)
            y = (y.aabb_min[UP] + y.aabb_max[UP]) / 2
            c = np.zeros(3)
            c[UP] = y
            for a in h:
                c[a] = rng.uniform(lo[a], hi[a])
            out.append(PredObject(top + 1 + k, label, c - size / 2, c + size / 2))
        rl = compute_relations(out, up=UP)
    elif axis == "relations":
        gone = {_rkey(r) for r in rng.sample(rels, round(x / 100 * m))}
        out = [_copy(o) for o in objs]
        rl = [r for r in rels if _rkey(r) not in gone]
    else:
        raise ValueError(axis)
    return Prediction(scene, out, [PredRelation(r.subj, r.pred, r.obj) for r in rl])
