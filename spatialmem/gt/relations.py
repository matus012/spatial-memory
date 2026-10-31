"""Geometric relations from AABBs — the rules frozen in docs/RELATIONS.md. Change both together."""
from __future__ import annotations

import numpy as np

from spatialmem.eval.boxes import intersection_volume, volume
from spatialmem.eval.contract import PredObject, PredRelation

ON_GAP_MIN = -0.08     # m, bottom(A) - top(B)
ON_GAP_MAX = 0.05      # m
ON_FOOTPRINT = 0.5     # fraction of footprint(A) over footprint(B)
IN_CONTAIN = 0.9       # vol(A ∩ B) / vol(A)
NEXT_TO_DIST = 0.30    # m, 2D footprint distance
MIN_SIDE = 0.01        # m, degenerate boxes dropped
# v2 (2026-10-27): wall-mounted classes are never subject or support of on / in (docs/RELATIONS.md v2)
WALL_MOUNTED: frozenset[str] = frozenset({
    "wall-plug", "switch", "picture", "panel", "pipe", "vent",                                   # Replica
    "mirror", "wall_lamp", "wall_clock", "light_switch", "socket", "electric_outlet", "wall_panel",  # HSSD (2026-10-27b)
    "curtain", "curtain_rod", "window_shutter", "towel_ring", "towel_rack", "wall_hook_rack", "wall_board",
    "photo_mount", "showerhead",
})


def _footprint_overlap(a: np.ndarray, b: np.ndarray, h: list[int]) -> np.ndarray:
    lo = np.maximum(a[:, None, 0, h], b[None, :, 0, h])
    hi = np.minimum(a[:, None, 1, h], b[None, :, 1, h])
    return np.prod(np.clip(hi - lo, 0.0, None), axis=-1)


def _footprint_dist(a: np.ndarray, b: np.ndarray, h: list[int]) -> np.ndarray:
    gap = np.maximum(a[:, None, 0, h] - b[None, :, 1, h], b[None, :, 0, h] - a[:, None, 1, h])
    return np.linalg.norm(np.clip(gap, 0.0, None), axis=-1)


def compute_relations(objects: list[PredObject], up: int = 2,
                      wall_mounted: frozenset[str] = WALL_MOUNTED) -> list[PredRelation]:
    """on / in / next_to between `objects` (callers pass non-structural objects only)."""
    objs = [o for o in objects if np.all(o.aabb_max - o.aabb_min >= MIN_SIDE)]
    n = len(objs)
    if n < 2:
        return []
    h = [k for k in range(3) if k != up]
    B = np.stack([np.stack([o.aabb_min, o.aabb_max]) for o in objs])
    ids = np.array([o.id for o in objs])
    top, bot = B[:, 1, up], B[:, 0, up]
    area = np.prod(B[:, 1, h] - B[:, 0, h], axis=-1)
    vol = volume(B)
    gap = bot[:, None] - top[None, :]                       # (A, B)
    fo = _footprint_overlap(B, B, h) / area[:, None]
    ci = intersection_volume(B, B) / vol[:, None]
    eye = np.eye(n, dtype=bool)

    wm = np.array([o.label in wall_mounted for o in objs])
    free = ~(wm[:, None] | wm[None, :])                     # wall-mounted: neither subject nor support
    on_raw = (gap >= ON_GAP_MIN) & (gap <= ON_GAP_MAX) & (fo >= ON_FOOTPRINT) & (top[:, None] > top[None, :]) & ~eye
    in_raw = (ci >= IN_CONTAIN) & (vol[:, None] < vol[None, :]) & ~on_raw & ~eye
    on_ok, in_ok = on_raw & free, in_raw & free

    rels: list[PredRelation] = []
    for a in range(n):
        cand = np.flatnonzero(on_ok[a])
        if len(cand):
            b = min(cand, key=lambda j: (-top[j], area[j], ids[j]))
            rels.append(PredRelation(int(ids[a]), "on", int(ids[b])))
        cand = np.flatnonzero(in_ok[a])
        if len(cand):
            b = min(cand, key=lambda j: (vol[j], ids[j]))
            rels.append(PredRelation(int(ids[a]), "in", int(ids[b])))
    # "no on/in relation in either direction" uses the qualifying sets, not only the kept one.
    linked = on_ok | in_ok
    linked = linked | linked.T
    u_overlap = np.minimum(top[:, None], top[None, :]) - np.maximum(bot[:, None], bot[None, :]) > 0
    near = _footprint_dist(B, B, h) <= NEXT_TO_DIST
    nt = u_overlap & near & ~linked & ~eye
    for a, b in zip(*np.nonzero(np.triu(nt, 1))):
        i, j = sorted((int(ids[a]), int(ids[b])))
        rels.append(PredRelation(i, "next_to", j))
    return rels
