"""Object map + association (task 004 Design step 4-5). numpy / scipy only, no GPU.

Per frame, every detection is scored against the objects that existed before the frame (ConceptGraphs-style
frame batch): candidates = objects whose AABB overlaps the detection's AABB grown by `bbox_margin`;
score = lam * geo + (1 - lam) * cos, geo = fraction of detection points within `geo_radius` of the object's points.
Best candidate with score >= tau absorbs the detection, else the detection starts a new object.
Merge pass: object pairs with overlapping (grown) AABBs, max(geo(a->b), geo(b->a)) >= tau_g and cos >= tau_c
(merge_use_cos false: geometry only, tau_c ignored, pairs ranked by geo alone — the lam = 1 ablation).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from spatialmem.ours.lift import voxel_down


@dataclass
class Detection:
    points: np.ndarray        # (N, 3) world
    feat: np.ndarray          # (D,) L2-normalised
    area: int                 # mask pixels


@dataclass
class MapObject:
    points: np.ndarray
    views: list[tuple[int, np.ndarray]]      # (mask area, feat), top-k by area
    n_det: int = 1
    _tree: cKDTree | None = field(default=None, repr=False)
    key: tuple[int, int] = field(default_factory=lambda: (next(_UID), 0), repr=False)   # (uid, version)

    @property
    def lo(self) -> np.ndarray:
        return self.points.min(0)

    @property
    def hi(self) -> np.ndarray:
        return self.points.max(0)

    @property
    def tree(self) -> cKDTree:
        if self._tree is None:
            self._tree = cKDTree(self.points)
        return self._tree

    def descriptor(self) -> np.ndarray:
        f = np.mean([v for _, v in self.views], axis=0)
        return f / max(np.linalg.norm(f), 1e-12)


def overlaps(lo: np.ndarray, hi: np.ndarray, LO: np.ndarray, HI: np.ndarray, margin: float) -> np.ndarray:
    """(..., 3) box(es) lo/hi grown by margin vs (M, 3) boxes LO/HI -> (..., M) bool AABB overlap."""
    return np.all((lo[..., None, :] - margin <= HI) & (LO <= hi[..., None, :] + margin), -1)


_UID = itertools.count()


def pair_geo(a: MapObject, b: MapObject, radius: float, tau_g: float) -> float:
    """max(geo(a->b), geo(b->a)) exactly whenever it is >= tau_g, else some value < tau_g.
    geo(x->y) <= fraction of x's points inside y's AABB grown by radius; that bound skips KD queries that cannot
    reach tau_g or raise the max."""
    best = 0.0
    for x, y in ((a, b), (b, a)):
        bound = np.mean(np.all((x.points >= y.lo - radius) & (x.points <= y.hi + radius), 1))
        if bound >= tau_g and bound > best:
            best = max(best, geo(x.points, y, radius))
    return best


def geo(points: np.ndarray, obj: MapObject, radius: float) -> float:
    d, _ = obj.tree.query(points, k=1, distance_upper_bound=radius)
    return float(np.mean(np.isfinite(d)))


class ObjectMap:
    def __init__(self, cfg: dict):
        self.c = cfg
        self.objects: list[MapObject] = []
        self._pair: dict[tuple, float] = {}       # pair_geo cache keyed on both objects' (uid, version)

    def _boxes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.objects:
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 0))
        return (np.stack([o.lo for o in self.objects]), np.stack([o.hi for o in self.objects]),
                np.stack([o.descriptor() for o in self.objects]))

    def _absorb(self, o: MapObject, points: np.ndarray, views: list[tuple[int, np.ndarray]], n_det: int) -> None:
        o.points = voxel_down(np.concatenate([o.points, points]), self.c["voxel"])
        o.views = sorted(o.views + views, key=lambda v: -v[0])[: self.c["top_k"]]
        o.n_det += n_det
        o._tree = None
        o.key = (o.key[0], o.key[1] + 1)

    def integrate(self, dets: list[Detection]) -> None:
        c = self.c
        LO, HI, D = self._boxes()
        assign: list[int] = []
        for d in dets:
            best, best_s = -1, -np.inf
            for j in np.flatnonzero(overlaps(d.points.min(0), d.points.max(0), LO, HI, c["bbox_margin"])):
                s = c["lam"] * geo(d.points, self.objects[j], c["geo_radius"]) + (1 - c["lam"]) * float(d.feat @ D[j])
                if s > best_s:
                    best, best_s = j, s
            assign.append(best if best_s >= c["tau"] else -1)
        for d, j in zip(dets, assign):
            if j < 0:
                self.objects.append(MapObject(points=d.points, views=[(d.area, d.feat)]))
            else:
                self._absorb(self.objects[j], d.points, [(d.area, d.feat)], 1)

    def merge_pass(self) -> int:
        """Greedy rounds: qualifying pairs by geo + cos descending, each object merged at most once per round."""
        c, merged = self.c, 0
        while True:
            objs = self.objects
            LO, HI, D = self._boxes()
            C = D @ D.T
            use_cos = c["merge_use_cos"]
            cand = np.triu(overlaps(LO, HI, LO, HI, c["bbox_margin"]) & ((C >= c["tau_c"]) if use_cos else True), 1)
            pairs = []
            for i, j in zip(*np.nonzero(cand)):
                k = (objs[i].key, objs[j].key)
                if k not in self._pair:
                    self._pair[k] = pair_geo(objs[i], objs[j], c["geo_radius"], c["tau_g"])
                g = self._pair[k]
                if g >= c["tau_g"]:
                    pairs.append((g + (float(C[i, j]) if use_cos else 0.0), int(i), int(j)))
            if not pairs:
                live = {o.key for o in objs}
                self._pair = {k: v for k, v in self._pair.items() if k[0] in live and k[1] in live}
                return merged
            used: set[int] = set()
            gone: set[int] = set()
            for _, i, j in sorted(pairs, reverse=True):
                if i in used or j in used:
                    continue
                self._absorb(objs[i], objs[j].points, objs[j].views, objs[j].n_det)
                used |= {i, j}
                gone.add(j)
                merged += 1
            self.objects = [o for k, o in enumerate(objs) if k not in gone]

    def final(self) -> list[MapObject]:
        return [o for o in self.objects if o.n_det >= self.c["min_detections"]]
