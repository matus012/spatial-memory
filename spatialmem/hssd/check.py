"""Task 002 stream checks, numpy only (runs in the image and on lpc).

G3  instance uniqueness: GT IDs distinct; every rendered ID is 0 (stage / empty) or a GT ID.
G4  instance / pose chain: back-project each frame's instance pixels with depth + pose (OpenCV c2w); per object
    with >= 500 points over the stream, fraction inside its GT world AABB grown by 2 cm.
    Gate (dev scenes): median >= 0.98 and >= 0.90 for >= 95 % of such objects.
G6  coverage: fraction of GT objects with >= 200 instance pixels in >= 1 frame (gate: median over eval >= 0.80).
"""
from __future__ import annotations

import numpy as np

G4_MARGIN = 0.02
G4_MIN_POINTS = 500
G4_STRIDE = 2
G6_MIN_PIXELS = 200


def backproject(depth: np.ndarray, intr: dict, c2w: np.ndarray) -> np.ndarray:
    """(H, W, 3) world points; depth = z in the camera frame (OpenCV axes), pixel centres at integers."""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x = (u - intr["cx"]) / intr["fx"] * depth
    y = (v - intr["cy"]) / intr["fy"] * depth
    P = np.stack([x, y, depth], -1)
    return P @ c2w[:3, :3].T + c2w[:3, 3]


class StreamCheck:
    def __init__(self, gt_objects: list[dict], intr: dict, cover_ids: list[int] | None = None):
        ids = [o["id"] for o in gt_objects]
        self.cover = np.array(ids if cover_ids is None else cover_ids, dtype=np.int64)
        self.ids = np.array(ids)
        self.dup_ids = len(ids) - len(set(ids))
        n = max(ids) + 1 if ids else 1
        self.lo = np.full((n, 3), np.inf)
        self.hi = np.full((n, 3), -np.inf)
        for o in gt_objects:
            self.lo[o["id"]] = np.asarray(o["aabb_min"]) - G4_MARGIN
            self.hi[o["id"]] = np.asarray(o["aabb_max"]) + G4_MARGIN
        self.valid = np.zeros(n, bool)
        self.valid[self.ids] = True
        self.intr = intr
        self.n_pts = np.zeros(n, np.int64)
        self.n_in = np.zeros(n, np.int64)
        self.max_px = np.zeros(n, np.int64)
        self.unknown_ids: set[int] = set()
        self.frames = 0

    def add(self, depth: np.ndarray, inst: np.ndarray, c2w: np.ndarray) -> None:
        self.frames += 1
        u = np.unique(inst)
        bad = [int(i) for i in u if i != 0 and (i >= len(self.valid) or not self.valid[i])]
        self.unknown_ids.update(bad)
        cnt = np.bincount(inst[(inst > 0) & (inst < len(self.valid))].ravel(), minlength=len(self.valid))
        self.max_px = np.maximum(self.max_px, cnt[:len(self.valid)])
        s = (slice(None, None, G4_STRIDE), slice(None, None, G4_STRIDE))
        d, k = depth[s], inst[s]
        # strided grid: pixel (r, c) of the subsampled image = (G4_STRIDE r, G4_STRIDE c) of the full one
        H, W = d.shape
        uu, vv = np.meshgrid(np.arange(W) * G4_STRIDE, np.arange(H) * G4_STRIDE)
        m = (d > 0) & np.isfinite(d) & (k > 0) & (k < len(self.valid))
        z = d[m]
        P = np.stack([(uu[m] - self.intr["cx"]) / self.intr["fx"] * z, (vv[m] - self.intr["cy"]) / self.intr["fy"] * z, z], -1)
        P = P @ c2w[:3, :3].T + c2w[:3, 3]
        ids = k[m]
        inside = np.all((P >= self.lo[ids]) & (P <= self.hi[ids]), axis=1)
        self.n_pts += np.bincount(ids, minlength=len(self.valid))
        self.n_in += np.bincount(ids[inside], minlength=len(self.valid))

    def result(self) -> dict:
        ok = self.valid & (self.n_pts >= G4_MIN_POINTS)
        frac = self.n_in[ok] / self.n_pts[ok]
        seen = self.max_px[self.cover] >= G6_MIN_PIXELS
        worst = np.argsort(frac)[:10]
        oid = np.flatnonzero(ok)
        return {
            "frames": self.frames, "n_gt": int(len(self.ids)), "G3_duplicate_gt_ids": self.dup_ids,
            "G3_unknown_rendered_ids": sorted(self.unknown_ids),
            "G4_objects": int(ok.sum()), "G4_median_inside": float(np.median(frac)) if frac.size else None,
            "G4_frac_objects_ge_0.90": float(np.mean(frac >= 0.90)) if frac.size else None,
            "G4_worst": [(int(oid[i]), float(frac[i]), int(self.n_pts[oid[i]])) for i in worst],
            "G6_coverage": float(seen.mean()) if len(self.cover) else None, "G6_n": int(len(self.cover)),
            "G6_seen": int(seen.sum()),
            "per_object": {int(i): {"n_pts": int(self.n_pts[i]), "inside": float(self.n_in[i] / max(self.n_pts[i], 1)),
                                    "max_px": int(self.max_px[i])} for i in self.ids},
        }
