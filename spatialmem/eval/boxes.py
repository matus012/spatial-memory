"""Axis-aligned 3D box geometry. Boxes are (..., 2, 3) arrays of [min, max]."""
from __future__ import annotations

import numpy as np

MIN_EXTENT = 0.02  # m; extents below this are padded symmetrically before IoU (docs/CONTRACT.md, 2026-10-27b)


def volume(b: np.ndarray) -> np.ndarray:
    return np.prod(np.clip(b[..., 1, :] - b[..., 0, :], 0.0, None), axis=-1)


def intersection_volume(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise intersection volume, a (N,2,3), b (M,2,3) -> (N, M)."""
    lo = np.maximum(a[:, None, 0, :], b[None, :, 0, :])
    hi = np.minimum(a[:, None, 1, :], b[None, :, 1, :])
    return np.prod(np.clip(hi - lo, 0.0, None), axis=-1)


def pad_min_extent(b: np.ndarray, min_extent: float = MIN_EXTENT) -> np.ndarray:
    """Expand every extent < min_extent symmetrically about its centre to min_extent."""
    b = np.array(b, dtype=np.float64, copy=True)
    ext = b[..., 1, :] - b[..., 0, :]
    grow = np.clip(min_extent - ext, 0.0, None) / 2
    b[..., 0, :] -= grow
    b[..., 1, :] += grow
    return b


def iou_matrix(a: np.ndarray, b: np.ndarray, min_extent: float = MIN_EXTENT) -> np.ndarray:
    """Pairwise 3D AABB IoU, (N, M), after min-extent padding of both sides (0 disables padding)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    if min_extent > 0:
        a, b = pad_min_extent(a, min_extent), pad_min_extent(b, min_extent)
    inter = intersection_volume(a, b)
    union = volume(a)[:, None] + volume(b)[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def from_points(p: np.ndarray) -> np.ndarray:
    return np.stack([p.min(axis=0), p.max(axis=0)])
