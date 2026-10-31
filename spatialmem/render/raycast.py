"""Scene-agnostic CPU mesh raycaster: depth + per-pixel face label at given camera poses.

Open3D RaycastingScene (Embree, CPU, deterministic, no GL context). Camera model: pinhole, OpenCV axes
(x right, y down, z forward), pixel centres at integer coordinates, pose = camera-to-world 4x4.
Depth = z in the camera frame (not ray length), metres; misses are 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d


@dataclass(frozen=True)
class Intrinsics:
    W: int
    H: int
    fx: float
    fy: float
    cx: float
    cy: float

    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])


def triangulate(faces: list[np.ndarray] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fan-triangulate polygons. Returns (T, 3) triangles and (T,) index of the source face."""
    if isinstance(faces, np.ndarray) and faces.ndim == 2:
        n = faces.shape[1]
        tris = np.concatenate([faces[:, [0, k, k + 1]] for k in range(1, n - 1)])
        src = np.tile(np.arange(len(faces)), n - 2)
        return tris.astype(np.int32), src
    tris, src = [], []
    for i, f in enumerate(faces):
        for k in range(1, len(f) - 1):
            tris.append((f[0], f[k], f[k + 1]))
            src.append(i)
    return np.asarray(tris, dtype=np.int32), np.asarray(src)


class MeshRaycaster:
    def __init__(self, vertices: np.ndarray, triangles: np.ndarray, tri_labels: np.ndarray | None = None):
        self.vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        self.triangles = np.ascontiguousarray(triangles, dtype=np.uint32)
        self.tri_labels = None if tri_labels is None else np.asarray(tri_labels)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.core.Tensor(self.vertices), o3d.core.Tensor(self.triangles))

    @staticmethod
    def rays(intr: Intrinsics, c2w: np.ndarray, stride: int = 1) -> tuple[np.ndarray, tuple[int, int]]:
        u = np.arange(0, intr.W, stride, dtype=np.float64)
        v = np.arange(0, intr.H, stride, dtype=np.float64)
        uu, vv = np.meshgrid(u, v)
        d_cam = np.stack([(uu - intr.cx) / intr.fx, (vv - intr.cy) / intr.fy, np.ones_like(uu)], -1)
        d_w = d_cam @ c2w[:3, :3].T                     # z_cam = 1 -> t_hit equals camera-frame depth
        o_w = np.broadcast_to(c2w[:3, 3], d_w.shape)
        return np.concatenate([o_w, d_w], -1).astype(np.float32), uu.shape

    def render(self, intr: Intrinsics, c2w: np.ndarray, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Returns depth (H, W) float32 metres (0 = miss) and label (H, W) int64 (-1 = miss).

        label = tri_labels[hit triangle] if tri_labels was given, else the triangle index.
        """
        r, shape = self.rays(intr, c2w, stride)
        ans = self.scene.cast_rays(o3d.core.Tensor(r))
        t = ans["t_hit"].numpy().reshape(shape)
        prim = ans["primitive_ids"].numpy().reshape(shape).astype(np.int64)
        hit = np.isfinite(t)
        prim = np.where(hit, prim, -1)
        depth = np.where(hit, t, 0.0).astype(np.float32)
        if self.tri_labels is not None:
            label = np.where(hit, self.tri_labels[np.clip(prim, 0, None)], -1)
        else:
            label = prim
        return depth, label


def backproject(depth: np.ndarray, intr: Intrinsics, c2w: np.ndarray, stride: int = 1) -> np.ndarray:
    """(H, W) depth -> (H, W, 3) world points (OpenCV pinhole, c2w); invalid depth gives garbage, mask it."""
    H, W = depth.shape
    uu, vv = np.meshgrid(np.arange(0, W * stride, stride, dtype=np.float64),
                         np.arange(0, H * stride, stride, dtype=np.float64))
    z = depth.astype(np.float64)
    p_cam = np.stack([(uu - intr.cx) / intr.fx * z, (vv - intr.cy) / intr.fy * z, z], -1)
    return p_cam @ c2w[:3, :3].T + c2w[:3, 3]
