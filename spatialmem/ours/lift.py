"""Lift a 2D mask to a 3D detection: back-projected mask pixels -> voxel downsample -> DBSCAN largest cluster.

Back-projection is `spatialmem.render.raycast.backproject` (OpenCV pinhole, c2w; proved in tasks 001/002).
"""
from __future__ import annotations

import numpy as np
import open3d as o3d


def voxel_down(points: np.ndarray, voxel: float) -> np.ndarray:
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    return np.asarray(pc.voxel_down_sample(voxel).points)


def largest_cluster(points: np.ndarray, eps: float, min_points: int) -> np.ndarray:
    """Points of the largest DBSCAN cluster; empty if every point is noise."""
    if len(points) < min_points:
        return points[:0]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    lab = np.asarray(pc.cluster_dbscan(eps=eps, min_points=min_points))
    if lab.max() < 0:
        return points[:0]
    return points[lab == np.bincount(lab[lab >= 0]).argmax()]


def lift(world: np.ndarray, valid: np.ndarray, mask: np.ndarray, voxel: float, eps: float, min_points: int) -> np.ndarray:
    """world (H, W, 3) back-projected frame, valid (H, W) depth > 0, mask (H, W) bool -> (N, 3) detection points."""
    return largest_cluster(voxel_down(world[mask & valid], voxel), eps, min_points)
