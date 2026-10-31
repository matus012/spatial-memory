"""ours v1 lift: a known plane rendered at a known pose comes back within 1 mm."""
import numpy as np

from spatialmem.ours.lift import largest_cluster, lift
from spatialmem.render.raycast import Intrinsics, MeshRaycaster, backproject

INTR = Intrinsics(W=160, H=120, fx=80.0, fy=80.0, cx=79.5, cy=59.5)


def look_pose() -> np.ndarray:
    """c2w (OpenCV axes): camera at (0.3, -0.2, 1.5) looking down -z, tilted 20 deg about x."""
    a = np.deg2rad(20)
    R_down = np.diag([1.0, -1.0, -1.0])                    # cam z -> world -z
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    T = np.eye(4)
    T[:3, :3] = R_down @ Rx
    T[:3, 3] = (0.3, -0.2, 1.5)
    return T


def test_plane_round_trip():
    z0 = 0.25                                                  # plane z = z0, world frame
    V = np.array([[-20, -20, z0], [20, -20, z0], [20, 20, z0], [-20, 20, z0]], float)
    rc = MeshRaycaster(V, np.array([[0, 1, 2], [0, 2, 3]]))
    T = look_pose()
    depth, _ = rc.render(INTR, T)
    assert (depth > 0).all()
    world = backproject(depth, INTR, T)
    mask = np.zeros(depth.shape, bool)
    mask[30:90, 40:120] = True
    p = lift(world, depth > 0, mask, voxel=0.01, eps=0.1, min_points=10)
    assert len(p) > 100
    assert np.abs(p[:, 2] - z0).max() < 1e-3
    sel = world[mask]                                          # recovered points stay inside the masked footprint
    assert (p.min(0)[:2] >= sel.min(0)[:2] - 1e-3).all() and (p.max(0)[:2] <= sel.max(0)[:2] + 1e-3).all()


def test_dbscan_drops_edge_bleed():
    rng = np.random.default_rng(0)
    main = rng.uniform(0, 0.3, (2000, 3))
    bleed = rng.uniform(0, 0.02, (5, 3)) + 2.0
    p = largest_cluster(np.concatenate([main, bleed]), eps=0.1, min_points=10)
    assert len(p) == 2000 and p.max() <= 0.3


def test_empty_mask_gives_no_points():
    world = np.zeros((4, 4, 3))
    assert len(lift(world, np.ones((4, 4), bool), np.zeros((4, 4), bool), 0.01, 0.1, 10)) == 0
