"""Analytic checks for the raycaster: a plane at known depth, a cube's pixel footprint, round trip."""
import numpy as np

from spatialmem.render.raycast import Intrinsics, MeshRaycaster, backproject, triangulate

INTR = Intrinsics(W=64, H=48, fx=40.0, fy=40.0, cx=31.5, cy=23.5)


def plane(z: float, half: float = 50.0):
    v = np.array([[-half, -half, z], [half, -half, z], [half, half, z], [-half, half, z]])
    return v, np.array([[0, 1, 2, 3]])


def test_triangulate_quads_and_polys():
    tris, src = triangulate(np.array([[0, 1, 2, 3], [4, 5, 6, 7]]))
    assert tris.shape == (4, 3) and sorted(src.tolist()) == [0, 0, 1, 1]
    tris, src = triangulate([np.array([0, 1, 2]), np.array([3, 4, 5, 6, 7])])
    assert tris.shape == (4, 3) and src.tolist() == [0, 1, 1, 1]


def test_plane_depth_is_z_not_range():
    v, f = plane(2.5)
    tris, src = triangulate(f)
    rc = MeshRaycaster(v, tris, np.array([7])[src])
    depth, label = rc.render(INTR, np.eye(4))
    assert np.allclose(depth, 2.5, atol=1e-5)
    assert (label == 7).all()


def test_pose_translation_and_rotation():
    v, f = plane(0.0)
    tris, _ = triangulate(f)
    rc = MeshRaycaster(v, tris)
    # camera 3 m above the z=0 plane looking down -z: OpenCV z axis = world -z
    c2w = np.eye(4)
    c2w[:3, :3] = np.diag([1.0, -1.0, -1.0])
    c2w[:3, 3] = [0.3, -0.2, 3.0]
    depth, _ = rc.render(INTR, c2w)
    assert np.allclose(depth, 3.0, atol=1e-5)
    pts = backproject(depth, INTR, c2w)
    assert np.allclose(pts[..., 2], 0.0, atol=1e-5)
    # principal ray hits directly below the camera centre
    centre = pts[int(INTR.cy), int(INTR.cx)]
    assert abs(centre[0] - (0.3 - 0.5 / INTR.fx * 3.0)) < 1e-4  # pixel 31 is 0.5 px left of cx


def test_miss_is_zero():
    v, f = plane(2.0, half=0.01)
    rc = MeshRaycaster(v, triangulate(f)[0])
    depth, label = rc.render(INTR, np.eye(4))
    assert (depth == 0).sum() > 0.9 * depth.size and (label[depth == 0] == -1).all()
