"""HSSD stream checks and frame maths without Habitat (task 002)."""
import numpy as np

from spatialmem.hssd.check import StreamCheck, backproject
from spatialmem.hssd.sim import GL_TO_CV, quat_to_mat

INTR = {"W": 64, "H": 48, "fx": 32.0, "fy": 32.0, "cx": 31.5, "cy": 23.5}


def wall(z=2.0):
    return np.full((48, 64), z), np.ones((48, 64), np.int64)


def test_g4_inside_and_outside():
    d, inst = wall(2.0)
    halfx, halfy = 2.0 * 32 / 32, 2.0 * 24 / 32
    near = {"id": 1, "aabb_min": [-halfx, -halfy, 1.99], "aabb_max": [halfx, halfy, 2.0]}
    c = StreamCheck([near], INTR)
    c.add(d, inst, np.eye(4))
    r = c.result()
    assert r["G4_median_inside"] == 1.0 and r["G3_unknown_rendered_ids"] == []
    far = {"id": 1, "aabb_min": [-5, -5, 3.0], "aabb_max": [5, 5, 4.0]}
    c = StreamCheck([far], INTR)
    c.add(d, inst, np.eye(4))
    assert c.result()["G4_median_inside"] == 0.0


def test_g4_uses_pose():
    d, inst = wall(2.0)
    T = np.eye(4)
    T[:3, 3] = [10, 0, 0]
    box = {"id": 1, "aabb_min": [8, -2, 1.9], "aabb_max": [12, 2, 2.1]}
    c = StreamCheck([box], INTR)
    c.add(d, inst, T)
    assert c.result()["G4_median_inside"] == 1.0


def test_g3_unknown_and_g6_pixels():
    d, inst = wall(2.0)
    inst[:, :10] = 7  # not a GT id
    inst[:, 10:] = 0
    inst[:5, 10:50] = 1  # 200 px
    c = StreamCheck([{"id": 1, "aabb_min": [-9, -9, 0], "aabb_max": [9, 9, 9]}], INTR)
    c.add(d, inst, np.eye(4))
    r = c.result()
    assert r["G3_unknown_rendered_ids"] == [7] and r["G6_coverage"] == 1.0


def test_backproject_centre_pixel():
    P = backproject(np.full((48, 64), 3.0), INTR, np.eye(4))
    assert np.allclose(P[0, 0], [(0 - 31.5) / 32 * 3, (0 - 23.5) / 32 * 3, 3])


def test_quat_and_gl_to_cv():
    a = np.pi / 3
    R = quat_to_mat(np.cos(a / 2), 0, np.sin(a / 2), 0)  # yaw a about +Y
    assert np.allclose(R @ [0, 0, -1], [-np.sin(a), 0, -np.cos(a)])  # habitat forward -Z turns left
    assert np.allclose((np.eye(4) @ GL_TO_CV)[:3, 2], [0, 0, -1])     # OpenCV +z (forward) = GL -Z


def test_group_parts_touching_components():
    from spatialmem.hssd.labels import group_label
    from spatialmem.hssd.sim import group_parts

    def r(par, lo, hi, cm="", lex=""):
        return {"parent": par, "aabb_min": lo, "aabb_max": hi, "category_main": cm, "category_lexicon": lex}
    recs = [r(None, [0, 0, 0], [1, 1, 1], cm="table"),
            r("cab", [5, 0, 0], [6, 1, 1], cm="cabinet", lex="drawer"),
            r("cab", [6.015, 0, 0], [7, 1, 1], cm="cabinet", lex="door"),   # 1.5 cm gap: same object
            r("cab", [9, 0, 0], [10, 1, 1], cm="cabinet", lex="drawer"),    # 2 m away: second cabinet
            r(None, [5.5, 0, 0], [6.5, 1, 1], cm="vase")]                   # touches, but not a part
    g = group_parts(recs)
    assert g == [[0], [1, 2], [3], [4]]
    assert [group_label([recs[i] for i in x]) for x in g] == ["table", "cabinet", "cabinet", "vase"]
    recs[1]["category_main"] = recs[2]["category_main"] = ""
    assert group_label([recs[1], recs[2]]) == "door"   # tie drawer/door -> alphabetical
