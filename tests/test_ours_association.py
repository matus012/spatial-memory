"""ours v1 association on toy objects: merge / split / merge pass / min-detections."""
import numpy as np

from spatialmem.ours.association import Detection, MapObject, ObjectMap

CFG = dict(voxel=0.01, bbox_margin=0.05, geo_radius=0.025, lam=0.5, tau=0.55, tau_g=0.5, tau_c=0.8, top_k=5,
           min_detections=2, merge_use_cos=True)


def box(lo, hi, step=0.02) -> np.ndarray:
    ax = [np.arange(a, b + 1e-9, step) for a, b in zip(lo, hi)]
    return np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)


def feat(i: int, d: int = 8) -> np.ndarray:
    f = np.zeros(d)
    f[i] = 1.0
    return f


def test_same_box_merges():
    M = ObjectMap(CFG)
    b = box((0, 0, 0), (0.4, 0.4, 0.4))
    M.integrate([Detection(b, feat(0), 1000)])
    M.integrate([Detection(b + 0.005, feat(0), 900)])
    assert len(M.objects) == 1 and M.objects[0].n_det == 2


def test_disjoint_boxes_split():
    M = ObjectMap(CFG)
    M.integrate([Detection(box((0, 0, 0), (0.4, 0.4, 0.4)), feat(0), 1000)])
    M.integrate([Detection(box((2, 0, 0), (2.4, 0.4, 0.4)), feat(0), 1000)])
    assert len(M.objects) == 2


def halves_map(f2: np.ndarray) -> ObjectMap:
    M = ObjectMap(CFG)
    M.objects = [MapObject(box((0, 0, 0), (0.6, 0.4, 0.4)), [(500, feat(0))], 2),
                 MapObject(box((0.2, 0, 0), (0.8, 0.4, 0.4)), [(500, f2)], 2)]
    return M


def test_merge_pass_fuses_high_cos_halves():
    M = halves_map(feat(0))
    assert M.merge_pass() == 1
    assert len(M.objects) == 1 and M.objects[0].n_det == 4
    o = M.objects[0]
    assert np.allclose(o.points.min(0), 0, atol=0.01) and np.isclose(o.points.max(0)[0], 0.8, atol=0.01)


def test_merge_pass_keeps_low_cos_halves_apart():
    M = halves_map(feat(1))
    assert M.merge_pass() == 0 and len(M.objects) == 2


def test_min_detections_drops_singletons():
    M = ObjectMap(CFG)
    b = box((0, 0, 0), (0.4, 0.4, 0.4))
    M.integrate([Detection(b, feat(0), 1000), Detection(box((2, 0, 0), (2.4, 0.4, 0.4)), feat(1), 800)])
    M.integrate([Detection(b, feat(0), 1000)])
    assert [o.n_det for o in M.objects] == [2, 1]
    assert len(M.final()) == 1 and M.final()[0].points.max() < 1


def test_descriptor_is_topk_mean_by_area():
    M = ObjectMap({**CFG, "top_k": 2})
    b = box((0, 0, 0), (0.4, 0.4, 0.4))
    f = [feat(0), 0.9 * feat(0) + np.sqrt(1 - 0.81) * feat(1), feat(2)]
    M.integrate([Detection(b, f[0], 100)])
    M.integrate([Detection(b, f[1], 300)])
    M.integrate([Detection(b, f[2], 50)])      # cos 0, geo 1 -> score 0.5 < tau: new object
    assert len(M.objects) == 2
    o = M.objects[0]
    assert [a for a, _ in o.views] == [300, 100]
    m = (f[0] + f[1]) / np.linalg.norm(f[0] + f[1])
    assert np.allclose(o.descriptor(), m)


def test_merge_pass_geometry_only_ignores_cos():
    M = halves_map(feat(1))
    M.c = {**CFG, "merge_use_cos": False}
    assert M.merge_pass() == 1 and len(M.objects) == 1
