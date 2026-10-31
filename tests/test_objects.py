"""G3 analytic cases for the object scorer."""
import numpy as np
import pytest

from spatialmem.eval.boxes import iou_matrix
from spatialmem.eval.contract import PredObject, PredRelation, Prediction
from spatialmem.eval.objects import score_objects, score_relations


def box(lo, size=(1.0, 1.0, 1.0)):
    lo = np.asarray(lo, float)
    return np.stack([lo, lo + np.asarray(size, float)])


def grid_scene(n: int, seed: int = 0) -> Prediction:
    """n well-separated boxes of varied size (cross-IoU = 0 even after a shift of < 1 extent)."""
    rng = np.random.default_rng(seed)
    objs = []
    for i in range(n):
        size = rng.uniform(0.2, 1.5, 3)
        lo = np.array([i * 4.0, (i % 3) * 4.0, 0.0])
        objs.append(PredObject(i, f"c{i % 5}", lo, lo + size))
    return Prediction("synthetic", objs)


def test_unit_cubes_offset_half():
    a, b = box([0, 0, 0])[None], box([0.5, 0, 0])[None]
    assert abs(iou_matrix(a, b)[0, 0] - 1 / 3) < 1e-9


def test_identical_and_disjoint():
    a = box([0, 0, 0])[None]
    assert iou_matrix(a, a)[0, 0] == 1.0
    assert iou_matrix(a, box([2, 0, 0])[None])[0, 0] == 0.0
    assert iou_matrix(a, box([1, 0, 0])[None])[0, 0] == 0.0  # touching faces


@pytest.mark.parametrize("tau", [0.25, 0.5])
def test_gt_as_pred_perfect(tau):
    gt = grid_scene(20)
    s = score_objects(gt, gt, tau)
    assert (s.precision, s.recall, s.f1, s.label_acc) == (1.0, 1.0, 1.0, 1.0)


@pytest.mark.parametrize("k", [0, 1, 5, 19])
def test_drop_k(k):
    n = 20
    gt = grid_scene(n)
    pred = Prediction("synthetic", gt.objects[k:])
    for tau in (0.25, 0.5):
        s = score_objects(pred, gt, tau)
        assert s.recall == (n - k) / n
        assert s.precision == 1.0


@pytest.mark.parametrize("n_relabel", [0, 3, 10, 20])
def test_relabel_fraction(n_relabel):
    n = 20
    gt = grid_scene(n)
    objs = [PredObject(o.id, ("WRONG" if i < n_relabel else o.label), o.aabb_min, o.aabb_max)
            for i, o in enumerate(gt.objects)]
    s = score_objects(Prediction("synthetic", objs), gt, 0.25)
    assert s.label_acc == 1 - n_relabel / n
    assert s.f1 == 1.0


@pytest.mark.parametrize("frac", [0.0, 0.1, 0.2, 0.3, 0.33, 0.34, 0.5, 0.59, 0.61, 0.9])
def test_shift_closed_form(frac):
    """Shift each box by frac * its own x-extent: IoU = (1 - frac) / (1 + frac) for every object.

    Boundaries (frac = 1/3 -> IoU 0.5, frac = 0.6 -> IoU 0.25) are bracketed, not hit: exact float ties
    are not decidable."""
    n = 20
    gt = grid_scene(n)
    objs = []
    for o in gt.objects:
        d = np.array([frac * (o.aabb_max[0] - o.aabb_min[0]), 0, 0])
        objs.append(PredObject(o.id, o.label, o.aabb_min + d, o.aabb_max + d))
    pred = Prediction("synthetic", objs)
    iou_self = (1 - frac) / (1 + frac)
    for tau in (0.25, 0.5):
        s = score_objects(pred, gt, tau)
        assert s.tp == (n if iou_self >= tau else 0)
        for i, j, iou in s.matches:
            assert i == j and abs(iou - iou_self) < 1e-9


def test_shift_absolute_offset_closed_form():
    """Unit cubes shifted by d along x: IoU = (1 - d) / (1 + d); match iff d <= (1 - tau) / (1 + tau)."""
    gt = Prediction("s", [PredObject(i, "c", [3.0 * i, 0, 0], [3.0 * i + 1, 1, 1]) for i in range(5)])
    for d in np.linspace(0, 1.2, 25):
        pred = Prediction("s", [PredObject(o.id, "c", o.aabb_min + [d, 0, 0], o.aabb_max + [d, 0, 0])
                                for o in gt.objects])
        for tau in (0.25, 0.5):
            iou = max(0.0, (1 - d) / (1 + d))
            s = score_objects(pred, gt, tau)
            assert s.tp == (5 if iou >= tau - 1e-12 else 0)


def test_max_cardinality_matching():
    """Greedy-by-IoU takes pred0-gt0 (0.538) and strands gt1; max cardinality matches both pairs."""
    gt = Prediction("s", [PredObject(0, "c", *box([0, 0, 0])), PredObject(1, "c", *box([0.6, 0, 0]))])
    pred = Prediction("s", [PredObject(0, "c", *box([0.3, 0, 0])), PredObject(1, "c", *box([-0.3, 0, 0]))])
    s = score_objects(pred, gt, 0.25)
    assert s.tp == 2


def test_empty_cases():
    gt = grid_scene(3)
    s = score_objects(Prediction("s", []), gt, 0.25)
    assert (s.tp, s.precision, s.recall, s.f1) == (0, 0.0, 0.0, 0.0)
    assert np.isnan(s.label_acc)


def test_relations_scoring():
    gt = grid_scene(4)
    gt.relations = [PredRelation(0, "on", 1), PredRelation(2, "next_to", 3), PredRelation(1, "in", 3)]
    s_obj = score_objects(gt, gt, 0.25)
    assert score_relations(gt, gt, s_obj).f1 == 1.0
    # reversed next_to is the same relation; a duplicate counts once; a wrong predicate is a FP
    pred = Prediction("s", gt.objects, [PredRelation(0, "on", 1), PredRelation(3, "next_to", 2),
                                        PredRelation(0, "on", 1), PredRelation(1, "on", 3)])
    r = score_relations(pred, gt, s_obj)
    assert (r.n_pred, r.tp, r.n_gt) == (3, 2, 3)
    assert abs(r.precision - 2 / 3) < 1e-12 and abs(r.recall - 2 / 3) < 1e-12
    # an unmatched endpoint cannot score
    pred2 = Prediction("s", gt.objects[1:], [PredRelation(2, "next_to", 3), PredRelation(0, "on", 1)])
    r2 = score_relations(pred2, gt, score_objects(pred2, gt, 0.25))
    assert r2.tp == 1


def test_json_roundtrip(tmp_path):
    gt = grid_scene(3)
    gt.relations = [PredRelation(0, "on", 1)]
    p = tmp_path / "x.json"
    gt.to_json(p)
    assert Prediction.from_json(p).to_json() == gt.to_json()
