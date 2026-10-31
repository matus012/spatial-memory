"""Matching objective, id handling, structural filtering, thin boxes, semantic entry (code review 1, 3, 7, 8, 10, 11)."""
import numpy as np

from spatialmem.eval.boxes import iou_matrix
from spatialmem.eval.contract import GroundTruth, PredObject, PredRelation, Prediction
from spatialmem.eval.objects import match, score_objects, score_relations
from spatialmem.eval.semantic import TRANSFER_K, metrics, pooled, score_semantic, transfer_labels
from tests.test_objects import grid_scene


def test_cardinality_beats_iou_sum():
    iou = np.array([[0.95, 0.30], [0.30, 0.00]])
    assert sorted((i, j) for i, j, _ in match(iou, 0.25)) == [(0, 1), (1, 0)]  # IoU-sum keeps only (0, 0)


def test_equal_cardinality_tie_broken_by_iou_sum():
    assert sorted((i, j) for i, j, _ in match(np.array([[0.90, 0.40], [0.40, 0.30]]), 0.25)) == [(0, 0), (1, 1)]
    assert sorted((i, j) for i, j, _ in match(np.array([[0.40, 0.90], [0.30, 0.40]]), 0.25)) == [(0, 1), (1, 0)]


def test_relations_with_permuted_ids():
    gt = grid_scene(4)
    gt.relations = [PredRelation(0, "next_to", 3), PredRelation(1, "on", 2)]
    perm = {0: 103, 1: 101, 2: 100, 3: 102}
    pred = Prediction("s", [PredObject(perm[o.id], o.label, o.aabb_min, o.aabb_max) for o in gt.objects],
                      [PredRelation(perm[3], "next_to", perm[0]), PredRelation(perm[1], "on", perm[2]),
                       PredRelation(perm[2], "on", perm[1])])
    s = score_objects(pred, gt, 0.25)
    assert {(p, g) for p, g, _ in s.matches} == {(perm[i], i) for i in range(4)}
    r = score_relations(pred, gt, s)
    assert (r.tp, r.n_pred, r.n_gt) == (2, 3, 2)


def test_structural_removed_both_sides():
    gt = grid_scene(3)
    wall = PredObject(50, "wall", [0, 0, 0], [10, 0.1, 3])
    gt_w = Prediction("s", gt.objects + [wall], [PredRelation(0, "next_to", 50)])
    pred_w = Prediction("s", gt.objects + [PredObject(77, "floor", [0, 0, -0.1], [10, 10, 0])])
    s = score_objects(pred_w, gt_w, 0.25)
    assert (s.n_pred, s.n_gt, s.f1) == (3, 3, 1.0)
    assert score_relations(pred_w, gt_w, s).n_gt == 0


def test_flat_box_matches_itself():
    pic = PredObject(1, "picture", [0, 0, 1], [0.6, 0.0, 1.5])  # zero thickness
    gt = Prediction("s", [pic])
    assert score_objects(gt, gt, 0.5).f1 == 1.0
    assert iou_matrix(gt.boxes(), gt.boxes(), min_extent=0)[0, 0] == 0.0  # the unpadded failure mode
    # padded to 2 cm about its centre; a copy offset 5 mm overlaps 1.5 of 2.5 cm along y
    near = Prediction("s", [PredObject(1, "picture", [0, 0.005, 1], [0.6, 0.005, 1.5])])
    assert abs(iou_matrix(near.boxes(), gt.boxes())[0, 0] - 0.015 / 0.025) < 1e-9
    # extents >= 2 cm are untouched
    cube = Prediction("s", [PredObject(1, "x", [0, 0, 0], [0.03, 0.03, 0.03])])
    shifted = Prediction("s", [PredObject(1, "x", [0.015, 0, 0], [0.045, 0.03, 0.03])])
    assert abs(iou_matrix(cube.boxes(), shifted.boxes())[0, 0] - 1 / 3) < 1e-9


def test_default_k_is_protocol_k():
    assert TRANSFER_K == 5
    p = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.011, 0], [0.02, 0, 0], [0, 0.021, 0], [0.03, 0, 0], [0, 0.031, 0]])
    lb = np.array([4, 1, 1, 4, 4, 1, 1])  # 5-NN {4,1,1,4,4} -> 4 ; 7-NN -> 1
    assert transfer_labels(p, lb, np.zeros((1, 3)))[0] == 4


def test_score_semantic_path_and_pooling():
    rng = np.random.default_rng(3)
    v = rng.random((400, 3))
    lab = rng.integers(-1, 6, 400)
    gt = GroundTruth("s", [], vertices=v, vertex_labels=lab)
    pred = Prediction("s", [], sem_points=v + 1e-7, sem_labels=np.where(lab >= 0, lab, 0))
    s, cm = score_semantic(pred, gt, 6, k=1)
    assert (s.miou, s.macc, s.fmiou) == (1.0, 1.0, 1.0)
    s2, cm2 = score_semantic(Prediction("s", [], sem_points=v, sem_labels=rng.integers(0, 6, 400)), gt, 6, k=1)
    assert s2.miou < 1.0
    assert pooled([cm, cm2]).miou == metrics(cm + cm2).miou


def test_sem_sidecar_roundtrip(tmp_path):
    p = Prediction("s", [PredObject(1, "x", [0, 0, 0], [1, 1, 1])], sem_points=np.random.rand(10, 3),
                   sem_labels=np.arange(10))
    p.to_json(tmp_path / "p.json")
    q = Prediction.from_json(tmp_path / "p.json")
    assert np.array_equal(q.sem_labels, p.sem_labels) and np.allclose(q.sem_points, p.sem_points)
