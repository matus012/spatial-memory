"""Object-level scoring: one-to-one matching at 3D IoU >= tau (docs/CONTRACT.md)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from spatialmem.eval.boxes import iou_matrix
from spatialmem.eval.contract import STRUCTURAL, Prediction


@dataclass
class ObjectScore:
    tau: float
    n_pred: int
    n_gt: int
    tp: int
    precision: float
    recall: float
    f1: float
    label_acc: float
    matches: list[tuple[int, int, float]]  # (pred id, gt id, iou) — object ids, not list indices
    n_label_correct: int = 0


def match(iou: np.ndarray, tau: float) -> list[tuple[int, int, float]]:
    """Max-cardinality assignment of pairs with IoU >= tau, ties by larger IoU sum."""
    n, m = iou.shape
    if n == 0 or m == 0:
        return []
    eligible = iou >= tau
    weight = np.where(eligible, 1.0 + iou / (max(n, m) + 1), 0.0)
    rows, cols = linear_sum_assignment(weight, maximize=True)
    return [(int(r), int(c), float(iou[r, c])) for r, c in zip(rows, cols) if eligible[r, c]]


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def without_structural(p: Prediction, exclude: tuple[str, ...] = STRUCTURAL) -> Prediction:
    """Drop objects with a structural label (and relations touching them) — applied to both sides."""
    keep = [o for o in p.objects if o.label not in exclude]
    ids = {o.id for o in keep}
    rels = [r for r in p.relations if r.subj in ids and r.obj in ids]
    return Prediction(p.scene, keep, rels, p.sem_points, p.sem_labels)


def score_objects(pred: Prediction, gt: Prediction, tau: float, exclude: tuple[str, ...] = STRUCTURAL) -> ObjectScore:
    """Object P/R/F1 + label acc @ tau. Callers must pass the same filtered `pred`/`gt` to score_relations."""
    pred, gt = without_structural(pred, exclude), without_structural(gt, exclude)
    iou = iou_matrix(pred.boxes(), gt.boxes())
    mt = match(iou, tau)
    tp = len(mt)
    p = tp / len(pred.objects) if pred.objects else 0.0
    r = tp / len(gt.objects) if gt.objects else 0.0
    correct = sum(pred.objects[i].label == gt.objects[j].label for i, j, _ in mt)
    acc = correct / tp if tp else float("nan")
    ids = [(pred.objects[i].id, gt.objects[j].id, v) for i, j, v in mt]
    return ObjectScore(tau, len(pred.objects), len(gt.objects), tp, p, r, _f1(p, r), acc, ids, correct)


@dataclass
class RelationScore:
    n_pred: int
    n_gt: int
    tp: int
    precision: float
    recall: float
    f1: float


def score_relations(pred: Prediction, gt: Prediction, obj_score: ObjectScore,
                    exclude: tuple[str, ...] = STRUCTURAL) -> RelationScore:
    """Relation P/R/F1; endpoints mapped through the object matching (tau 0.25 by contract)."""
    pred, gt = without_structural(pred, exclude), without_structural(gt, exclude)
    p2g = {pi: gi for pi, gi, _ in obj_score.matches}
    gt_keys = {r.key() for r in gt.relations}
    pred_keys = {r.key() for r in pred.relations}
    tp = 0
    for s, pr, o in pred_keys:
        if s in p2g and o in p2g:
            a, b = p2g[s], p2g[o]
            if pr == "next_to":
                a, b = sorted((a, b))
            tp += (a, pr, b) in gt_keys
    p = tp / len(pred_keys) if pred_keys else 0.0
    r = tp / len(gt_keys) if gt_keys else 0.0
    return RelationScore(len(pred_keys), len(gt_keys), tp, p, r, _f1(p, r))
