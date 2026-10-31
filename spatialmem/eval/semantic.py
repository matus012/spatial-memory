"""Replica semantic metrics: mIoU / mAcc / f-mIoU from a confusion matrix (docs/CONTRACT.md).

Semantics follow OVO's eval_utils (tberriel/OVO@e0d16f7, MIT): GT rows of ignored classes are dropped;
a class whose TP+FP+FN = 0 is nan and excluded; f-mIoU weights by GT count; scenes pooled by summing
confusion matrices.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from spatialmem.eval.contract import GroundTruth, Prediction

TRANSFER_K = 5  # OVO match_labels_to_vtx: tree.query(mesh_vtx, k=5)


def transfer_labels(points: np.ndarray, labels: np.ndarray, targets: np.ndarray, k: int = TRANSFER_K) -> np.ndarray:
    """Per-target label = mode of the k nearest predicted points' labels (OVO protocol, k=5).

    Unassigned predicted points (label < 0) are dropped first. Mode ties -> smallest label (torch.mode).
    """
    keep = labels >= 0
    points, labels = points[keep], labels[keep]
    if len(points) == 0:
        raise ValueError("all predicted points are unassigned")
    k = min(k, len(points))
    _, idx = cKDTree(points).query(targets, k=k)
    idx = idx.reshape(len(targets), k)
    nb = labels[idx]
    nb_sorted = np.sort(nb, axis=1)
    # mode with smallest-value tie break: count each value per row
    best = nb_sorted[:, 0].copy()
    best_cnt = np.zeros(len(nb), dtype=int)
    for j in range(k):
        cnt = (nb_sorted == nb_sorted[:, j : j + 1]).sum(axis=1)
        better = cnt > best_cnt
        best[better] = nb_sorted[better, j]
        best_cnt[better] = cnt[better]
    return best


def confusion(gt: np.ndarray, pred: np.ndarray, num_classes: int, ignore: list[int] | tuple = ()) -> np.ndarray:
    """(C, C) matrix, rows = GT, cols = pred. GT < 0 or in `ignore` dropped. Every kept pred must be >= 0."""
    gt = np.asarray(gt, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    if gt.shape != pred.shape:
        raise ValueError(f"shape mismatch gt {gt.shape} pred {pred.shape}")
    keep = (gt >= 0) & ~np.isin(gt, np.asarray(list(ignore), dtype=np.int64))
    gt, pred = gt[keep], pred[keep]
    if (pred < 0).any() or (pred >= num_classes).any():
        raise ValueError("predicted labels must be in [0, num_classes)")
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (gt, pred), 1)
    return cm


@dataclass
class SemanticScore:
    miou: float
    macc: float
    fmiou: float
    per_class_iou: np.ndarray
    per_class_acc: np.ndarray
    valid: np.ndarray


def metrics(cm: np.ndarray, ignore: list[int] | tuple = ()) -> SemanticScore:
    c = cm.shape[0]
    tp = np.diag(cm).astype(np.float64)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    denom = tp + fp + fn
    keep = np.ones(c, dtype=bool)
    keep[list(ignore)] = False
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(denom > 0, tp / denom, np.nan)
        acc = np.where(denom > 0, tp / np.maximum(tp + fn, 1e-6), np.nan)
    valid = keep & ~np.isnan(iou)
    w = (tp + fn)[valid]
    return SemanticScore(
        miou=float(np.mean(iou[valid])),
        macc=float(np.mean(acc[valid])),
        fmiou=float(np.sum(iou[valid] * w) / w.sum()),
        per_class_iou=iou,
        per_class_acc=acc,
        valid=valid,
    )


def score_semantic(pred: Prediction, gt: GroundTruth, num_classes: int, k: int = TRANSFER_K) -> tuple[SemanticScore, np.ndarray]:
    """The mapper path: pred.sem_points/sem_labels -> k-NN mode onto gt.vertices -> confusion -> metrics.

    Returns the per-scene score and its confusion matrix; pool scenes with `pooled(cms)`.
    """
    if pred.sem_points is None or pred.sem_labels is None:
        raise ValueError("prediction carries no semantic points")
    if gt.vertices is None or gt.vertex_labels is None:
        raise ValueError("GT carries no vertices / labels")
    lab = transfer_labels(np.asarray(pred.sem_points, float), np.asarray(pred.sem_labels, np.int64), gt.vertices, k)
    cm = confusion(gt.vertex_labels, lab, num_classes)
    return metrics(cm), cm


def pooled(cms: list[np.ndarray]) -> SemanticScore:
    """Protocol headline: metrics of the summed confusion matrices (OVO eval_semantics: confusion.sum(0))."""
    return metrics(np.sum(np.stack(cms), axis=0))
