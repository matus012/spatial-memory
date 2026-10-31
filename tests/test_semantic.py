"""Semantic metrics vs OVO's own metric code and an sklearn confusion-matrix implementation (G4 core)."""
import numpy as np
import pytest
from sklearn.metrics import confusion_matrix

from spatialmem.eval.reference import ovo_metrics
from spatialmem.eval.semantic import confusion, metrics, transfer_labels


def sk_metrics(gt, pred, c, ignore):
    keep = (gt >= 0) & ~np.isin(gt, ignore)
    cm = confusion_matrix(gt[keep], pred[keep], labels=np.arange(c))
    tp = np.diag(cm).astype(float)
    fn, fp = cm.sum(1) - tp, cm.sum(0) - tp
    den = tp + fp + fn
    ok = (den > 0) & ~np.isin(np.arange(c), ignore)
    iou, acc = tp[ok] / den[ok], tp[ok] / np.maximum(tp[ok] + fn[ok], 1e-6)
    w = (tp + fn)[ok]
    return {"miou": iou.mean(), "macc": acc.mean(), "fmiou": (iou * w).sum() / w.sum()}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_references(seed):
    rng = np.random.default_rng(seed)
    c = 12
    gt = rng.integers(-1, c, 5000)
    pred = np.where(rng.random(5000) < 0.6, np.clip(gt, 0, None), rng.integers(0, c, 5000))
    ignore = [11]
    ours = metrics(confusion(gt, pred, c, ignore), ignore)
    ref, sk = ovo_metrics(gt, pred, c, ignore), sk_metrics(gt, pred, c, ignore)
    for k in ("miou", "macc", "fmiou"):
        assert abs(getattr(ours, k) - ref[k]) <= 1e-12
        assert abs(getattr(ours, k) - sk[k]) <= 1e-12


def test_perfect():
    gt = np.repeat(np.arange(5), 10)
    s = metrics(confusion(gt, gt, 6))
    assert (s.miou, s.macc, s.fmiou) == (1.0, 1.0, 1.0)


def test_false_positive_class_counts_as_zero():
    """A class absent from GT but predicted gets IoU 0 (OVO semantics), not nan."""
    gt = np.array([0, 0, 1, 1])
    pred = np.array([0, 2, 1, 1])
    s = metrics(confusion(gt, pred, 3))
    assert s.valid.tolist() == [True, True, True]
    assert abs(s.miou - (0.5 + 1.0 + 0.0) / 3) < 1e-12


def test_transfer_identity_and_mode():
    pts = np.random.default_rng(0).random((200, 3))
    lab = np.arange(200) % 7
    assert (transfer_labels(pts, lab, pts, k=1) == lab).all()
    p = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0.02, 0, 0], [0, 0.02, 0], [5, 5, 5.0]])
    # 5-NN mode: three neighbours labelled 4, two labelled 1
    assert transfer_labels(p, np.array([4, 4, 4, 1, 1, 9]), np.zeros((1, 3)), k=5)[0] == 4
    # 2-2-1 tie -> smallest label
    assert transfer_labels(p, np.array([4, 4, 1, 1, 7, 9]), np.zeros((1, 3)), k=5)[0] == 1
    # unassigned (-1) points are dropped before the search
    assert transfer_labels(p, np.array([-1, -1, -1, 2, 2, 2]), np.zeros((1, 3)), k=5)[0] == 2
