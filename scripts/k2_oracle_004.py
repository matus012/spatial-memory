"""Task 004 K2 diagnostic D4 (substitute-the-suspect-input): keep ours' objects and points, replace the labels by GT.
object oracle = each object takes the majority GT class of its points (nearest labelled GT vertex);
point oracle  = each predicted point takes its nearest GT vertex label (ceiling of our point set + k5 transfer).
Scored with the unchanged protocol path (score_semantic + pooled)."""
from __future__ import annotations

import sys

import numpy as np
from scipy.spatial import cKDTree

from spatialmem.eval.contract import Prediction
from spatialmem.eval.score_baselines import PRED, load_gt, load_pred, protocol
from spatialmem.eval.semantic import pooled, score_semantic

SCENES = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]


def main(tag: str) -> None:
    nc = protocol().num_classes
    cms = {"as_predicted": [], "object_oracle": [], "point_oracle": []}
    for s in SCENES:
        gt, p = load_gt(s), load_pred(tag, "replica", s, hssd=False)
        obj = np.load(PRED / tag / "replica" / f"{s}.sem_obj.npy")
        pts, lab = np.asarray(p.sem_points, float), np.asarray(p.sem_labels, np.int64)
        assert len(obj) == len(pts)
        keep = gt.vertex_labels >= 0
        _, i = cKDTree(gt.vertices[keep]).query(pts, k=1)
        nn = gt.vertex_labels[keep][i]
        maj = np.zeros(obj.max() + 1, np.int64)
        for k in range(obj.max() + 1):
            maj[k] = np.bincount(nn[obj == k]).argmax()
        for name, l in [("as_predicted", lab), ("object_oracle", maj[obj]), ("point_oracle", nn)]:
            sc, cm = score_semantic(Prediction(s, [], sem_points=pts, sem_labels=l), gt, nc)
            cms[name].append(cm)
            print(f"{s:8s} {name:14s} mIoU {sc.miou*100:6.2f} mAcc {sc.macc*100:6.2f} f-mIoU {sc.fmiou*100:6.2f}")
    for name, c in cms.items():
        sc = pooled(c)
        print(f"POOLED   {name:14s} mIoU {sc.miou*100:6.2f} mAcc {sc.macc*100:6.2f} f-mIoU {sc.fmiou*100:6.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ours_v1_diag")
