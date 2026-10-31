"""Scorer self-check on real Replica GT (task 001 gates G2, G3, G4).

G2  GT-as-prediction: object F1@0.25/0.5, label acc, relation F1 (relations recomputed from the predicted
    boxes, not copied), semantic mIoU/mAcc/f-mIoU (labels on the GT vertices) — all must be 1.000.
    Also reported, not gated: the protocol's k=5 transfer applied to GT-as-pred (ceiling of the protocol).
G3  analytic perturbations on the real object sets: drop k, relabel a fraction, shift by a fraction of
    each box's own x-extent (IoU = (1 - s) / (1 + s)).
G4  semantic metrics on GT-as-pred and on a 30 % random relabel (seed 0) vs OVO's own metric code and
    an sklearn confusion-matrix implementation, per scene and pooled; |diff| <= 1e-6.

Run: uv run python -m spatialmem.eval.selfcheck --all [--out file.json]      (protocol 8)
     uv run python -m spatialmem.eval.selfcheck --dev [--out file.json]      (Replica-dev, task 006)
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from sklearn.metrics import confusion_matrix

from spatialmem.eval.contract import PredObject, Prediction
from spatialmem.eval.objects import score_objects, score_relations
from spatialmem.eval.reference import ovo_metrics
from spatialmem.eval.semantic import confusion, metrics, score_semantic
from spatialmem.gt.relations import compute_relations
from spatialmem.gt.replica import DEV_SCENES, ROOT, SCENES, load_gt, protocol

TOL = 1e-6


def naive_iou(a_lo, a_hi, b_lo, b_hi, min_extent: float = 0.02) -> float:
    """Independent per-pair IoU (plain Python loops), same padding rule as the contract."""
    def pad(lo, hi):
        lo, hi = list(lo), list(hi)
        for k in range(3):
            g = max(0.0, min_extent - (hi[k] - lo[k])) / 2
            lo[k] -= g
            hi[k] += g
        return lo, hi
    (a_lo, a_hi), (b_lo, b_hi) = pad(a_lo, a_hi), pad(b_lo, b_hi)
    inter = va = vb = 1.0
    for k in range(3):
        inter *= max(0.0, min(a_hi[k], b_hi[k]) - max(a_lo[k], b_lo[k]))
        va *= a_hi[k] - a_lo[k]
        vb *= b_hi[k] - b_lo[k]
    u = va + vb - inter
    return inter / u if u > 0 else 0.0


def brute_max_matches(pred: Prediction, gt: Prediction, tau: float) -> int:
    """Max-cardinality count of pairs with IoU >= tau, independent of objects.match (bipartite matching)."""
    elig = np.array([[naive_iou(p.aabb_min, p.aabb_max, g.aabb_min, g.aabb_max) >= tau for g in gt.objects]
                     for p in pred.objects], dtype=bool)
    if not elig.any():
        return 0
    m = maximum_bipartite_matching(csr_matrix(elig.astype(np.int8)), perm_type="column")
    return int((m >= 0).sum())


def sk_metrics(gt: np.ndarray, pred: np.ndarray, c: int) -> dict:
    keep = gt >= 0
    cm = confusion_matrix(gt[keep], pred[keep], labels=np.arange(c))
    return _from_cm(cm)


def _from_cm(cm: np.ndarray) -> dict:
    tp = np.diag(cm).astype(float)
    fn, fp = cm.sum(1) - tp, cm.sum(0) - tp
    den = tp + fp + fn
    ok = den > 0
    iou, acc = tp[ok] / den[ok], tp[ok] / np.maximum(tp[ok] + fn[ok], 1e-6)
    w = (tp + fn)[ok]
    return {"miou": float(iou.mean()), "macc": float(acc.mean()), "fmiou": float((iou * w).sum() / w.sum())}


def check(name: str, ok: bool, value, expected, out: list) -> None:
    out.append({"check": name, "ok": bool(ok), "value": value, "expected": expected})


def scene_check(scene: str, up: int, rng: np.random.Generator) -> tuple[list, dict]:
    res: list = []
    gt = load_gt(scene)
    c = protocol().num_classes
    n = len(gt.objects)

    # ---- G2 objects / relations
    pred = Prediction(scene, [PredObject(o.id, o.label, o.aabb_min.copy(), o.aabb_max.copy()) for o in gt.objects])
    pred.relations = compute_relations(pred.objects, up=up)
    for tau in (0.25, 0.5):
        s = score_objects(pred, gt, tau)
        check(f"G2 {scene} F1@{tau}", s.f1 == 1.0, s.f1, 1.0, res)
        check(f"G2 {scene} label_acc@{tau}", s.label_acc == 1.0, s.label_acc, 1.0, res)
    r = score_relations(pred, gt, score_objects(pred, gt, 0.25))
    check(f"G2 {scene} relation F1 (n={r.n_gt})", r.f1 == 1.0, r.f1, 1.0, res)

    # ---- G2 / G4 semantic, GT-as-pred on the GT vertices
    lab = gt.vertex_labels
    pred_lab = np.where(lab >= 0, lab, 0)  # ignored GT vertices are dropped by the metric anyway
    cm_gt = confusion(lab, pred_lab, c)
    sm = metrics(cm_gt)
    for k in ("miou", "macc", "fmiou"):
        check(f"G2 {scene} {k}", getattr(sm, k) == 1.0, getattr(sm, k), 1.0, res)
    ref, sk = ovo_metrics(lab, pred_lab, c, []), sk_metrics(lab, pred_lab, c)
    for k in ("miou", "macc", "fmiou"):
        d = max(abs(getattr(sm, k) - ref[k]), abs(getattr(sm, k) - sk[k]))
        check(f"G4 {scene} GT-as-pred {k} vs OVO/sklearn", d <= TOL, d, f"<= {TOL}", res)

    # ---- G4 noisy: 30 % of vertices relabelled uniformly at random (seeded per scene)
    noisy = pred_lab.copy()
    flip = rng.random(len(noisy)) < 0.30
    noisy[flip] = rng.integers(0, c, int(flip.sum()))
    cm_noisy = confusion(lab, noisy, c)
    sn = metrics(cm_noisy)
    ref, sk = ovo_metrics(lab, noisy, c, []), sk_metrics(lab, noisy, c)
    for k in ("miou", "macc", "fmiou"):
        d = max(abs(getattr(sn, k) - ref[k]), abs(getattr(sn, k) - sk[k]))
        check(f"G4 {scene} noisy30 {k}={getattr(sn, k):.4f} vs OVO/sklearn", d <= TOL, d, f"<= {TOL}", res)

    # ---- protocol k=5 transfer ceiling (reported, not gated)
    keep = lab >= 0
    st, cm_t = score_semantic(Prediction(scene, [], sem_points=gt.vertices[keep], sem_labels=lab[keep]), gt, c)
    ceiling = {"miou": st.miou, "macc": st.macc, "fmiou": st.fmiou,
               "vertices_changed": float(1 - np.trace(cm_t) / cm_t.sum())}

    # ---- G3 on the real object set
    k_drop = max(1, n // 10)
    drop = set(rng.choice(n, k_drop, replace=False).tolist())
    p_drop = Prediction(scene, [o for i, o in enumerate(pred.objects) if i not in drop])
    for tau in (0.25, 0.5):
        s = score_objects(p_drop, gt, tau)
        check(f"G3 {scene} drop {k_drop}/{n} recall@{tau}",
              s.tp == n - k_drop and s.n_gt == n and s.n_pred == n - k_drop,
              [s.recall, s.precision], [(n - k_drop) / n, 1.0], res)
    m_rel = round(0.2 * n)
    rel = set(rng.choice(n, m_rel, replace=False).tolist())
    p_rel = Prediction(scene, [PredObject(o.id, "__wrong__" if i in rel else o.label, o.aabb_min, o.aabb_max)
                               for i, o in enumerate(pred.objects)])
    s = score_objects(p_rel, gt, 0.25)
    check(f"G3 {scene} relabel {m_rel}/{n} label_acc", s.tp == n and s.n_label_correct == n - m_rel,
          [s.tp, s.n_label_correct, s.label_acc], [n, n - m_rel, 1 - m_rel / n], res)
    for frac in (0.2, 0.5):
        iou_self = (1 - frac) / (1 + frac)
        objs = []
        for o in pred.objects:
            d = np.zeros(3)
            d[0] = frac * (o.aabb_max[0] - o.aabb_min[0])
            objs.append(PredObject(o.id, o.label, o.aabb_min + d, o.aabb_max + d))
        p_sh = Prediction(scene, objs)
        for tau in (0.25, 0.5):
            s = score_objects(p_sh, gt, tau)
            # closed form: each self pair has IoU (1 - s)/(1 + s) (>= that for boxes < 2 cm along x, which are
            # padded) -> if that clears tau all n self pairs are eligible and max cardinality is exactly n.
            # Either way the count must equal an independent brute-force max matching on a naive IoU.
            expect = brute_max_matches(p_sh, gt, tau)
            n_self = sum(i == j for i, j, _ in s.matches)
            ok = s.tp == expect and (s.tp == n if iou_self >= tau else True)
            check(f"G3 {scene} shift {frac} (self IoU {iou_self:.3f}) TP@{tau} = brute-force max matching",
                  ok, [s.tp, n_self], [expect, n if iou_self >= tau else "brute"], res)
    return res, {"lab": lab, "pred_gt": pred_lab, "pred_noisy": noisy, "cm_gt": cm_gt, "cm_noisy": cm_noisy, "ceiling": ceiling, "n_objects": n,
                 "n_relations": len(gt.relations)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dev", action="store_true", help="the Replica-dev scenes (configs/replica_dev.yaml, task 006)")
    ap.add_argument("--scenes", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    scenes = SCENES if a.all else DEV_SCENES if a.dev else tuple(a.scenes)
    rng = np.random.default_rng(0)
    all_res, cm_gt, cm_noisy, summary = [], 0, 0, {}
    cat: dict[str, list] = {"lab": [], "gt": [], "noisy30": []}
    for s in scenes:
        up = json.loads((ROOT / "gt" / f"{s}.json").read_text())["up_axis"]
        res, extra = scene_check(s, up, rng)
        all_res += res
        cm_gt = cm_gt + extra["cm_gt"]
        cm_noisy = cm_noisy + extra["cm_noisy"]
        cat["lab"].append(extra["lab"])
        cat["gt"].append(extra["pred_gt"])
        cat["noisy30"].append(extra["pred_noisy"])
        summary[s] = {"objects": extra["n_objects"], "relations": extra["n_relations"],
                      "k5_transfer_ceiling": extra["ceiling"]}
        bad = [r for r in res if not r["ok"]]
        print(f"{s}: {len(res) - len(bad)}/{len(res)} ok  objects={extra['n_objects']} "
              f"relations={extra['n_relations']}  k5-ceiling mIoU={extra['ceiling']['miou']:.4f}")
        for r in bad:
            print("   FAIL", r)
    # pooled (headline aggregation of the protocol)
    pooled = {}
    for tag, cm in (("gt", cm_gt), ("noisy30", cm_noisy)):
        ours = metrics(np.asarray(cm))
        ref = _from_cm(np.asarray(cm))
        pooled[tag] = {k: getattr(ours, k) for k in ("miou", "macc", "fmiou")}
        for k in ("miou", "macc", "fmiou"):
            d = abs(getattr(ours, k) - ref[k])
            if tag == "gt":
                check(f"G2 pooled {k}", getattr(ours, k) == 1.0, getattr(ours, k), 1.0, all_res)
            check(f"G4 pooled {tag} {k}={getattr(ours, k):.4f} vs sklearn-form", d <= TOL, d, f"<= {TOL}", all_res)
        # vendored OVO code + sklearn on the CONCATENATED labels of all scenes (not the summed confusion matrices)
        lab_c, pred_c = np.concatenate(cat["lab"]), np.concatenate(cat[tag])
        c = protocol().num_classes
        ref_o, ref_s = ovo_metrics(lab_c, pred_c, c, []), sk_metrics(lab_c, pred_c, c)
        for k in ("miou", "macc", "fmiou"):
            d = max(abs(getattr(ours, k) - ref_o[k]), abs(getattr(ours, k) - ref_s[k]))
            check(f"G4 pooled {tag} {k} vs OVO/sklearn (concatenated)", d <= TOL, d, f"<= {TOL}", all_res)
    n_bad = sum(not r["ok"] for r in all_res)
    print(f"pooled: GT-as-pred {pooled['gt']}  noisy30 {pooled['noisy30']}")
    print(f"TOTAL {len(all_res) - n_bad}/{len(all_res)} checks ok")
    if a.out:
        def clean(x):
            if isinstance(x, float) and math.isnan(x):
                return None
            return x
        json.dump({"checks": [{k: clean(v) for k, v in r.items()} for r in all_res], "summary": summary,
                   "pooled": pooled}, open(a.out, "w"), indent=1, default=float)
    raise SystemExit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
