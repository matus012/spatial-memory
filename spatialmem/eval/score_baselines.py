"""Task 003 step 6: score baseline predictions with the frozen 001 scorer (unchanged).

Inputs: data/baselines/<method>/{replica,hssd,replica_dev}/<scene>.json (CONTRACT Prediction + .sem.npz sidecar).
replica_dev (task 006) = the Replica-dev scenes (configs/replica_dev.yaml), scored and pooled exactly like replica.
GT: Replica gt/<scene>.json + protocol vertices (spatialmem.gt.replica.load_gt); HSSD data/hssd_runs/all/<scene>/gt.json
(canonical labels, spatialmem.hssd.gt.load_hssd_gt). HSSD predicted labels are canonicalised too (CONTRACT 2026-10-27e).

Per scene + pooled (sum of TP / #pred / #GT over scenes): object P / R / F1 @ 0.25 and @ 0.5, label acc, relation
P / R / F1 (@ 0.25 matching). Replica: mIoU / mAcc / f-mIoU (pooled confusion, k=5 transfer) beside the protocol's
k=5 ceiling. Label confusions among matched objects @ 0.25 (top-10 pred != gt pairs).

  uv run python -m spatialmem.eval.score_baselines --methods ovo cg --out <dir>
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import yaml

from spatialmem.eval.contract import Prediction
from spatialmem.eval.objects import _f1, score_objects, score_relations
from spatialmem.eval.semantic import metrics, pooled, score_semantic
from spatialmem.gt.replica import DEV_SCENES, ROOT, SCENES, load_gt, protocol
from spatialmem.hssd.gt import load_hssd_gt
from spatialmem.hssd.vocab import canonical

PRED = ROOT / "data" / "baselines"
HSSD_GT = ROOT / "data" / "hssd_runs" / "all"
TAUS = (0.25, 0.5)


def scene_lists() -> dict[str, list[str]]:
    d = yaml.safe_load((ROOT / "configs" / "hssd_scenes.yaml").read_text())
    return {"replica": list(SCENES), "hssd_eval": d["eval"], "hssd_dev": d["dev"], "replica_dev": list(DEV_SCENES)}


def load_pred(method: str, dataset: str, scene: str, hssd: bool) -> Prediction:
    p = Prediction.from_json(PRED / method / dataset / f"{scene}.json")
    if hssd:
        for o in p.objects:
            o.label = canonical(o.label)
    return p


def score_scene(pred: Prediction, gt: Prediction) -> dict:
    r = {"n_pred": len(pred.objects), "n_gt": len(gt.objects)}
    for tau in TAUS:
        s = score_objects(pred, gt, tau)
        r[f"obj@{tau}"] = {"tp": s.tp, "n_pred": s.n_pred, "n_gt": s.n_gt, "p": s.precision, "r": s.recall, "f1": s.f1,
                           "label_correct": s.n_label_correct, "label_acc": s.label_acc}
    s25 = score_objects(pred, gt, 0.25)
    rel = score_relations(pred, gt, s25)
    r["rel"] = {"tp": rel.tp, "n_pred": rel.n_pred, "n_gt": rel.n_gt, "p": rel.precision, "r": rel.recall, "f1": rel.f1}
    pl = {o.id: o.label for o in pred.objects}
    gl = {o.id: o.label for o in gt.objects}
    r["confusions"] = [(gl[g], pl[p]) for p, g, _ in s25.matches if pl[p] != gl[g]]
    return r


def pool(rows: list[dict]) -> dict:
    out = {}
    for tau in TAUS:
        k = f"obj@{tau}"
        tp, npr, ngt, lc = (sum(r[k][x] for r in rows) for x in ("tp", "n_pred", "n_gt", "label_correct"))
        p, rc = tp / max(npr, 1), tp / max(ngt, 1)
        out[k] = {"tp": tp, "n_pred": npr, "n_gt": ngt, "p": p, "r": rc, "f1": _f1(p, rc),
                  "label_acc": lc / tp if tp else float("nan")}
    tp, npr, ngt = (sum(r["rel"][x] for r in rows) for x in ("tp", "n_pred", "n_gt"))
    p, rc = tp / max(npr, 1), tp / max(ngt, 1)
    out["rel"] = {"tp": tp, "n_pred": npr, "n_gt": ngt, "p": p, "r": rc, "f1": _f1(p, rc)}
    out["mean_scene_f1@0.25"] = float(np.mean([r["obj@0.25"]["f1"] for r in rows]))
    return out


def run_replica(method: str, split: str, scenes: list[str]) -> dict:
    """Replica-format split (protocol 8 or Replica-dev): objects + relations + semantic (pooled, k=5 ceiling beside)."""
    proto = protocol()
    rows, cms, cms_ceiling = {}, [], []
    for s in scenes:
        f = PRED / method / split / f"{s}.json"
        if not f.exists():
            continue
        gt = load_gt(s)
        pred = load_pred(method, split, s, hssd=False)
        rows[s] = score_scene(pred, gt)
        if pred.sem_points is not None and len(pred.sem_points):
            sc, cm = score_semantic(pred, gt, proto.num_classes)
            cms.append(cm)
            rows[s]["sem"] = {"miou": sc.miou, "macc": sc.macc, "fmiou": sc.fmiou}
            keep = gt.vertex_labels >= 0
            ce, cmc = score_semantic(Prediction(s, [], sem_points=gt.vertices[keep], sem_labels=gt.vertex_labels[keep]), gt, proto.num_classes)
            cms_ceiling.append(cmc)
            rows[s]["sem_k5_ceiling"] = {"miou": ce.miou, "macc": ce.macc, "fmiou": ce.fmiou}
    out = {"scenes": rows, "pooled": pool(list(rows.values())) if rows else None}
    if cms:
        ps, pc = pooled(cms), pooled(cms_ceiling)
        out["sem_pooled"] = {"miou": ps.miou, "macc": ps.macc, "fmiou": ps.fmiou, "n_scenes": len(cms)}
        out["sem_k5_ceiling_pooled"] = {"miou": pc.miou, "macc": pc.macc, "fmiou": pc.fmiou}
    return out


def run(method: str) -> dict:
    lists = scene_lists()
    res: dict = {"method": method}
    res["replica"] = run_replica(method, "replica", lists["replica"])
    # HSSD: objects + relations
    for split in ("hssd_eval", "hssd_dev"):
        rows = {}
        for s in lists[split]:
            f = PRED / method / "hssd" / f"{s}.json"
            if not f.exists():
                continue
            gt, _ = load_hssd_gt(HSSD_GT / s / "gt.json")
            rows[s] = score_scene(load_pred(method, "hssd", s, hssd=True), gt)
        res[split] = {"scenes": rows, "pooled": pool(list(rows.values())) if rows else None, "n_scenes": len(rows)}
    res["replica_dev"] = run_replica(method, "replica_dev", lists["replica_dev"])
    for split in ("replica", "hssd_eval", "replica_dev"):
        c = collections.Counter(x for r in res[split]["scenes"].values() for x in map(tuple, r["confusions"]))
        res[split]["confusion_top10"] = [{"gt": g, "pred": p, "n": n} for (g, p), n in c.most_common(10)]
    return res


def ovo_own_eval_check(own_dir: Path) -> dict:
    """G3: our metrics vs OVO's vendored reference on OVO's own per-vertex predictions (<scene>.txt, reduced ids)."""
    from spatialmem.eval.reference import ovo_metrics
    from spatialmem.eval.semantic import confusion
    proto = protocol()
    gts, prs = [], []
    for s in SCENES:
        f = own_dir / f"{s}.txt"
        if not f.exists():
            continue
        gt = load_gt(s)
        pr = np.loadtxt(f, dtype=np.int64)
        assert len(pr) == len(gt.vertex_labels), (s, len(pr), len(gt.vertex_labels))
        keep = gt.vertex_labels >= 0
        gts.append(gt.vertex_labels[keep])
        prs.append(np.where(pr[keep] >= 0, pr[keep], proto.num_classes))   # unassigned -> the ignore index
    g, p = np.concatenate(gts), np.concatenate(prs)
    ours = metrics(confusion(g, p, proto.num_classes + 1), ignore=[proto.num_classes])
    ref = ovo_metrics(g, p, proto.num_classes + 1, [proto.num_classes])
    return {"n_scenes": len(gts), "ours": {"miou": ours.miou, "macc": ours.macc, "fmiou": ours.fmiou}, "ovo_reference": ref}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["ovo", "cg"])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for m in a.methods:
        r = run(m)
        if m == "ovo" and (PRED / "ovo" / "replica_own").exists():
            r["G3_own_eval_check"] = ovo_own_eval_check(PRED / "ovo" / "replica_own")
        (a.out / f"scores_{m}.json").write_text(json.dumps(r, indent=1, default=float))
        for split in ("replica", "hssd_eval", "hssd_dev", "replica_dev"):
            pp = r[split]["pooled"]
            if pp:
                print(f"{m} {split}: n={len(r[split]['scenes'])} F1@.25 {pp['obj@0.25']['f1']:.3f} F1@.5 {pp['obj@0.5']['f1']:.3f} "
                      f"labelacc@.25 {pp['obj@0.25']['label_acc']:.3f} relF1 {pp['rel']['f1']:.3f}")
        for split in ("replica", "replica_dev"):
            if r[split].get("sem_pooled"):
                print(f"{m} {split} semantic pooled {r[split]['sem_pooled']} (k5 ceiling {r[split]['sem_k5_ceiling_pooled']})")


if __name__ == "__main__":
    main()
