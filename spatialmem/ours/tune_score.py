"""Task 006 tuning rows: score "ours" tags on Replica-dev (pooled semantic, OVO protocol) + HSSD dev 5 (objects) with the
frozen scorer (`spatialmem.eval.score_baselines`, unchanged) and write one JSONL row per tag (configs/ours_v2_objective.md).

  uv run python -m spatialmem.ours.tune_score --tags T [T ...] --out DIR

Row: config knobs (from the tag's run_meta), Replica-dev pooled mIoU / mAcc / f-mIoU + per-scene mean mIoU, HSSD dev 5
mean per-scene F1@0.25 (eligibility >= 0.351, OVO), pooled F1@0.25 / @0.5, label acc (pooled, @0.25), relation F1,
object__outside share (HSSD dev predictions). Full per-scene scores -> DIR/<tag>.score.json.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from spatialmem.eval.score_baselines import PRED, HSSD_GT, load_pred, pool, run_replica, scene_lists, score_scene
from spatialmem.hssd.gt import load_hssd_gt

ELIGIBLE_F1 = 0.351          # configs/ours_v2_objective.md: HSSD dev 5 mean F1@0.25 >= OVO
KNOBS = ("encoder", "descriptor", "label", "tau", "tau_c", "lam", "tau_g", "top_k", "min_detections", "stride")


def score_tag(tag: str) -> tuple[dict, dict]:
    lists = scene_lists()
    metas = sorted((PRED / tag).glob("run_meta*.json")) + sorted((PRED / tag / "hssd").glob("run_meta*.json"))
    cfgs = {json.dumps({k: json.loads(m.read_text())["config"].get(k) for k in KNOBS}, sort_keys=True) for m in metas}
    assert len(cfgs) == 1, (tag, cfgs)          # every scene of a tag ran the same config
    rd = run_replica(tag, "replica_dev", lists["replica_dev"])
    rows, labels = {}, collections.Counter()
    for s in lists["hssd_dev"]:
        gt, _ = load_hssd_gt(HSSD_GT / s / "gt.json")
        pred = load_pred(tag, "hssd", s, hssd=True)
        rows[s] = score_scene(pred, gt)
        labels.update(o.label for o in pred.objects)
    hp = pool(list(rows.values()))
    n_obj = sum(labels.values())
    sem = rd["sem_pooled"]
    row = {"tag": tag, **json.loads(cfgs.pop()),
           "replica_dev_n_scenes": sem["n_scenes"], "replica_dev_miou": sem["miou"], "replica_dev_macc": sem["macc"],
           "replica_dev_fmiou": sem["fmiou"],
           "replica_dev_scene_mean_miou": float(np.mean([r["sem"]["miou"] for r in rd["scenes"].values()])),
           "replica_dev_k5_ceiling_miou": rd["sem_k5_ceiling_pooled"]["miou"],
           "hssd_dev_n_scenes": len(rows), "hssd_dev_mean_f1@0.25": hp["mean_scene_f1@0.25"],
           "hssd_dev_pooled_f1@0.25": hp["obj@0.25"]["f1"], "hssd_dev_pooled_f1@0.5": hp["obj@0.5"]["f1"],
           "hssd_dev_label_acc@0.25": hp["obj@0.25"]["label_acc"], "hssd_dev_rel_f1": hp["rel"]["f1"],
           "hssd_dev_object__outside_share": labels.get("object__outside", 0) / max(n_obj, 1),
           "eligible": hp["mean_scene_f1@0.25"] >= ELIGIBLE_F1}
    return row, {"replica_dev": rd, "hssd_dev": {"scenes": rows, "pooled": hp}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for t in a.tags:
        row, full = score_tag(t)
        (a.out / f"{t}.score.json").write_text(json.dumps(full, indent=1, default=float))
        with open(a.out / "rows.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
