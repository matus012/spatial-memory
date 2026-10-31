"""Task 002 G5 (HSSD part): GT-as-predictions on every HSSD scene GT (gt.json from spatialmem.hssd.run).

Per scene, on canonical labels (configs/hssd_synonyms.yaml): object F1 = label acc = 1.000 at tau 0.25 / 0.5, relation
F1 = 1.000, and the stored relations (exported on raw labels) equal `compute_relations(objects, up=1)` recomputed here
on canonical labels (relations v2) — i.e. canonicalisation does not change any relation. The Replica half of G5 is
`spatialmem.eval.selfcheck --all`.

Run: uv run python -m spatialmem.eval.selfcheck_hssd --root data/hssd_runs/all [--out file.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from spatialmem.eval.objects import score_objects, score_relations
from spatialmem.gt.relations import compute_relations
from spatialmem.hssd.gt import load_hssd_gt


def check_scene(gt_json: Path) -> list[dict]:
    gt, up = load_hssd_gt(gt_json)   # canonical labels (CONTRACT 2026-10-27e); relations as exported on raw labels
    d = {"scene": gt.scene, "up_axis": up}
    objs, rels = gt.objects, gt.relations
    out = []
    for tau in (0.25, 0.5):
        s = score_objects(gt, gt, tau)
        out += [{"check": f"{d['scene']} F1@{tau}", "value": s.f1}, {"check": f"{d['scene']} label_acc@{tau}", "value": s.label_acc}]
    r = score_relations(gt, gt, score_objects(gt, gt, 0.25))
    out.append({"check": f"{d['scene']} relation F1 (n={len(rels)})", "value": r.f1 if rels else 1.0})
    again = {x.key() for x in compute_relations(objs, up=d["up_axis"])}
    out.append({"check": f"{d['scene']} stored relations == recomputed on canonical labels (v2)",
                "value": float(again == {x.key() for x in rels})})
    for c in out:
        c["ok"] = c["value"] == 1.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    res = []
    for g in sorted(a.root.glob("*/gt.json")):
        c = check_scene(g)
        res += c
        print(f"{g.parent.name}: {sum(x['ok'] for x in c)}/{len(c)} ok")
    n_ok = sum(x["ok"] for x in res)
    print(f"TOTAL {n_ok}/{len(res)} checks ok over {len(res) // 6} scenes")
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))
    raise SystemExit(0 if n_ok == len(res) else 1)


if __name__ == "__main__":
    main()
