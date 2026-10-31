"""Repairs of a real map against GT (P1B_REPAIR §1) and the 21 repaired variants `<m>__<repair>`.
`uv run python -m spatialmem.p1b.repair`  (data/p1b/repair/, results/p1b/repair/)."""
from __future__ import annotations

import numpy as np

from spatialmem.eval.boxes import iou_matrix
from spatialmem.eval.contract import PredObject, Prediction
from spatialmem.eval.objects import match, without_structural
from spatialmem.gt.relations import compute_relations

KINDS = ("label", "extra", "miss", "box", "label+extra", "label+extra+miss", "all")
STEPS = {"label+extra": ("label", "extra"), "label+extra+miss": ("label", "extra", "miss"), "all": ("label", "extra", "miss", "box")}
TAU = 0.25
UP = 1


def repair(pred: Prediction, gt: Prediction, kind: str) -> Prediction:
    """Repaired copy of `pred`: matching once on the unrepaired map (scorer's IoU matrix + match @ 0.25), steps applied in
    order, relations recomputed by compute_relations(up=1). New GT-added ids: max predicted id + 1, + 2, ... in GT id order
    (max over the structural-filtered unrepaired map)."""
    steps = STEPS.get(kind, (kind,))
    assert all(s in ("label", "extra", "miss", "box") for s in steps), kind
    p, g = without_structural(pred), without_structural(gt)
    mt = match(iou_matrix(p.boxes(), g.boxes()), TAU)
    p2g = dict((i, j) for i, j, _ in mt)
    top = max((o.id for o in p.objects), default=0)
    objs = {i: PredObject(o.id, o.label, o.aabb_min.copy(), o.aabb_max.copy(), o.score) for i, o in enumerate(p.objects)}
    for st in steps:
        if st == "label":
            for i, j in p2g.items():
                objs[i].label = g.objects[j].label
        elif st == "extra":
            objs = {i: o for i, o in objs.items() if i in p2g}
        elif st == "box":
            for i, j in p2g.items():
                objs[i].aabb_min, objs[i].aabb_max = g.objects[j].aabb_min.copy(), g.objects[j].aabb_max.copy()
        else:
            seen = set(p2g.values())
            gone = sorted((o for j, o in enumerate(g.objects) if j not in seen), key=lambda o: o.id)
            for k, o in enumerate(gone):
                objs[len(p.objects) + k] = PredObject(top + 1 + k, o.label, o.aabb_min.copy(), o.aabb_max.copy())
    out = [o for _, o in sorted(objs.items())]
    return Prediction(pred.scene, out, compute_relations(out, up=UP))


def main() -> None:
    import sys
    from spatialmem.hssd.vocab import ROOT
    from spatialmem.p1b.degrade import REAL
    from spatialmem.p1b.jsonio import dump
    from spatialmem.p1b.variants import chunks, process
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial

    d, rd = ROOT / "data" / "p1b" / "repair", ROOT / "results" / "p1b" / "repair"
    d.mkdir(parents=True, exist_ok=True)
    names = sorted(f"{m}__{k}" for m in REAL for k in KINDS)
    with ProcessPoolExecutor(max_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 8) as ex:
        res = {r["name"]: r for r in ex.map(partial(process, out_dir=d), names)}
    for m in REAL:
        f = res[f"{m}__all"]["features"]
        assert all(round(f[k], 3) == 1.0 for k in ("f1", "label_acc", "rel_f1")), (m, f)
    feats = {v: res[v]["features"] for v in names}
    over = {v: res[v]["over"] for v in names}
    dump(rd / "features.json", feats)
    dump(rd / "overflow.json", over)
    man = chunks(names, d, 1, None)
    dump(d / "MANIFEST.json", man)
    print(f"{'variant':26s} {'F1':>6s} {'lab':>6s} {'rel':>6s}  {'F1post':>6s} {'labP':>6s} {'relP':>6s}  houses_over max_removed hit_rel hit_find")
    for v in names:
        f, o = feats[v], over[v]
        print(f"{v:26s} {f['f1']:6.3f} {f['label_acc']:6.3f} {f['rel_f1']:6.3f}  {f['post']['f1']:6.3f} {f['post']['label_acc']:6.3f} "
              f"{f['post']['rel_f1']:6.3f}  {sum(r['removed'] > 0 for r in o.values()):4d} {max(r['removed'] for r in o.values()):8d} "
              f"{sum(r['hit_rel'] for r in o.values()):6d} {sum(r['hit_find'] for r in o.values()):6d}")
    allr = [r for o in over.values() for r in o.values()]
    print("max tokens after rule:", max(r["tokens_after"] for r in allr), " exact max (houses > 29k):",
          max((r.get("tokens_after_max_all", 0) for r in allr), default=0), " before:", max(r["tokens"] for r in allr))
    print("chunk:", man["chunks"])


if __name__ == "__main__":
    main()
