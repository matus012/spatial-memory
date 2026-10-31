"""G4 schema check (the 005 check, as a module): labels in the vocabulary, finite AABB with min <= max, unique object
ids, relation ids exist, Replica sem points / labels aligned with labels in [-1, 51) and sem_obj.npy aligned.

  uv run python -m spatialmem.ours.g4_check TAG [TAG ...]     (data/baselines/<tag>/{replica,hssd,replica_dev}/*.json)
"""
from __future__ import annotations

import sys

import numpy as np

from spatialmem.baselines.ovo import vocabulary
from spatialmem.eval.contract import Prediction
from spatialmem.eval.score_baselines import PRED


def check(tag: str) -> tuple[int, list[str]]:
    n, bad = 0, []
    for ds, vocab in (("replica", "replica"), ("replica_dev", "replica"), ("hssd", "hssd")):
        voc = set(vocabulary(vocab))
        for f in sorted((PRED / tag / ds).glob("*.json")):
            if f.name.endswith(".stats.json"):
                continue
            n += 1
            p, err = Prediction.from_json(f), []
            ids = [o.id for o in p.objects]
            err += [f"label {o.label}" for o in p.objects if o.label not in voc]
            err += [f"aabb {o.id}" for o in p.objects
                    if not (np.isfinite(o.aabb_min).all() and np.isfinite(o.aabb_max).all() and (o.aabb_min <= o.aabb_max).all())]
            if len(set(ids)) != len(ids):
                err.append("duplicate ids")
            err += [f"relation {r}" for r in p.relations if r.subj not in set(ids) or r.obj not in set(ids)]
            if ds != "hssd":
                if p.sem_points is None or len(p.sem_points) != len(p.sem_labels):
                    err.append("sem arrays not aligned")
                elif len(p.sem_labels) and not (p.sem_labels.min() >= -1 and p.sem_labels.max() < 51):
                    err.append("sem labels out of range")
                obj = f.with_name(f.stem + ".sem_obj.npy")
                if obj.exists() and len(np.load(obj)) != len(p.sem_points):
                    err.append("sem_obj not aligned")
            if err:
                bad.append(f"{tag}/{ds}/{f.stem}: {err[:5]}")
    return n, bad


def main() -> None:
    tot = 0
    for t in sys.argv[1:]:
        n, bad = check(t)
        tot += n
        print(f"{t}: {n - len(bad)}/{n} schema-valid", *bad, sep="\n  " if bad else "")
    print(f"total checked {tot}")


if __name__ == "__main__":
    main()
