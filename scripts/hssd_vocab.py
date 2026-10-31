"""HSSD label vocabulary over the 35 frozen scenes under the spatialmem/hssd/labels.py rule (no Habitat).
Per instance placement; decomposed parts take their parent's main_category first (as group_label does).
  python scripts/hssd_vocab.py --root <data/hssd-hab> --scenes configs/hssd_scenes.yaml
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spatialmem.hssd.labels import EXCLUDED, label  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--scenes", type=Path, required=True)
    a = ap.parse_args()
    R = a.root
    main_cat = {r["id"]: (r["main_category"], r["wnsynsetkey"]) for r in csv.DictReader(open(R / "semantics" / "objects.csv", encoding="utf-8"))}
    lex = {int(c["id"]): c["name"] for c in json.loads((R / "semantics" / "hssd-hab_semantic_lexicon.json").read_text())["classes"]}
    idx = {p.name.split(".")[0]: p for p in (R / "objects").rglob("*.object_config.json")}
    d = yaml.safe_load(a.scenes.read_text())
    cnt, src, missing = collections.Counter(), collections.Counter(), 0
    for s in d["dev"] + d["eval"]:
        for o in json.loads((R / "scenes" / f"{s}.scene_instance.json").read_text())["object_instances"]:
            t = o["template_name"].split("/")[-1]
            if t not in idx:
                missing += 1
                continue
            cfg = json.loads(idx[t].read_text())
            par = t.split("_part_")[0] if "_part_" in t else None
            mc, syn = main_cat.get(par or t, ("", ""))
            rec = {"category_main": mc, "category_lexicon": lex.get(int(cfg.get("semantic_id", -1)), ""), "wnsynset": syn}
            lb = label(rec)
            cnt[lb] += 1
            src["main" if mc and mc not in EXCLUDED else "lexicon" if rec["category_lexicon"] and rec["category_lexicon"] not in EXCLUDED
                else "synset" if syn else "undefined"] += 1
    n = sum(cnt.values())
    print(f"instances {n} (+{missing} without a config); label source {dict(src)}; {len(cnt)} labels")
    print(" ".join(f"{k}:{v}" for k, v in cnt.most_common()))


if __name__ == "__main__":
    main()
