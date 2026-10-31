"""Collect chunk raw outputs (runner format: qid,row,model,raw_output,...) into per-variant results (find rule label+dist).
`python -m spatialmem.p1b.collect --raw chunk_0_raw.jsonl ... [--prompts-dir data/p1b/sweep] [--out results/p1b]`."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from spatialmem.p1b.questions import read
from spatialmem.p1b.score import score_row, summarise
from spatialmem.p1b.jsonio import dump
from spatialmem.p1b.variants import RES, SWEEP

RULE = "label+dist"


def collect(raw_files: list[Path], prompts_dir: Path, out: Path) -> dict:
    rows: dict[str, dict[str, str]] = defaultdict(dict)
    for f in raw_files:
        for ln in f.read_text().splitlines():
            d = json.loads(ln)
            assert d["qid"] not in rows[d["row"]], f"duplicate {d['row']} {d['qid']}"
            rows[d["row"]][d["qid"]] = d["raw_output"]
    qs = read()
    runs = {}
    for v in sorted(rows):
        prompts = {d["qid"]: d for d in map(json.loads, (prompts_dir / f"prompts_{v}.jsonl").read_text().splitlines())}
        assert set(rows[v]) == set(prompts), f"{v}: raw qids != prompt qids ({len(rows[v])} vs {len(prompts)})"
        res = score_row(v, qs, rows[v], prompts, RULE)
        dump(out / "per_question" / f"{v}.json", sorted([r["qid"], r["type"], int(r["correct"]), int(r["invalid"])] for r in res))
        runs[v] = summarise(v, res)
        del runs[v]["row"]
    dump(out / "runs.json", runs)
    return runs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, nargs="+", required=True)
    ap.add_argument("--prompts-dir", type=Path, default=SWEEP)
    ap.add_argument("--out", type=Path, default=RES)
    a = ap.parse_args()
    for v, s in collect(a.raw, a.prompts_dir, a.out).items():
        print(f"{v:24s} {s['correct']:4d}/{s['n']}  {100 * s['accuracy']:5.1f} %  invalid {s['invalid']}")
