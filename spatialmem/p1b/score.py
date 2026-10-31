"""Scoring (P1B_PROTOCOL §5). The given map / label list is recovered from the prompt the model saw.
`python -m spatialmem.p1b.score --row R --raw raw.jsonl --prompts prompts.jsonl --out DIR`."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spatialmem.p1b import parse as P
from spatialmem.p1b.gtmap import Map
from spatialmem.p1b.questions import OUT as QUESTIONS, TYPES, read
from spatialmem.p1b.serialise import parse_any as parse_map

FIND_TOL_M = 1.0
FIND_RULES = ("dist", "label+dist")   # "dist": 008 rule; "label+dist": 009 (P1B_SWEEP amendment) -- label == asked label A as well
IMG_ROWS = ("img7b", "img")
SEP = "\n\nQuestion: "
LIST_PRE = "one label from this list: "


def context(user_text: str) -> tuple[Map | None, list[str] | None]:
    """(map, None) for a map prompt, (None, vocabulary) for an images-only prompt."""
    if user_text.startswith("HOUSE "):
        return parse_map(user_text.split(SEP)[0]), None
    fmt = user_text.split("Answer format: ", 1)[1]
    return None, ([x.replace(" ", "_") for x in fmt[len(LIST_PRE):].split(", ")] if fmt.startswith(LIST_PRE) else None)


def score_one(q: dict, raw: str, m: Map | None, vocab: list[str] | None, find_rule: str = "dist") -> dict:
    """Parse `raw` for question `q` against the given map `m` (map rows) or label list `vocab` (images row)."""
    t = q["type"]
    if t == "existence":
        val = P.parse_existence(raw)
    elif t == "counting":
        val = P.parse_counting(raw)
    elif t == "relation":
        val = P.parse_relation(raw, sorted({o.label for o in m.objects}) if m is not None else vocab)
    else:
        val = P.parse_find(raw, {o.id for o in m.objects} if m is not None else set())
    if val is None:
        ok = False
    elif t == "find":
        assert find_rule in FIND_RULES, find_rule
        o = m.obj(val)
        ok = bool(np.linalg.norm(o.center - np.array(q["answer_center"])) <= FIND_TOL_M) and (find_rule == "dist" or o.label == q["label"])
    else:
        ok = val == q["answer"]
    return {"parsed": val, "correct": ok, "invalid": val is None}


def summarise(row: str, res: list[dict]) -> dict:
    def agg(rs: list[dict]) -> dict:
        return {"n": len(rs), "correct": sum(r["correct"] for r in rs), "accuracy": sum(r["correct"] for r in rs) / len(rs),
                "invalid": sum(r["invalid"] for r in rs), "not_asked": sum(r["not_asked"] for r in rs)}
    return {"row": row, **agg(res), "per_type": {t: agg([r for r in res if r["type"] == t]) for t in TYPES}}


def table(s: dict) -> str:
    rows = [("overall", s)] + list(s["per_type"].items())
    return "\n".join([f"row {s['row']}", f"{'type':10s} {'n':>5s} {'correct':>8s} {'acc %':>7s} {'invalid':>8s} {'not_asked':>10s}"]
                     + [f"{k:10s} {v['n']:5d} {v['correct']:8d} {100 * v['accuracy']:7.1f} {v['invalid']:8d} {v['not_asked']:10d}"
                        for k, v in rows])


def score_row(row: str, qs: list[dict], raw: dict[str, str], prompts: dict[str, dict], find_rule: str = "dist") -> list[dict]:
    """One result per question. A question with no prompt is img7b find (not asked, wrong); any other gap is an error."""
    out = []
    for q in qs:
        base = {"qid": q["qid"], "row": row, "type": q["type"]}
        if q["qid"] not in prompts:
            assert row in IMG_ROWS and q["type"] == "find", f"{q['qid']}: no prompt in row {row}"
            out.append({**base, "raw_output": None, "parsed": None, "correct": False, "invalid": False, "not_asked": True})
            continue
        m, vocab = context(prompts[q["qid"]]["user_text"])
        out.append({**base, "raw_output": raw[q["qid"]], **score_one(q, raw[q["qid"]], m, vocab, find_rule), "not_asked": False})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--find-rule", default="dist", choices=(*FIND_RULES, "both"))
    ap.add_argument("--questions", type=Path, default=QUESTIONS)
    a = ap.parse_args()
    prompts = {d["qid"]: d for d in map(json.loads, a.prompts.read_text().splitlines())}
    raws = [json.loads(ln) for ln in a.raw.read_text().splitlines()]
    assert all(r["row"] == a.row for r in raws), "raw file has rows other than --row"
    raw = {r["qid"]: r["raw_output"] for r in raws}
    assert len(raw) == len(raws), "duplicate qid in raw file"
    assert set(raw) == set(prompts), "raw qids != prompt qids"
    a.out.mkdir(parents=True, exist_ok=True)
    qs = read(a.questions)
    for rule in FIND_RULES if a.find_rule == "both" else (a.find_rule,):
        res = score_row(a.row, qs, raw, prompts, rule)
        s = summarise(a.row, res)
        s["find_rule"] = rule
        tag = "" if rule == "dist" else "_" + rule
        (a.out / f"results_{a.row}{tag}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in res))
        (a.out / f"summary_{a.row}{tag}.json").write_text(json.dumps(s, indent=1))
        print(f"find rule: {rule}\n" + table(s))


if __name__ == "__main__":
    main()
