"""Rule solver (gate G2): answers each question by parsing its TEXT and querying the GT map, never by reading the stored
answer; its raw text goes through the same parser + scorer as the planner. `python -m spatialmem.p1b.solver`."""
from __future__ import annotations

import re

from spatialmem.p1b.gtmap import Map, eval_scenes, load, vocabulary
from spatialmem.p1b.questions import plural, read, spoken
from spatialmem.p1b.serialise import parse, serialise
from spatialmem.p1b.score import FIND_RULES, score_one, summarise, table

PRED = {"on": "on", "in": "in", "next to": "next_to"}
RE_EXIST = re.compile(r"Is there an? (.+) in the house\?")
RE_REL = re.compile(r"What is the (.+) (on|in|next to)\?")
RE_COUNT = re.compile(r"How many (.+) are there in the house\?")
RE_FIND = re.compile(r"Which object is the (.+) that is (on|in|next to) the (.+)\?")


def _count(m: Map, label: str) -> int:
    return len(m.with_label(label))


def _related(m: Map, a: int, pred: str) -> list[int]:
    return m.related(a, pred)


def label_of(s: str) -> str:
    return s.replace(" ", "_")


def plural_index(vocab: list[str]) -> dict[str, str]:
    idx: dict[str, str] = {}
    for L in vocab:
        p = plural(spoken(L))
        assert p not in idx, f"plural collision {p!r}: {idx[p]} / {L}"
        idx[p] = L
    return idx


def solve(question: str, m: Map, plurals: dict[str, str]) -> str:
    if g := RE_EXIST.fullmatch(question):
        return "yes" if _count(m, label_of(g[1])) >= 1 else "no"
    if g := RE_COUNT.fullmatch(question):
        return str(_count(m, plurals[g[1]]))
    if g := RE_REL.fullmatch(question):
        a = m.with_label(label_of(g[1]))[0]
        return min({m.obj(b).label for b in _related(m, a.id, PRED[g[2]])}, default="none")
    if g := RE_FIND.fullmatch(question):
        B = label_of(g[3])
        hits = [a.id for a in m.with_label(label_of(g[1])) if any(m.obj(b).label == B for b in _related(m, a.id, PRED[g[2]]))]
        return str(hits[0]) if hits else "none"
    raise ValueError(question)


def run(qs: list[dict], maps: dict[str, Map], vocab: list[str], find_rule: str = "dist") -> list[dict]:
    plurals = plural_index(vocab)
    seen = {s: parse(serialise(m)) for s, m in maps.items()}   # the 1-decimal map the planner sees (find distance uses it)
    out = []
    for q in qs:
        m = maps[q["scene"]]
        raw = solve(q["question"], m, plurals)
        out.append({"type": q["type"], **score_one(q, raw, seen[q["scene"]], None, find_rule)})
    return out


if __name__ == "__main__":
    maps = {s: load(s) for s in eval_scenes()}
    qs = read()
    vocab = vocabulary(list(maps.values()))
    for rule in FIND_RULES:
        res = run(qs, maps, vocab, rule)
        for r in res:
            r["not_asked"] = False
        s = summarise("solver", res)
        print(f"find rule: {rule}\n" + table(s))
        assert all(v["accuracy"] == 1.0 for v in [s, *s["per_type"].values()]), f"G2 FAIL ({rule})"
    print("G2 PASS: 100.0 % overall and per type, both find rules")
