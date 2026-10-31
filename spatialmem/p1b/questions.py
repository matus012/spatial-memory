"""Question generator (P1B_PROTOCOL §2). `python -m spatialmem.p1b.questions` writes data/p1b/questions_v1.jsonl."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from spatialmem.eval.contract import PREDICATES
from spatialmem.hssd.vocab import ROOT
from spatialmem.p1b.gtmap import Map, eval_scenes, load, vocabulary

OUT = ROOT / "data" / "p1b" / "questions_v1.jsonl"
TYPES = ("existence", "relation", "counting", "find")
PRED_TEXT = {"on": "on", "in": "in", "next_to": "next to"}
QUOTA = {"existence": 4, "relation": 4, "counting": 4, "find": 4}   # existence = 2 pos + 2 neg


def spoken(label: str) -> str:
    return label.replace("_", " ")


def article(s: str) -> str:
    return "an" if s[0] in "aeiou" else "a"


def plural(s: str) -> str:
    if s.endswith(("ss", "x", "z", "ch", "sh")):
        return s + "es"
    if s.endswith("s"):
        return s
    if s.endswith("y") and len(s) > 1 and s[-2].isalpha() and s[-2] not in "aeiou":
        return s[:-1] + "ies"
    return s + "s"


def _round_robin(rng: random.Random, pools: dict[str, list], k: int) -> list[tuple[str, tuple]]:
    """k draws without replacement, cycling over predicates in PREDICATES order, skipping empty pools."""
    out: list[tuple[str, tuple]] = []
    while len(out) < k and any(pools.values()):
        for p in PREDICATES:
            if pools[p] and len(out) < k:
                out.append((p, pools[p].pop(rng.randrange(len(pools[p])))))
    return out


def generate(m: Map, vocab: list[str]) -> list[dict]:
    n: dict[str, int] = {}
    for o in m.objects:
        n[o.label] = n.get(o.label, 0) + 1
    qs: list[dict] = []

    def add(t: str, question: str, answer, **kw) -> None:
        qs.append({"qid": f"{m.scene}:{t}:{sum(q['type'] == t for q in qs)}", "scene": m.scene, "type": t,
                   "question": question, "answer": answer, **kw})

    # existence. Sort key: label (alphabetical).
    rng = random.Random(f"114:{m.scene}:existence")
    pos, neg = sorted(n), sorted(set(vocab) - set(n))
    for pol, pool, ans in (("pos", pos, "yes"), ("neg", neg, "no")):
        for L in rng.sample(pool, min(2, len(pool))):
            s = spoken(L)
            add("existence", f"Is there {article(s)} {s} in the house?", ans, label=L, polarity=pol)

    # relation. Candidate (A, p, B): n(A) = 1, labels of everything a is p-related to == {B}, B != A.
    # Sort key per predicate: (A,) -- one candidate per (A, p) pair.
    rng = random.Random(f"114:{m.scene}:relation")
    pools: dict[str, list] = {p: [] for p in PREDICATES}
    for A in sorted(n):
        if n[A] != 1:
            continue
        a = m.with_label(A)[0]
        for p in PREDICATES:
            Bs = {m.obj(b).label for b in m.related(a.id, p)}
            if len(Bs) == 1 and (B := next(iter(Bs))) != A:
                pools[p].append((A, B))
    for p, (A, B) in _round_robin(rng, pools, QUOTA["relation"]):
        add("relation", f"What is the {spoken(A)} {PRED_TEXT[p]}?", B, label=A, pred=p)

    # counting. Sort key: label.
    rng = random.Random(f"114:{m.scene}:counting")
    for L in rng.sample([L for L in sorted(n) if n[L] <= 6], min(QUOTA["counting"], sum(v <= 6 for v in n.values()))):
        add("counting", f"How many {plural(spoken(L))} are there in the house?", n[L], label=L)

    # find. Candidate (A, p, B): n(A) >= 2, A != B, exactly one object of label A p-related to some object of label B.
    # Sort key per predicate: (A, B) -- one candidate per (A, p, B) triple.
    rng = random.Random(f"114:{m.scene}:find")
    pools = {p: [] for p in PREDICATES}
    for A in sorted(n):
        if n[A] < 2:
            continue
        for p in PREDICATES:
            hit: dict[str, list[int]] = {}
            for a in m.with_label(A):
                for B in {m.obj(b).label for b in m.related(a.id, p)} - {A}:
                    hit.setdefault(B, []).append(a.id)
            pools[p] += [(A, B, ids[0]) for B, ids in sorted(hit.items()) if len(ids) == 1]
    for p, (A, B, aid) in _round_robin(rng, pools, QUOTA["find"]):
        add("find", f"Which object is the {spoken(A)} that is {PRED_TEXT[p]} the {spoken(B)}?", f"obj_{aid}",
            label=A, pred=p, label_b=B, answer_id=aid, answer_center=m.obj(aid).center.tolist())
    return qs


def write(path: Path = OUT) -> list[dict]:
    maps = [load(s) for s in eval_scenes()]
    vocab = vocabulary(maps)
    qs = [q for m in maps for q in generate(m, vocab)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("".join(json.dumps(q) + "\n" for q in qs).encode())   # bytes: no CRLF on Windows
    return qs


def read(path: Path = OUT) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines()]


if __name__ == "__main__":
    qs = write()
    print(f"{OUT}")
    for t in TYPES:
        print(f"{t:10s} {sum(q['type'] == t for q in qs)}")
    per = [sum(q["scene"] == s for q in qs) for s in eval_scenes()]
    print(f"per house min {min(per)} max {max(per)}  total {len(qs)}")
    print(f"sha256 {hashlib.sha256(OUT.read_bytes()).hexdigest()}")
