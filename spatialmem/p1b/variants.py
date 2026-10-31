"""Build the 64 map variants of P1B_SWEEP §1 (gt + 60 degraded + cg / ovo / ours_v2): features (§4), overflow rule (§3),
v1.2 prompts, chunk files, manifest. `uv run python -m spatialmem.p1b.variants`  (data/p1b/sweep/, results/p1b/)."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np

from spatialmem.eval.contract import PredRelation, Prediction
from spatialmem.eval.objects import score_objects
from spatialmem.eval.score_baselines import HSSD_GT, load_pred, pool, score_scene
from spatialmem.hssd.gt import load_hssd_gt
from spatialmem.hssd.vocab import ROOT
from spatialmem.p1b import prompt as PR
from spatialmem.p1b.degrade import REAL, degrade, degraded_names, parse_name
from spatialmem.p1b.jsonio import dump
from spatialmem.p1b.gtmap import Map, eval_scenes, from_prediction, load, vocabulary
from spatialmem.p1b.questions import read
from spatialmem.p1b.repair import repair
from spatialmem.p1b.serialise import serialise
from spatialmem.p1b.tokens import DIR, count

SWEEP = ROOT / "data" / "p1b" / "sweep"
RES = ROOT / "results" / "p1b"
LIMIT = 30_000
FALLBACK_LIMIT = 32_736      # 32,768 native context - max_tokens 32 (P1B_REPAIR amendment)
N_CHUNKS = 4
# P1a HSSD eval 30 pooled (RESULTS.md): F1@.25, label acc, relation F1 -- asserted to 3 decimals
RESULTS_MD = {"cg": (0.406, 0.409, 0.096), "ovo": (0.312, 0.523, 0.089), "ours_v2": (0.371, 0.450, 0.095)}


def map_variants() -> list[str]:
    """The 64 map variants in the fixed order: gt first, then sorted."""
    return ["gt"] + sorted(degraded_names() + list(REAL))


@lru_cache(maxsize=1)
def _ctx():
    from transformers import AutoTokenizer
    scenes = eval_scenes()
    gts = {s: load_hssd_gt(HSSD_GT / s / "gt.json")[0] for s in scenes}
    maps = {s: load(s) for s in scenes}
    qs: dict[str, list[dict]] = defaultdict(list)
    for q in read():
        qs[q["scene"]].append(q)
    return scenes, gts, maps, vocabulary(list(maps.values())), qs, AutoTokenizer.from_pretrained(DIR)


def make_pred(name: str, scene: str) -> Prediction:
    scenes, gts, maps, vocab, qs, tok = _ctx()
    if name == "gt":
        return from_map_free(gts[scene])
    if name in REAL:
        return load_pred(name, "hssd", scene, True)
    if "__" in name:
        m, kind = name.split("__")
        return repair(load_pred(m, "hssd", scene, True), gts[scene], kind)
    axis, lv, seed = parse_name(name)
    return degrade(gts[scene], axis, lv, seed, scene, vocab)


def from_map_free(gt: Prediction) -> Prediction:
    return Prediction(gt.scene, gt.objects, gt.relations)


def _counts(r: dict) -> dict:
    return {"obj_tp": r["obj@0.25"]["tp"], "obj_n_pred": r["obj@0.25"]["n_pred"], "obj_n_gt": r["obj@0.25"]["n_gt"],
            "label_correct": r["obj@0.25"]["label_correct"], "rel_tp": r["rel"]["tp"], "rel_n_pred": r["rel"]["n_pred"],
            "rel_n_gt": r["rel"]["n_gt"]}


def pooled(rows: list[dict]) -> dict:
    p = pool(rows)
    return {"f1": p["obj@0.25"]["f1"], "label_acc": p["obj@0.25"]["label_acc"], "rel_f1": p["rel"]["f1"]}


def _prompt(q: dict, row: str, text: str) -> dict:
    return PR.build(q, row, text, [], 0, "")


def _toks(p: dict, tok) -> int:
    return count(p, tok)


def _relevant(m: Map, q: dict) -> set[tuple[int, str, int]]:
    """GT triples that make a relation / find question's answer true (§3)."""
    p = q["pred"]
    if q["type"] == "relation":
        a, B = m.with_label(q["label"])[0].id, q["answer"]
    else:
        a, B = q["answer_id"], q["label_b"]
    out = set()
    for b in m.related(a, p):
        if m.obj(b).label == B:
            out.add((a, p, b) if p != "next_to" else (min(a, b), p, max(a, b)))
    return out


def trim(m: Map, f_long, f_worst) -> tuple[list, bool]:
    """Overflow rule (P1B_SWEEP §3 + P1B_REPAIR fallback). f_long / f_worst: map text -> tokens of the longest-question prompt /
    of the worst question prompt. Returns (removed lines, fallback flag). Lines go farthest centre-to-centre first, ties by
    (subject id, object id). Rule: next_to until f_long <= LIMIT (bisection); if all next_to still leave it above, remove all
    next_to and keep if f_worst <= FALLBACK_LIMIT, else also `in` then `on` lines until f_worst <= FALLBACK_LIMIT."""
    ctr = {o.id: o.center for o in m.objects}
    key = lambda r: (-float(np.linalg.norm(ctr[r[0]] - ctr[r[2]])), r[0], r[2])
    nt, ins, ons = (sorted((r for r in m.relations if r[1] == p), key=key) for p in ("next_to", "in", "on"))

    def txt(order: list, k: int) -> str:
        gone = set(order[:k])
        return serialise(Map(m.scene, m.objects, [r for r in m.relations if r not in gone]))

    def bisect(order: list, lo: int, hi: int, f, limit: int) -> int:     # lo fails, hi must fit (monotone)
        assert f(txt(order, hi)) <= limit, "does not fit"
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (mid, hi) if f(txt(order, mid)) > limit else (lo, mid)
        return hi

    if f_long(txt(nt, len(nt))) <= LIMIT:
        return nt[:bisect(nt, 0, len(nt), f_long, LIMIT)], False
    order = nt + ins + ons
    k = len(nt) if f_worst(txt(order, len(nt))) <= FALLBACK_LIMIT else bisect(order, len(nt), len(order), f_worst, FALLBACK_LIMIT)
    return order[:k], True


def process(name: str, out_dir: Path = SWEEP) -> dict:
    """One variant: features, overflow, prompt file. Returns plain numbers (+ file sha256)."""
    scenes, gts, maps, vocab, qs, tok = _ctx()
    rows, post_rows, over, text = {}, {}, {}, {}
    for s in scenes:
        pred, gt = make_pred(name, s), gts[s]
        rows[s] = score_scene(pred, gt)
        m = from_prediction(pred)
        t0 = serialise(m)
        longest = max(qs[s], key=lambda q: len(f"Question: {q['question']}\nAnswer format: {PR.FMT[q['type']]}"))
        n0 = _toks(_prompt(longest, name, t0), tok)
        rec = {"tokens": n0, "tokens_after": n0, "removed": 0, "hit_rel": 0, "hit_find": 0}
        text[s] = t0
        if n0 > LIMIT:
            tl = lambda t: _toks(_prompt(longest, name, t), tok)
            tw = lambda t: max(_toks(_prompt(q, name, t), tok) for q in qs[s])
            gone_l, fb = trim(m, tl, tw)
            gone = set(gone_l)
            hi = len(gone_l)
            m2 = Map(m.scene, m.objects, [r for r in m.relations if r not in gone])
            t1 = serialise(m2)
            n1 = tl(t1)
            if name not in REAL and "__" not in name:
                p2g = {o.id: o.id for o in m.objects if o.id in {g.id for g in maps[s].objects}}
            else:
                p2g = {pi: gi for pi, gi, _ in score_objects(pred, gt, 0.25).matches}
            gone_g = set()
            for a, p, b in gone:
                if a in p2g and b in p2g:
                    x, y = p2g[a], p2g[b]
                    gone_g.add((min(x, y), p, max(x, y)) if p == "next_to" else (x, p, y))
            hits = {"relation": 0, "find": 0}
            for q in qs[s]:
                if q["type"] in hits and _relevant(maps[s], q) & gone_g:
                    hits[q["type"]] += 1
            mx = max(_toks(_prompt(q, name, t1), tok) for q in qs[s])
            rec.update(tokens_after=n1, removed=hi, hit_rel=hits["relation"], hit_find=hits["find"], tokens_after_max_all=mx)
            if fb:
                rec.update(fallback=1, removed_next_to=sum(r[1] == "next_to" for r in gone), removed_in=sum(r[1] == "in" for r in gone),
                           removed_on=sum(r[1] == "on" for r in gone))
            text[s] = t1
            post = Prediction(pred.scene, pred.objects, [r for r in pred.relations if (r.subj, r.pred, r.obj) not in gone])
            post_rows[s] = score_scene(post, gt)
        over[s] = rec
    out = out_dir / f"prompts_{name}.jsonl"
    h = hashlib.sha256()
    n = 0
    with out.open("wb") as f:
        for s in scenes:
            for q in qs[s]:
                b = (json.dumps(_prompt(q, name, text[s])) + "\n").encode()
                f.write(b)
                h.update(b)
                n += 1
    pre = pooled(list(rows.values()))
    allrows = [post_rows.get(s, rows[s]) for s in scenes]
    return {"name": name, "n": n, "sha256": h.hexdigest(), "over": over,
            "features": {**pre, "houses": {s: _counts(rows[s]) for s in scenes},
                         "post": pooled(allrows), "post_houses": {s: _counts(r) for s, r in post_rows.items()}}}


def img_prompts() -> tuple[int, str]:
    """008 img7b prompts with row 'img', otherwise byte-identical."""
    h, n = hashlib.sha256(), 0
    with (SWEEP / "prompts_img.jsonl").open("wb") as f:
        for ln in (ROOT / "data" / "p1b" / "prompts_img7b.jsonl").read_text().splitlines():
            d = json.loads(ln)
            d["row"] = "img"
            b = (json.dumps(d) + "\n").encode()
            f.write(b)
            h.update(b)
            n += 1
    return n, h.hexdigest()


def chunks(names: list[str], d: Path = SWEEP, n_chunks: int = N_CHUNKS, first: str | None = "gt") -> dict:
    """Round-robin over the fixed order (gt first, then sorted, img included) into N_CHUNKS files."""
    order = ([first] if first else []) + sorted(set(names) - {first})
    man: dict = {"variants": {}, "chunks": {}}
    hs = [hashlib.sha256() for _ in range(n_chunks)]
    files = [(d / f"chunk_{c}.jsonl").open("wb") for c in range(n_chunks)]
    for i, v in enumerate(order):
        c = i % n_chunks
        b = (d / f"prompts_{v}.jsonl").read_bytes()
        files[c].write(b)
        hs[c].update(b)
        man["variants"][v] = {"chunk": c, "n": b.count(b"\n"), "sha256": hashlib.sha256(b).hexdigest()}
    for c, f in enumerate(files):
        f.close()
        man["chunks"][str(c)] = {"n": sum(v["n"] for v in man["variants"].values() if v["chunk"] == c), "sha256": hs[c].hexdigest()}
    return man


def main() -> None:
    SWEEP.mkdir(parents=True, exist_ok=True)
    names = map_variants()
    with ProcessPoolExecutor(max_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 8) as ex:
        res = {r["name"]: r for r in ex.map(process, names)}
    feats = {v: res[v]["features"] for v in names}
    for v, tgt in RESULTS_MD.items():
        got = tuple(round(feats[v][k], 3) for k in ("f1", "label_acc", "rel_f1"))
        assert got == tgt, f"{v}: {got} != RESULTS.md {tgt}"
    assert all(round(feats["gt"][k], 3) == 1.0 for k in ("f1", "label_acc", "rel_f1")), feats["gt"]
    over = {v: res[v]["over"] for v in names}
    dump(RES / "features.json", feats)
    dump(RES / "overflow.json", over)
    n_img, h_img = img_prompts()
    man = chunks(names + ["img"])
    dump(SWEEP / "MANIFEST.json", man)
    print(f"{'variant':22s} {'F1':>6s} {'lab':>6s} {'rel':>6s}  {'F1post':>6s} {'labP':>6s} {'relP':>6s}  houses_over  max_removed")
    for v in names:
        f, o = feats[v], over[v]
        trig = [s for s, r in o.items() if r["removed"] > 0]
        print(f"{v:22s} {f['f1']:6.3f} {f['label_acc']:6.3f} {f['rel_f1']:6.3f}  {f['post']['f1']:6.3f} {f['post']['label_acc']:6.3f} "
              f"{f['post']['rel_f1']:6.3f}  {len(trig):3d}  {max(r['removed'] for r in o.values())}")
    allr = [r for o in over.values() for r in o.values()]
    print("variants with overflow:", sum(any(r["removed"] > 0 for r in o.values()) for o in over.values()), "of", len(names))
    print("max tokens after rule (longest-question prompt):", max(r["tokens_after"] for r in allr),
          " exact max over all prompts of houses >29k:", max((r.get("tokens_after_max_all", 0) for r in allr), default=0))
    print("max tokens before rule:", max(r["tokens"] for r in allr), " img prompts:", n_img)
    print("chunks:", {c: (v["n"], v["sha256"]) for c, v in man["chunks"].items()})


if __name__ == "__main__":
    main()
