"""P1b sweep (task 009): degradation axes."""
import random

import numpy as np
import pytest

from spatialmem.eval.contract import STRUCTURAL, Prediction
from spatialmem.eval.objects import without_structural
from spatialmem.hssd.gt import load_hssd_gt
from spatialmem.p1b.degrade import AXES, LEVELS, degrade
from spatialmem.p1b.gtmap import RUNS, eval_scenes, load, vocabulary

from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
needs_data = pytest.mark.skipif(not ((DATA / "hssd_runs").is_dir() and (DATA / "p1b").is_dir()),
                                reason="needs the git-ignored data/hssd_runs and data/p1b (README, Reproduce)")

SCENE = "102344457"


@pytest.fixture(scope="module")
def gts() -> dict[str, Prediction]:
    return {s: without_structural(load_hssd_gt(RUNS / s / "gt.json")[0]) for s in eval_scenes()}


@pytest.fixture(scope="module")
def vocab() -> list[str]:
    return vocabulary([load(s) for s in eval_scenes()])


def sig(p: Prediction):
    return ([(o.id, o.label, o.aabb_min.tolist(), o.aabb_max.tolist()) for o in p.objects],
            sorted((r.subj, r.pred, r.obj) for r in p.relations))


def keys(p: Prediction) -> set:
    return {r.key() for r in p.relations}


@needs_data
def test_exact_counts(gts, vocab):
    g = gts[SCENE]
    N, M = len(g.objects), len(g.relations)
    assert (N, M) == (len(g.objects), len(g.relations)) and not any(o.label in STRUCTURAL for o in g.objects)
    for lv in LEVELS["drop"]:
        d = degrade(g, "drop", lv, 0, SCENE, vocab)
        assert len(d.objects) == N - round(float(lv) / 100 * N)
        ids = {o.id for o in d.objects}
        assert all(r.subj in ids and r.obj in ids for r in d.relations)
        assert len(d.relations) == sum(r.subj in ids and r.obj in ids for r in g.relations)
    for lv in LEVELS["label"]:
        d = degrade(g, "label", lv, 0, SCENE, vocab)
        k = round(float(lv) / 100 * N)
        ch = [(a, b) for a, b in zip(g.objects, d.objects) if a.label != b.label]
        assert len(d.objects) == N and len(ch) == k                       # never keeps the original label
        assert all(b.label in vocab for _, b in ch)
        assert all(np.array_equal(a.aabb_min, b.aabb_min) for a, b in zip(g.objects, d.objects)) and sig(d)[1] == sig(g)[1]
    for lv in LEVELS["extra"]:
        d = degrade(g, "extra", lv, 0, SCENE, vocab)
        k = round(float(lv) / 100 * N)
        top = max(o.id for o in g.objects)
        assert len(d.objects) == N + k and [o.id for o in d.objects[N:]] == list(range(top + 1, top + 1 + k))
        assert {o.label for o in d.objects[N:]} <= {o.label for o in g.objects}
    for lv in LEVELS["relations"]:
        d = degrade(g, "relations", lv, 0, SCENE, vocab)
        assert len(d.relations) == M - round(float(lv) / 100 * M) and keys(d) <= keys(g)
        assert sig(d)[0] == sig(g)[0]
    for lv in LEVELS["position"]:
        d = degrade(g, "position", lv, 0, SCENE, vocab)
        assert len(d.objects) == N
        assert all(np.allclose(a.aabb_max - a.aabb_min, b.aabb_max - b.aabb_min) for a, b in zip(g.objects, d.objects))


@needs_data
def test_extra_boxes_inside_house(gts, vocab):
    g = gts[SCENE]
    lo, hi = np.min([o.aabb_min for o in g.objects], 0), np.max([o.aabb_max for o in g.objects], 0)
    d = degrade(g, "extra", "100", 1, SCENE, vocab)
    for o in d.objects[len(g.objects):]:
        c = (o.aabb_min + o.aabb_max) / 2
        assert lo[0] <= c[0] <= hi[0] and lo[2] <= c[2] <= hi[2]
        assert any(np.allclose(o.aabb_max - o.aabb_min, p.aabb_max - p.aabb_min) and p.label == o.label for p in g.objects)
        assert any(np.isclose(c[1], (p.aabb_min[1] + p.aabb_max[1]) / 2) for p in g.objects)


@pytest.mark.parametrize("axis", AXES)
@needs_data
def test_deterministic_and_seeds_differ(gts, vocab, axis):
    g, lv = gts[SCENE], LEVELS[axis][1]
    a, b = degrade(g, axis, lv, 0, SCENE, vocab), degrade(g, axis, lv, 0, SCENE, vocab)
    assert sig(a) == sig(b)
    assert sig(degrade(g, axis, lv, 1, SCENE, vocab)) != sig(a)
    assert sig(degrade(g, axis, lv, 0, "102815859", vocab)) != sig(a)    # scene is part of the stream


@pytest.mark.parametrize("axis", AXES)
@needs_data
def test_level_zero_is_gt(gts, vocab, axis):
    lv = "0.00" if axis == "position" else "0"
    for s, g in gts.items():
        d = degrade(g, axis, lv, 0, s, vocab)
        assert sig(d)[0] == sig(g)[0], (axis, s)
        assert keys(d) == keys(g), (axis, s)


@needs_data
def test_position_sigma0_recompute_equals_gt_relations(gts, vocab):
    for s, g in gts.items():
        d = degrade(g, "position", "0.00", 0, s, vocab)
        assert sorted((r.subj, r.pred, r.obj) for r in d.relations) == sorted((r.subj, r.pred, r.obj) for r in g.relations), s


@needs_data
def test_pinned_draw_order(gts, vocab):
    """Independent re-derivation of drop / label with the spec's draw order, and a wrong order that must differ."""
    g, s = gts[SCENE], SCENE
    objs = sorted(g.objects, key=lambda o: o.id)
    n = len(objs)
    k = round(0.25 * n)
    gone = {o.id for o in random.Random(f"009:drop:25:0:{s}").sample(objs, k)}
    assert [o.id for o in degrade(g, "drop", "25", 0, s, vocab).objects] == [o.id for o in objs if o.id not in gone]
    rng = random.Random(f"009:label:25:0:{s}")
    pick = rng.sample(objs, k)
    want = {o.id: rng.choice(sorted(set(vocab) - {o.label})) for o in pick}
    got = {a.id: b.label for a, b in zip(objs, degrade(g, "label", "25", 0, s, vocab).objects) if a.label != b.label}
    assert got == want
    rng = random.Random(f"009:label:25:0:{s}")                 # wrong order: label draws interleaved before the sample
    wrong_labels = [rng.choice(sorted(set(vocab) - {o.label})) for o in objs[:k]]
    wrong_pick = rng.sample(objs, k)
    assert {o.id: l for o, l in zip(wrong_pick, wrong_labels)} != want
    wrong_gone = {o.id for o in random.Random(f"009:drop:25:0:{s}").sample(objs[::-1], k)}   # unsorted (reversed) list
    assert wrong_gone != gone


# ---- collect / sweep ---------------------------------------------------------------------------------------------------
import hashlib
import json

from spatialmem.eval.score_baselines import pool
from spatialmem.p1b import collect, prompt as PR, sweep
from spatialmem.p1b.degrade import REAL, degraded_names
from spatialmem.p1b.questions import TYPES, read
from spatialmem.p1b.serialise import serialise
from spatialmem.hssd.vocab import ROOT
from spatialmem.p1b.variants import map_variants


def test_variant_names():
    names = map_variants()
    assert len(names) == 64 and names[0] == "gt" and len(set(names)) == 64 and set(REAL) <= set(names)
    assert len(degraded_names()) == 60 and "position_0.10_s0" in names and "extra_100_s2" in names


@needs_data
def test_collect_fake_raw(tmp_path, gts, vocab):
    qs = read()
    maps = {s: load(s) for s in eval_scenes()}
    text = {s: serialise(m) for s, m in maps.items()}
    pdir = tmp_path / "p"
    pdir.mkdir()
    for row, kind in (("gt", "map"), ("img", "img")):
        ps = [PR.build(q, "img7b" if kind == "img" else row, text[q["scene"]], vocab, 100, "/r") for q in qs if not (kind == "img" and q["type"] == "find")]
        for p in ps:
            p["row"] = row
        (pdir / f"prompts_{row}.jsonl").write_text("".join(json.dumps(p) + "\n" for p in ps))
        key = {q["qid"]: q for q in qs}
        raw = [{"qid": p["qid"], "row": row, "model": "fake", "raw_output": str(key[p["qid"]]["answer_id"]) if key[p["qid"]]["type"] == "find"
                else str(key[p["qid"]]["answer"])} for p in ps]
        (tmp_path / f"raw_{row}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in raw))
    out = tmp_path / "out"
    runs = collect.collect([tmp_path / "raw_gt.jsonl", tmp_path / "raw_img.jsonl"], pdir, out)
    assert runs["gt"]["accuracy"] == 1.0 and all(v["accuracy"] == 1.0 for v in runs["gt"]["per_type"].values())   # label+dist rule too
    assert runs["img"]["per_type"]["find"] == {"n": 120, "correct": 0, "accuracy": 0.0, "invalid": 0, "not_asked": 120}
    assert runs["img"]["correct"] == 359 and json.loads((out / "runs.json").read_text())["gt"]["n"] == 479
    pq = json.loads((out / "per_question" / "gt.json").read_text())
    assert pq == sorted(pq) and len(pq) == 479 and all(r[2] == 1 and r[3] == 0 for r in pq)
    assert set(map(len, pq)) == {4}


@needs_data
def test_sweep_gt_prompts_equal_008_gt72b_v12():
    f, g = ROOT / "data/p1b/sweep/prompts_gt.jsonl", ROOT / "data/p1b/prompts_gt72b_v12.jsonl"
    a = [json.loads(x) for x in f.read_text().splitlines()]
    b = [json.loads(x) for x in g.read_text().splitlines()]
    assert len(a) == 479 and all(x["row"] == "gt" for x in a)
    assert [{**x, "row": "gt72b"} for x in a] == b


def test_pooled_matches_score_baselines_pool():
    rng = np.random.default_rng(0)
    houses, rows = {}, []
    for i in range(7):
        tp, npr, ngt = int(rng.integers(1, 20)), int(rng.integers(20, 40)), int(rng.integers(20, 40))
        lc, rtp, rnp, rng_ = int(rng.integers(0, tp + 1)), int(rng.integers(0, 9)), int(rng.integers(10, 30)), int(rng.integers(10, 30))
        houses[f"h{i}"] = dict(obj_tp=tp, obj_n_pred=npr, obj_n_gt=ngt, label_correct=lc, rel_tp=rtp, rel_n_pred=rnp, rel_n_gt=rng_)
        o = {"tp": tp, "n_pred": npr, "n_gt": ngt, "label_correct": lc, "f1": 0.0}
        rows.append({"obj@0.25": o, "obj@0.5": o, "rel": {"tp": rtp, "n_pred": rnp, "n_gt": rng_}})
    p, q = pool(rows), sweep.pooled(houses)
    assert q == pytest.approx({"f1": p["obj@0.25"]["f1"], "label_acc": p["obj@0.25"]["label_acc"], "rel_f1": p["rel"]["f1"]})


def synth(res):
    """Synthetic results/p1b for 6 houses and the 64 + img variants."""
    from spatialmem.p1b.jsonio import dump
    rng = np.random.default_rng(114)
    scenes = [f"h{i}" for i in range(6)]
    feats, over, runs = {}, {}, {}
    for v in map_variants():
        q = 1.0 if v == "gt" else float(rng.uniform(0.1, 0.9))
        hs = {}
        for s in scenes:
            n = int(rng.integers(40, 60))
            tp = round(q * n)
            hs[s] = dict(obj_tp=tp, obj_n_pred=n + int(rng.integers(0, 5)), obj_n_gt=n, label_correct=round(tp * rng.uniform(0.3, 1)),
                         rel_tp=round(q * 20), rel_n_pred=22, rel_n_gt=20)
        feats[v] = {**sweep.pooled(hs), "houses": hs, "post": sweep.pooled(hs), "post_houses": {}}
        over[v] = {s: {"removed": int(v == "cg" and s == "h1"), "tokens": 1, "tokens_after": 1, "hit_rel": 0, "hit_find": 0} for s in scenes}
        pq = []
        for s in scenes:
            for t in TYPES:
                for k in range(4):
                    pq.append([f"{s}:{t}:{k}", t, int(rng.random() < 0.2 + 0.7 * q), 0])
        runs[v] = {"n": len(pq), "correct": sum(r[2] for r in pq), "accuracy": sum(r[2] for r in pq) / len(pq), "invalid": 0, "not_asked": 0,
                   "per_type": {t: {"n": 24, "correct": sum(r[2] for r in pq if r[1] == t), "accuracy": sum(r[2] for r in pq if r[1] == t) / 24,
                                    "invalid": 0, "not_asked": 0} for t in TYPES}}
        dump(res / "per_question" / f"{v}.json", sorted(pq))
    runs["img"] = runs["gt"]
    dump(res / "features.json", feats)
    dump(res / "overflow.json", over)
    dump(res / "runs.json", runs)


def test_sweep_byte_identical_and_reading(tmp_path):
    outs = []
    for i in range(2):
        res = tmp_path / f"r{i}"
        synth(res)
        s = sweep.analyse(res)
        from spatialmem.p1b.jsonio import dump
        dump(res / "summary.json", s)
        sweep.figures(s, tmp_path / f"f{i}")
        outs.append((res / "summary.json", tmp_path / f"f{i}" / "009_curves.png", tmp_path / f"f{i}" / "009_predictor.png"))
    for a, b in zip(*outs):
        assert a.read_bytes() == b.read_bytes() and len(a.read_bytes()) > 1000
    s = json.loads(outs[0][0].read_bytes())
    assert s["headline"]["n_fit"] == 61 and s["headline"]["reading"] in ("map scores predict planner success", "fails", "partial")
    assert s["robustness"]["houses"] == ["h0", "h2", "h3", "h4", "h5"]            # h1 overflowed in `cg`
    assert set(s["curves"]) == set(sweep.AXES) and len(s["curves"]["drop"]["levels"]) == 5
    assert set(s["headline"]["real"]) == set(REAL)


def test_reading_thresholds():
    assert sweep.reading({"a": 5.0, "b": 0.0}) == "map scores predict planner success"
    assert sweep.reading({"a": 5.01, "b": 0.0}) == "partial" and sweep.reading({"a": 10.0, "b": 1.0}) == "partial"
    assert sweep.reading({"a": 10.01, "b": 1.0}) == "fails"


def test_ols_recovers_known_coefficients():
    rng = np.random.default_rng(1)
    X = np.c_[np.ones(50), rng.random((50, 3))]
    beta, r2 = sweep.ols(X, X @ np.array([0.1, 0.2, 0.3, 0.4]))
    assert np.allclose(beta, [0.1, 0.2, 0.3, 0.4]) and r2 == pytest.approx(1.0)
