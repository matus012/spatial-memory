"""P1b repair attribution (task 010): repairs, relation-recompute premise, analysis."""
import numpy as np
import pytest

from spatialmem.eval.contract import Prediction
from spatialmem.eval.objects import score_objects, without_structural
from spatialmem.eval.score_baselines import load_pred, pool, score_scene
from spatialmem.hssd.gt import load_hssd_gt
from spatialmem.gt.relations import compute_relations
from spatialmem.p1b.degrade import REAL
from spatialmem.p1b.gtmap import RUNS, eval_scenes
from spatialmem.p1b.repair import KINDS, repair

from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
needs_data = pytest.mark.skipif(not ((DATA / "hssd_runs").is_dir() and (DATA / "p1b").is_dir()),
                                reason="needs the git-ignored data/hssd_runs and data/p1b (README, Reproduce)")

SCENES = eval_scenes()


@pytest.fixture(scope="module")
def gts() -> dict[str, Prediction]:
    return {s: load_hssd_gt(RUNS / s / "gt.json")[0] for s in SCENES}


def rel(p: Prediction) -> list:
    return sorted(r.key() for r in p.relations)


def objsig(p: Prediction):
    return [(o.id, o.label, o.aabb_min.tolist(), o.aabb_max.tolist()) for o in sorted(p.objects, key=lambda o: o.id)]


@pytest.mark.parametrize("kind", KINDS)
@needs_data
def test_gt_as_prediction_is_identity(gts, kind):
    for s in SCENES[:10]:
        g = without_structural(gts[s])
        r = repair(gts[s], gts[s], kind)
        assert objsig(r) == objsig(g) and rel(r) == rel(g), (kind, s)


@needs_data
def test_exact_counts_and_box_scope(gts):
    for m in REAL:
        for s in SCENES[:6]:
            pred, g = without_structural(load_pred(m, "hssd", s, True)), without_structural(gts[s])
            mt = score_objects(pred, g, 0.25).matches          # (pred id, gt id, iou)
            gl, pl = {o.id: o.label for o in g.objects}, {o.id: o.label for o in pred.objects}
            n_ch = sum(pl[a] != gl[b] for a, b, _ in mt)
            r = repair(pred, g, "label")
            assert len(r.objects) == len(pred.objects) and sum(a.label != b.label for a, b in zip(pred.objects, r.objects)) == n_ch
            assert all(a.label == gl[b] for a, b, _ in mt for a in [next(o for o in r.objects if o.id == a)])
            assert len(repair(pred, g, "extra").objects) == len(mt)
            r = repair(pred, g, "miss")
            top = max(o.id for o in pred.objects)
            assert len(r.objects) == len(pred.objects) + len(g.objects) - len(mt)
            added = r.objects[len(pred.objects):]
            assert [o.id for o in added] == list(range(top + 1, top + 1 + len(added)))
            assert sorted(o.label for o in added) == sorted(o.label for o in g.objects if o.id not in {b for _, b, _ in mt})
            r = repair(pred, g, "box")
            matched = {a for a, _, _ in mt}
            gb = {o.id: o for o in g.objects}
            for a, b in zip(pred.objects, r.objects):
                if a.id in matched:
                    j = next(y for x, y, _ in mt if x == a.id)
                    assert np.array_equal(b.aabb_min, gb[j].aabb_min) and np.array_equal(b.aabb_max, gb[j].aabb_max)
                else:
                    assert np.array_equal(b.aabb_min, a.aabb_min) and np.array_equal(b.aabb_max, a.aabb_max)
            ids = [o.id for o in repair(pred, g, "all").objects]
            assert len(ids) == len(g.objects) and len(set(ids)) == len(ids)


@needs_data
def test_combo_matches_stepwise_subset(gts):
    pred, g = load_pred("cg", "hssd", SCENES[0], True), gts[SCENES[0]]
    ok = repair(pred, g, "label+extra")
    assert {o.id for o in ok.objects} <= {o.id for o in pred.objects}
    assert objsig(repair(pred, g, "label+extra+miss"))[: len(ok.objects)] == objsig(ok)   # miss only appends higher ids


@needs_data
def test_premise_recompute_equals_stored_relations():
    """acc(M) = 009 row needs compute_relations(up=1) to reproduce each mapper's stored relations exactly."""
    for m in REAL:
        for s in SCENES:
            p = without_structural(load_pred(m, "hssd", s, True))
            assert sorted(r.key() for r in compute_relations(p.objects, up=1)) == rel(p), (m, s)


@pytest.mark.parametrize("m", REAL)
@needs_data
def test_all_repair_features_are_one(gts, m):
    rows = [score_scene(repair(load_pred(m, "hssd", s, True), gts[s], "all"), gts[s]) for s in SCENES]
    p = pool(rows)
    assert (p["obj@0.25"]["f1"], p["obj@0.25"]["label_acc"], p["rel"]["f1"]) == (1.0, 1.0, 1.0)


# ---- analysis / collect --------------------------------------------------------------------------------------------------
import json

from spatialmem.p1b import repair_analysis as RA
from spatialmem.p1b.jsonio import dump
from spatialmem.p1b.questions import TYPES


def synth(res):
    rng = np.random.default_rng(10)
    types = [t for t in TYPES for _ in range(30)]
    def pq(p):
        return sorted([f"s{i}:{t}:{i}", t, int(rng.random() < p), 0] for i, t in enumerate(types))
    gain = {"label": 0.05, "extra": 0.1, "miss": 0.3, "box": 0.02, "label+extra": 0.15, "label+extra+miss": 0.5, "all": 0.6}
    for m in REAL:
        dump(res / "per_question" / f"{m}.json", pq(0.3))
        for k in KINDS:
            dump(res / "repair" / "per_question" / f"{m}__{k}.json", pq(0.3 + gain[k]))
    v = [f"{m}__{k}" for m in REAL for k in KINDS]
    dump(res / "repair" / "features.json", {x: {"f1": 0.5} for x in v})
    dump(res / "repair" / "overflow.json", {x: {"h": {"removed": 0}} for x in v})


def test_dominant_rule():
    g = {"label": 1.0, "extra": 7.0, "miss": 12.0, "box": 0.0}
    assert RA.dominant(g)["reading"] == "miss" and RA.dominant(g)["margin_pp"] == 5.0           # margin exactly 5.0 -> dominant
    g["miss"] = 11.99
    assert RA.dominant(g)["reading"] == "no single dominant error"


def test_analysis_byte_identical_and_gains(tmp_path):
    outs = []
    for i in range(2):
        res = tmp_path / f"r{i}"
        synth(res)
        s = RA.analyse(res)
        dump(res / "repair.json", s)
        RA.figure(s, tmp_path / f"f{i}")
        outs.append((res / "repair.json", tmp_path / f"f{i}" / "010_repair.png"))
    for a, b in zip(*outs):
        assert a.read_bytes() == b.read_bytes() and len(a.read_bytes()) > 1000
    s = json.loads(outs[0][0].read_bytes())
    for m in REAL:
        o = s["maps"][m]["overall"]
        assert o["dominant"]["top"] == "miss" and o["dominant"]["reading"] == "miss"
        assert o["gains_pp"]["miss"] == pytest.approx(100 * (s["accuracy"][f"{m}__miss"]["overall"] - s["accuracy"][m]["overall"]), abs=1e-4)
        assert s["maps"][m]["g2"]["delta_pp"] == pytest.approx(s["maps"][m]["g2"]["all_pct"] - 85.4, abs=1e-4)
    assert len(s["per_question"]) == 21


@needs_data
def test_collect_repair_named_rows(tmp_path):
    from spatialmem.p1b import collect, prompt as PR
    from spatialmem.p1b.gtmap import load, vocabulary
    from spatialmem.p1b.questions import read
    from spatialmem.p1b.serialise import serialise
    qs = read()
    maps = {s: load(s) for s in SCENES}
    text = {s: serialise(m) for s, m in maps.items()}
    ps = [{**PR.build(q, "gt", text[q["scene"]], [], 0, ""), "row": "cg__label+extra"} for q in qs]
    (tmp_path / "prompts_cg__label+extra.jsonl").write_text("".join(json.dumps(p) + "\n" for p in ps))
    key = {q["qid"]: q for q in qs}
    raw = [{"qid": q["qid"], "row": "cg__label+extra", "model": "f", "raw_output": str(q["answer_id"] if q["type"] == "find" else q["answer"])} for q in qs]
    (tmp_path / "raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in raw))
    runs = collect.collect([tmp_path / "raw.jsonl"], tmp_path, tmp_path / "res" / "repair")
    assert runs["cg__label+extra"]["accuracy"] == 1.0
    assert (tmp_path / "res" / "repair" / "per_question" / "cg__label+extra.json").exists() and (tmp_path / "res" / "repair" / "runs.json").exists()


def test_trim_rule_and_fallback(monkeypatch):
    from spatialmem.p1b import variants as V
    from spatialmem.p1b.gtmap import Map, MapObj
    from spatialmem.p1b.serialise import serialise
    n = 30
    objs = [MapObj(i, "a", np.array([float(i), 0, 0]), np.ones(3)) for i in range(n)]
    nt = [(i, "next_to", j) for i in range(n) for j in range(i + 1, n) if j - i <= 3]
    ins = [(i, "in", i + 1) for i in range(n - 1)]
    ons = [(i, "on", i + 2) for i in range(n - 2)]
    m = Map("s", objs, nt + ins + ons)
    size = lambda t: len(t)
    full_nt = size(serialise(Map("s", objs, ins + ons)))
    # normal case: limit between "no next_to" and "all"; farthest first; minimal removals
    monkeypatch.setattr(V, "LIMIT", full_nt + 300)
    gone, fb = V.trim(m, size, size)
    assert not fb and all(r[1] == "next_to" for r in gone) and 0 < len(gone) < len(nt)
    d = [abs(r[2] - r[0]) for r in gone]
    assert d == sorted(d, reverse=True) and size(serialise(Map("s", objs, [r for r in m.relations if r not in set(gone)]))) <= V.LIMIT
    assert size(serialise(Map("s", objs, [r for r in m.relations if r not in set(gone[:-1])]))) > V.LIMIT       # minimal
    # fallback, all next_to suffice under the larger limit
    monkeypatch.setattr(V, "LIMIT", full_nt - 100)
    monkeypatch.setattr(V, "FALLBACK_LIMIT", full_nt + 50)
    gone, fb = V.trim(m, size, size)
    assert fb and sorted(gone) == sorted(nt)
    # fallback, also `in` then `on` lines
    monkeypatch.setattr(V, "FALLBACK_LIMIT", full_nt - 200)
    gone, fb = V.trim(m, size, size)
    kinds = [r[1] for r in gone]
    assert fb and kinds[: len(nt)] == ["next_to"] * len(nt) and set(kinds[len(nt):]) <= {"in", "on"} and "in" in kinds
    assert kinds[len(nt):] == sorted(kinds[len(nt):], key=lambda k: k != "in")             # `in` before `on`
    assert size(serialise(Map("s", objs, [r for r in m.relations if r not in set(gone)]))) <= V.FALLBACK_LIMIT
    assert size(serialise(Map("s", objs, [r for r in m.relations if r not in set(gone[:-1])]))) > V.FALLBACK_LIMIT
