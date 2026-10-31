"""P1b protocol v1 (task 008 step A): map text, questions, parser, scorer, solver, prompts."""
import hashlib
import random

import numpy as np
import pytest

from spatialmem.eval.contract import PredObject, PredRelation, Prediction
from spatialmem.p1b import parse as P
from spatialmem.p1b import prompt, solver
from spatialmem.p1b.gtmap import Map, MapObj, eval_scenes, from_prediction, load, vocabulary
from spatialmem.p1b.questions import TYPES, article, generate, plural, read, spoken, write
from spatialmem.p1b.score import context, score_one, score_row, summarise
from spatialmem.p1b.serialise import parse, parse_any, parse_v11, serialise, serialise_v11

from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
needs_data = pytest.mark.skipif(not ((DATA / "hssd_runs").is_dir() and (DATA / "p1b").is_dir()),
                                reason="needs the git-ignored data/hssd_runs and data/p1b (README, Reproduce)")


@pytest.fixture(scope="module")
def maps():
    return {s: load(s) for s in eval_scenes()}


@pytest.fixture(scope="module")
def vocab(maps):
    return vocabulary(list(maps.values()))


@pytest.fixture(scope="module")
def qs(maps, vocab):
    return [q for m in maps.values() for q in generate(m, vocab)]


# ---- gtmap / serialise (G3) ------------------------------------------------------------------------------------------
def test_from_prediction_drops_structural_and_their_relations():
    z = np.zeros(3)
    p = Prediction("s", [PredObject(1, "wall", z, z + 1), PredObject(2, "chair", z, z + 2), PredObject(3, "floor", z, z + 1),
                         PredObject(4, "table", z, z + 1)],
                   [PredRelation(2, "on", 3), PredRelation(2, "next_to", 4), PredRelation(1, "next_to", 2), PredRelation(2, "on", 99)])
    m = from_prediction(p)
    assert [o.id for o in m.objects] == [2, 4] and m.relations == [(2, "next_to", 4)]
    assert np.allclose(m.objects[0].center, [1, 1, 1]) and np.allclose(m.objects[0].size, [2, 2, 2])


@needs_data
def test_gt_maps_have_no_structural(maps):
    assert len(maps) == 30
    assert all(o.label not in ("wall", "floor", "ceiling", "undefined") for m in maps.values() for o in m.objects)


def test_serialise_v11_format():
    m = Map("h", [MapObj(10, "chest_of_drawers", np.array([-0.001, 1.005, 2.0]), np.array([1, 2, 3.456])), MapObj(9, "a", np.zeros(3), np.ones(3))],
            [(10, "on", 9), (9, "next_to", 10), (9, "in", 10)])
    t = serialise_v11(m).split("\n")
    assert t[:2] == ["HOUSE h (metres, y up)", "OBJECTS: [id,label,[cx,cy,cz],[sx,sy,sz]]"]
    assert t[2] == '[9,"a",[0.0,0.0,0.0],[1.0,1.0,1.0]]'
    assert t[3] == '[10,"chest_of_drawers",[0.0,1.0,2.0],[1.0,2.0,3.5]]'   # -0.0 -> 0.0; 1.005 -> 1.0; ids numeric
    assert t[4] == "RELATIONS: [subject id,relation,object id]"
    assert t[5:] == ['[9,"in",10]', '[9,"next_to",10]', '[10,"on",9]']
    assert " " not in "".join(t[2:4] + t[5:])
    assert not serialise_v11(m).endswith("\n")


@needs_data
def test_g3_roundtrip_v11_all_30(maps):
    for s, m in maps.items():
        t = serialise_v11(m)
        r = parse_v11(t)
        assert serialise_v11(r) == t, s
        assert r.scene == s and [o.id for o in r.objects] == [o.id for o in m.objects]
        assert [o.label for o in r.objects] == [o.label for o in m.objects]
        assert r.relations == sorted(m.relations)
        for a, b in zip(r.objects, m.objects):
            assert np.allclose(a.center, np.round(b.center, 1), atol=1e-9) and np.allclose(a.size, np.round(b.size, 1), atol=1e-9)


def test_serialise_v12_format():
    m = Map("h", [MapObj(10, "chest_of_drawers", np.array([-0.001, 1.005, 2.0]), np.array([1, 2, 3.456])), MapObj(9, "a", np.zeros(3), np.ones(3))],
            [(10, "on", 9), (9, "next_to", 10), (9, "in", 10), (9, "on", 10), (9, "on", 9)])
    t = serialise(m).split("\n")
    assert t == ["HOUSE h (metres, y up)", "OBJECTS: [id,label,[cx,cy,cz]]", '[9,"a",[0.0,0.0,0.0]]', '[10,"chest_of_drawers",[0.0,1.0,2.0]]',
                 "RELATIONS on: [subject id,subject label,object id,object label]", '[9,"a",9,"a"]', '[9,"a",10,"chest_of_drawers"]',
                 '[10,"chest_of_drawers",9,"a"]',
                 "RELATIONS in: [subject id,subject label,object id,object label]", '[9,"a",10,"chest_of_drawers"]',
                 "RELATIONS next_to: [subject id,subject label,object id,object label]", '[9,"a",10,"chest_of_drawers"]']
    e = serialise(Map("h", [], [])).split("\n")
    assert [x.split(":")[0] for x in e[2:]] == ["RELATIONS on", "RELATIONS in", "RELATIONS next_to"]   # sections always present
    assert " " not in "".join(t[2:4] + t[5:7])
    bad = serialise(m).replace('[9,"a",10,"chest_of_drawers"]', '[9,"b",10,"chest_of_drawers"]', 1)
    with pytest.raises(AssertionError):
        parse(bad)


@needs_data
def test_g3_roundtrip_v12_all_30(maps):
    for s, m in maps.items():
        t = serialise(m)
        r = parse(t)
        assert serialise(r) == t, s
        assert r.scene == s and [o.id for o in r.objects] == [o.id for o in m.objects]
        assert [o.label for o in r.objects] == [o.label for o in m.objects]
        assert sorted(r.relations) == sorted(m.relations)
        for a, b in zip(r.objects, m.objects):
            assert np.allclose(a.center, np.round(b.center, 1), atol=1e-9) and not a.size.any()
        assert parse_any(t).relations == r.relations and parse_any(serialise_v11(m)).scene == s
        assert parse_any(serialise_v11(m)).objects[0].size.any()   # v1.1 text still parsed with sizes


# ---- questions -------------------------------------------------------------------------------------------------------
def test_article_plural():
    assert [article(x) for x in ("apple", "chair", "umbrella", "tv")] == ["an", "a", "an", "a"]
    for s, p in [("chair", "chairs"), ("box", "boxes"), ("buzz", "buzzes"), ("bench", "benches"), ("dish", "dishes"), ("glass", "glasses"),
                 ("shelves", "shelves"), ("stairs", "stairs"), ("pantry", "pantries"), ("toy", "toys"), ("tv", "tvs"), ("y", "ys"),
                 ("coffee table", "coffee tables")]:
        assert plural(s) == p, s
    assert spoken("chest_of_drawers") == "chest of drawers"


@needs_data
def test_generator_deterministic_bytes(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write(a)
    write(b)
    assert a.read_bytes() == b.read_bytes() and b"\r" not in a.read_bytes()
    assert hashlib.sha256(a.read_bytes()).hexdigest() == hashlib.sha256(b.read_bytes()).hexdigest()
    assert read(a) == read(b)


@needs_data
def test_g1_bars_and_structure(qs, maps):
    assert len(qs) >= 450
    assert all(sum(q["type"] == t for q in qs) >= 80 for t in TYPES)
    assert all(sum(q["scene"] == s for q in qs) >= 10 for s in maps)
    assert len({q["qid"] for q in qs}) == len(qs)
    for q in qs:
        assert q["qid"].startswith(f"{q['scene']}:{q['type']}:")
        assert sum(r["scene"] == q["scene"] and r["type"] == q["type"] for r in qs) <= 4   # quota never exceeded
    ex = [q for q in qs if q["type"] == "existence"]
    assert {q["polarity"] for q in ex} == {"pos", "neg"} and all((q["answer"] == "yes") == (q["polarity"] == "pos") for q in ex)


@needs_data
def test_seed_rule_existence(maps, vocab):
    m = maps["102344457"]
    rng = random.Random("114:102344457:existence")
    have = sorted({o.label for o in m.objects})
    want = rng.sample(have, 2) + rng.sample(sorted(set(vocab) - set(have)), 2)
    assert [q["label"] for q in generate(m, vocab) if q["type"] == "existence"] == want


@needs_data
def test_answers_are_consistent_with_map(qs, maps):
    for q in qs:
        m = maps[q["scene"]]
        n = len(m.with_label(q["label"]))
        if q["type"] == "existence":
            assert (n >= 1) == (q["answer"] == "yes")
        elif q["type"] == "counting":
            assert q["answer"] == n and 1 <= n <= 6
        elif q["type"] == "relation":
            (a,) = m.with_label(q["label"])
            assert {m.obj(b).label for b in m.related(a.id, q["pred"])} == {q["answer"]} and q["answer"] != q["label"]
        else:
            assert n >= 2 and q["label"] != q["label_b"]
            hits = [a.id for a in m.with_label(q["label"]) if any(m.obj(b).label == q["label_b"] for b in m.related(a.id, q["pred"]))]
            assert hits == [q["answer_id"]] and q["answer"] == f"obj_{hits[0]}"


# ---- parser (§5) -----------------------------------------------------------------------------------------------------
def test_norm():
    assert P.norm("  Chest_Of-Drawers\n\t. ") == "chest of drawers ."


def test_parse_existence():
    f = P.parse_existence
    assert f("Yes") == "yes" and f("No.") == "no" and f("no, there is not") == "no" and f("Yes, there is no doubt") == "yes"
    assert f("There is no chair") == "no" and f("nobody knows") is None and f("yesterday") is None and f("maybe") is None and f("") is None
    assert f("not sure") is None   # 'not' is not 'no'


def test_parse_counting():
    f = P.parse_counting
    assert f("3") == 3 and f("There are 12 chairs") == 12 and f("three") == 3 and f("Twenty") == 20 and f("zero") == 0
    assert f("one or two") == 1 and f("2 (two)") == 2 and f("two, maybe 3") == 2 and f("about 3 or two") == 3   # earliest wins
    assert f("sixteen") == 16 and f("someone") is None and f("twenty-one") == 20 and f("none") is None and f("a few") is None
    assert f("3.5") == 3 and f("-3") == 3 and f("1520") == 1520


def test_parse_relation():
    f = P.parse_relation
    labs = ["table", "table_lamp", "coffee_table", "lamp", "chest_of_drawers", "tv"]
    assert f("table", labs) == "table" and f("Table lamp", labs) == "table_lamp" and f("table_lamp", labs) == "table_lamp"
    assert f("coffee table", labs) == "coffee_table"                      # earliest start beats the shorter, later 'table'
    assert f("the lamp on the table", labs) == "lamp"                    # earliest
    assert f("chest-of-drawers.", labs) == "chest_of_drawers"
    assert f("tables", labs) is None and f("stable", labs) is None and f("", labs) is None and f("sofa", labs) is None   # whole words only
    assert f("table lamp", ["table", "table_lamp"]) == "table_lamp"      # tie on start -> longer
    assert f("table lamp", ["table_lamp", "table"]) == "table_lamp"      # independent of candidate order


def test_parse_find():
    f = P.parse_find
    ids = {12, 3}
    assert f("obj_12", ids) == 12 and f("OBJ 12", ids) == 12 and f("The answer is obj_3.", ids) == 3 and f("obj-12", ids) == 12
    assert f("obj_99", ids) is None and f("obj_99 or obj_12", ids) is None   # first obj match must exist
    assert f("12", ids) == 12 and f("Object 12", ids) == 12 and f("id: 3.", ids) == 3 and f("", ids) is None and f("none", ids) is None
    assert f("3 or obj_12", ids) == 12                                       # obj pattern beats an earlier bare integer
    assert f("99, then 12", ids) is None and f("7", ids) is None             # first bare integer must exist


@needs_data
def test_find_rounding_error_bound(maps):
    """1-decimal rounding moves a centre by <= sqrt(3)*0.05 = 0.087 m."""
    for m in maps.values():
        r = parse(serialise(m))
        assert max(np.linalg.norm(a.center - b.center) for a, b in zip(m.objects, r.objects)) <= 0.0867


@needs_data
def test_question_file_sha_unchanged(tmp_path):
    f = tmp_path / "q.jsonl"
    write(f)
    assert hashlib.sha256(f.read_bytes()).hexdigest().startswith("d0d9ee72e031820ea66120d430efc22dd1cf8f9e4cc1db542b385d892cfca11f"[:64])


def test_score_one_find_distance():
    q = {"type": "find", "answer_center": [0.0, 0.0, 0.0]}
    m = Map("s", [MapObj(1, "a", np.array([1.0, 0, 0]), np.ones(3)), MapObj(2, "a", np.array([0.6, 0.9, 0]), np.ones(3)),
                  MapObj(3, "a", np.array([1.0, 0.1, 0]), np.ones(3))])
    assert score_one(q, "obj_1", m, None)["correct"]            # exactly 1.0 m
    assert not score_one(q, "obj_3", m, None)["correct"]        # 1.005 m
    assert not score_one(q, "obj_2", m, None)["correct"]        # sqrt(.36+.81) = 1.08 m
    r = score_one(q, "obj_7", m, None)
    assert r["invalid"] and not r["correct"]


def test_score_one_images_vocab_and_invalid():
    q = {"type": "relation", "answer": "table_lamp"}
    assert score_one(q, "Table lamp", None, ["table", "table_lamp"])["correct"]
    assert not score_one(q, "table", None, ["table", "table_lamp"])["correct"]
    assert score_one({"type": "counting", "answer": 2}, "many", None, None) == {"parsed": None, "correct": False, "invalid": True}


# ---- prompts ---------------------------------------------------------------------------------------------------------
def test_frame_indices():
    assert prompt.frame_indices(1521) == [0, 217, 434, 651, 869, 1086, 1303, 1520]
    assert prompt.frame_indices(8) == list(range(8))


@pytest.fixture(scope="module")
def prompts(qs, maps, vocab):
    text = {s: serialise(m) for s, m in maps.items()}
    return {r: [prompt.build(q, r, text[q["scene"]], vocab, 1521, "/root") for q in qs if not (r == "img7b" and q["type"] == "find")]
            for r in prompt.ROWS}


@needs_data
def test_prompt_shapes(prompts, qs):
    assert len(prompts["gt7b"]) == len(prompts["gt72b"]) == len(qs) and len(prompts["img7b"]) == sum(q["type"] != "find" for q in qs)
    for r, ps in prompts.items():
        for p in ps:
            assert set(p) == {"qid", "row", "system", "user_text", "images"} and p["row"] == r
    assert all(p["images"] == [] and p["user_text"].startswith("HOUSE ") for p in prompts["gt7b"])
    p = next(p for p in prompts["img7b"] if ":relation:" in p["qid"])
    assert len(p["images"]) == 8 and p["images"][1].endswith("/rgb/000217.jpg") and p["images"][0].startswith("/root/")
    assert "Answer format: one label from this list: " in p["user_text"] and "chest of drawers" in p["user_text"] and "_" not in p["user_text"]
    assert p["system"].endswith("photos of the house given above. Reply with the answer only, in the requested format, with no explanation.")
    g = prompts["gt7b"][0]
    f = next(p for p in prompts["gt7b"] if ":find:" in p["qid"])
    assert f["user_text"].endswith("Answer format: one object id (the number), exactly as written in the map")
    assert g["user_text"].endswith("\n\nQuestion: " + next(q for q in qs if q["qid"] == g["qid"])["question"] + "\nAnswer format: yes or no")


@needs_data
def test_context_recovers_map_and_vocab(prompts, vocab):
    g = prompts["gt7b"][0]
    m, v = context(g["user_text"])
    assert v is None and serialise(m) == g["user_text"].split("\n\nQuestion: ")[0]
    p = next(p for p in prompts["img7b"] if ":relation:" in p["qid"])
    assert context(p["user_text"]) == (None, vocab)


@needs_data
def test_img7b_find_counted_wrong(qs, prompts):
    ps = {p["qid"]: p for p in prompts["img7b"]}
    raw = {q["qid"]: "yes" if q["type"] == "existence" else "1" for q in qs if q["type"] in ("existence", "counting")}
    raw.update({q["qid"]: q["answer"] for q in qs if q["type"] == "relation"})
    s = summarise("img7b", score_row("img7b", qs, raw, ps))
    f = s["per_type"]["find"]
    assert f["n"] == 120 and f["correct"] == 0 and f["not_asked"] == 120 and f["invalid"] == 0
    assert s["per_type"]["relation"]["accuracy"] == 1.0 and s["n"] == len(qs)
    with pytest.raises(AssertionError):
        score_row("gt7b", qs, raw, ps)   # a missing prompt is an error outside img7b-find


# ---- solver (G2) + mutation ------------------------------------------------------------------------------------------
def run_solver(maps, vocab, qs):
    res = solver.run(qs, maps, vocab)
    return {t: np.mean([r["correct"] for r in res if r["type"] == t]) for t in TYPES}, np.mean([r["correct"] for r in res])


@needs_data
def test_solver_100(maps, vocab, qs):
    per, tot = run_solver(maps, vocab, qs)
    assert tot == 1.0 and all(v == 1.0 for v in per.values())


@needs_data
def test_mutation_count_off_by_one(maps, vocab, qs, monkeypatch):
    monkeypatch.setattr(solver, "_count", lambda m, label: len(m.with_label(label)) + 1)
    per, tot = run_solver(maps, vocab, qs)
    assert per["counting"] == 0.0 and per["existence"] < 1.0 and tot < 1.0


@needs_data
def test_mutation_ignore_predicate(maps, vocab, qs, monkeypatch):
    """Predicate ignored (union of all relations): relation / find answers go wrong and the scorer must show it."""
    monkeypatch.setattr(solver, "_related", lambda m, a, pred: [b for p in ("on", "in", "next_to") for b in m.related(a, p)])
    per, tot = run_solver(maps, vocab, qs)
    assert tot < 1.0 and per["relation"] < 1.0 and per["find"] < 1.0 and per["existence"] == per["counting"] == 1.0


@needs_data
def test_mutation_first_related_is_scored_wrong(maps, vocab, qs, monkeypatch):
    """Solver answers with the first related object whatever the predicate."""
    monkeypatch.setattr(solver, "_related", lambda m, a, pred: [b for p in ("on", "in", "next_to") for b in m.related(a, p)][:1])
    per, tot = run_solver(maps, vocab, qs)
    assert tot < 1.0 and (per["relation"] < 1.0 or per["find"] < 1.0)


def test_find_rule_label_plus_dist():
    q = {"type": "find", "label": "chair", "answer_center": [0.0, 0.0, 0.0]}
    m = Map("s", [MapObj(1, "chair", np.array([0.5, 0, 0]), np.ones(3)), MapObj(2, "chair", np.array([5.0, 0, 0]), np.ones(3)),
                  MapObj(3, "table", np.array([0.5, 0, 0]), np.ones(3))])
    ok = lambda raw, rule: score_one(q, raw, m, None, rule)["correct"]
    assert ok("1", "label+dist") and ok("1", "dist")           # right label, near
    assert not ok("2", "label+dist") and not ok("2", "dist")   # right label, far
    assert not ok("3", "label+dist") and ok("3", "dist")       # wrong label (the reference object), near
    assert not ok("9", "label+dist")                           # not in the map
    with pytest.raises(AssertionError):
        ok("1", "nope")


@needs_data
def test_solver_100_both_find_rules(maps, vocab, qs):
    for rule in ("dist", "label+dist"):
        assert all(r["correct"] for r in solver.run(qs, maps, vocab, rule))
