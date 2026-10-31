"""docs/RELATIONS.md v2: wall-mounted classes are never subject or support of on / in."""
from spatialmem.eval.contract import PredObject
from spatialmem.gt.relations import compute_relations

SOFA = ([0, 0, 0], [2, 1, 0.5])


def lab(i, label, lo, hi):
    return PredObject(i, label, lo, hi)


def keys(objs, **kw):
    return {r.key() for r in compute_relations(objs, **kw)}


def sofa():
    return lab(1, "sofa", *SOFA)


def on_sofa(label):  # bottom on the sofa top, full footprint
    return lab(2, label, [0.5, 0.2, 0.5], [0.9, 0.4, 0.8])


def in_table(label):
    return [lab(1, "table", [0, 0, 0], [1, 1, 0.75]), lab(2, label, [0.4, 0.4, 0.3], [0.5, 0.5, 0.4])]


def test_picture_not_on_sofa():
    k = keys([sofa(), on_sofa("picture")])
    assert not any(p in ("on", "in") for _, p, _ in k)


def test_wallplug_not_in_table():
    assert not any(p in ("on", "in") for _, p, _ in keys(in_table("wall-plug")))


def test_non_wall_mounted_unchanged():
    assert (2, "on", 1) in keys([sofa(), on_sofa("cup")])
    assert (2, "in", 1) in keys(in_table("cup"))


def test_wall_mounted_as_support():
    panel = lab(1, "panel", [0, 0, 0], [1, 1, 0.75])
    cup = lab(2, "cup", [0.4, 0.4, 0.75], [0.5, 0.5, 0.9])
    assert (2, "on", 1) not in keys([panel, cup])
    assert (2, "on", 1) in keys([panel, cup], wall_mounted=frozenset())


def test_wall_mounted_in_support_box():
    cab = lab(1, "panel", [0, 0, 0], [1, 1, 2])
    book = lab(2, "book", [0.4, 0.4, 0.9], [0.5, 0.5, 1.0])
    assert (2, "in", 1) not in keys([cab, book])


def test_picture_over_sofa_next_to():
    # bottom/footprint put it "on" under v1 (so v1 blocks next_to); v2 frees the pair for next_to
    pic = lab(2, "picture", [0.5, 0.2, 0.48], [0.9, 0.4, 0.8])  # sunk 2 cm: u-overlap > 0
    assert (1, "next_to", 2) in keys([sofa(), pic])
    assert (1, "next_to", 2) not in keys([sofa(), pic], wall_mounted=frozenset())


def test_empty_set_reproduces_v1():
    objs = [sofa(), on_sofa("picture")]
    assert keys(objs, wall_mounted=frozenset()) == {(2, "on", 1)}
    objs = in_table("wall-plug")
    assert keys(objs, wall_mounted=frozenset()) == {(2, "in", 1)}
