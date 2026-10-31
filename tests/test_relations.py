"""docs/RELATIONS.md rules on hand-built scenes (z up unless stated)."""
from spatialmem.eval.contract import PredObject
from spatialmem.gt.relations import compute_relations


def o(i, lo, hi):
    return PredObject(i, "x", lo, hi)


def keys(objs, up=2):
    return {r.key() for r in compute_relations(objs, up=up)}


def test_cup_on_table():
    table = o(1, [0, 0, 0], [1.2, 0.8, 0.75])
    cup = o(2, [0.5, 0.3, 0.74], [0.6, 0.4, 0.85])
    assert keys([table, cup]) == {(2, "on", 1)}


def test_on_picks_highest_support():
    rug = o(1, [-1, -1, 0], [3, 3, 0.02])
    table = o(2, [0, 0, 0.02], [1, 1, 0.75])
    book = o(3, [0.2, 0.2, 0.75], [0.4, 0.5, 0.8])
    k = keys([rug, table, book])
    assert (2, "on", 1) in k and (3, "on", 2) in k and (3, "on", 1) not in k


def test_book_in_shelf():
    shelf = o(1, [0, 0, 0], [1, 0.4, 2])
    book = o(2, [0.1, 0.05, 0.9], [0.15, 0.3, 1.2])
    assert keys([shelf, book]) == {(2, "in", 1)}


def test_next_to_symmetric_and_distance():
    a = o(5, [0, 0, 0], [0.5, 0.5, 1])
    b = o(3, [0.75, 0, 0], [1.25, 0.5, 1])     # gap 0.25 m
    c = o(9, [2.0, 0, 0], [2.5, 0.5, 1])       # gap 0.75 m from b
    assert keys([a, b, c]) == {(3, "next_to", 5)}


def test_next_to_needs_vertical_overlap():
    a = o(1, [0, 0, 0], [0.5, 0.5, 0.5])
    lamp = o(2, [0.6, 0, 2.0], [0.9, 0.3, 2.3])  # hanging above, near in 2D
    assert keys([a, lamp]) == set()


def test_on_excludes_next_to_and_degenerate_dropped():
    table = o(1, [0, 0, 0], [1, 1, 0.75])
    cup = o(2, [0.4, 0.4, 0.75], [0.5, 0.5, 0.9])
    flat = o(3, [0.1, 0.1, 0.75], [0.3, 0.3, 0.755])  # 5 mm thick -> dropped
    assert keys([table, cup, flat]) == {(2, "on", 1)}


def test_up_axis_y():
    table = o(1, [0, 0, 0], [1.2, 0.75, 0.8])
    cup = o(2, [0.5, 0.74, 0.3], [0.6, 0.85, 0.4])
    assert keys([table, cup], up=1) == {(2, "on", 1)}
