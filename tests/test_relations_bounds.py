"""Each docs/RELATIONS.md threshold on both sides, each tie-break, and the pruned-on / next_to rule."""
import pytest

from spatialmem.eval.contract import PredObject
from spatialmem.gt.relations import compute_relations

TABLE = ([0, 0, 0], [1, 1, 0.75])


def o(i, lo, hi):
    return PredObject(i, "x", lo, hi)


def keys(objs, up=2):
    return {r.key() for r in compute_relations(objs, up=up)}


def cup_at(gap, fo=1.0, height=0.1):
    """Cup bottom at table top + gap; fo = fraction of the cup footprint over the table (along x)."""
    w = 0.2
    x0 = 1.0 - fo * w
    return o(2, [x0, 0.4, 0.75 + gap], [x0 + w, 0.6, 0.75 + gap + height])


@pytest.mark.parametrize("gap,on", [(-0.079, True), (-0.081, False), (0.049, True), (0.051, False)])
def test_on_gap_bounds(gap, on):
    assert ((2, "on", 1) in keys([o(1, *TABLE), cup_at(gap)])) == on


@pytest.mark.parametrize("fo,on", [(0.51, True), (0.49, False)])
def test_on_footprint_bound(fo, on):
    assert ((2, "on", 1) in keys([o(1, *TABLE), cup_at(0.0, fo=fo)])) == on


def test_on_requires_top_above_support_top():
    # gap and footprint hold, but the box top (0.74) is below the table top (0.75)
    assert (2, "on", 1) not in keys([o(1, *TABLE), o(2, [0.4, 0.4, 0.70], [0.6, 0.6, 0.74])])


@pytest.mark.parametrize("ci,inside", [(0.91, True), (0.89, False)])
def test_in_containment_bound(ci, inside):
    shelf = o(1, [0, 0, 0], [1, 0.4, 2])
    x0 = 1.0 - ci * 0.1  # book of x-length 0.1 sticking out of the shelf's +x face by (1 - ci) of it
    book = o(2, [x0, 0.05, 0.9], [x0 + 0.1, 0.3, 1.2])
    assert ((2, "in", 1) in keys([shelf, book])) == inside


def test_in_requires_strictly_smaller_volume():
    a, b = o(1, [0, 0, 0], [1, 1, 1]), o(2, [0, 0, 0], [1, 1, 1])
    assert not any(p == "in" for _, p, _ in keys([a, b]))


def test_in_excluded_when_on_qualifies():
    # sunk 7 cm into the support, 5 mm above its top: on qualifies (gap -0.07, top above) AND
    # in qualifies (ci = 0.07 / 0.075 = 0.933 >= 0.9, smaller volume) -> only "on" is kept
    base = o(1, [0, 0, 0], [1, 1, 0.5])
    lid = o(2, [0.2, 0.2, 0.43], [0.8, 0.8, 0.505])
    assert keys([base, lid]) == {(2, "on", 1)}


@pytest.mark.parametrize("d,near", [(0.299, True), (0.301, False)])
def test_next_to_distance_bound(d, near):
    a = o(1, [0, 0, 0], [0.5, 0.5, 1])
    b = o(2, [0.5 + d, 0, 0], [1.0 + d, 0.5, 1])
    assert ((1, "next_to", 2) in keys([a, b])) == near


def test_next_to_distance_is_euclidean_2d():
    a = o(1, [0, 0, 0], [0.5, 0.5, 1])
    assert (1, "next_to", 2) in keys([a, o(2, [0.7, 0.7, 0], [1.0, 1.0, 1])])   # 0.283 m
    assert keys([a, o(3, [0.73, 0.73, 0], [1.0, 1.0, 1])]) == set()              # 0.325 m


def test_next_to_vertical_overlap_is_strict():
    a = o(1, [0, 0, 0], [0.5, 0.5, 1.0])
    b = o(2, [0.6, 0, 1.0], [1.0, 0.5, 1.5])  # u-intervals touch at 1.0: overlap 0
    assert keys([a, b]) == set()


def test_min_side_exactly_one_cm_is_kept():
    table = o(1, [-1, -1, 0], [1, 1, 0.75])
    sliver = o(2, [0.0, 0.0, 0.75], [0.01, 0.2, 0.9])  # x side == MIN_SIDE exactly (0.01 - 0.0)
    assert (2, "on", 1) in keys([table, sliver])


def test_min_side_bound():
    table = o(1, *TABLE)
    ok = o(2, [0.4, 0.4, 0.75], [0.5, 0.5, 0.7601])    # 1.01 cm tall -> kept
    bad = o(3, [0.6, 0.6, 0.75], [0.7, 0.7, 0.7599])   # 0.99 cm -> dropped
    k = keys([table, ok, bad])
    assert (2, "on", 1) in k and not any(3 in (s, t) for s, _, t in k)


def test_on_tie_break_top_then_area_then_id():
    cup = o(9, [0.4, 0.4, 0.745], [0.5, 0.5, 0.85])
    low, high = o(1, [0, 0, 0], [1, 1, 0.70]), o(2, [0, 0, 0], [1, 1, 0.74])
    k = keys([low, high, cup])
    assert (9, "on", 2) in k and (9, "on", 1) not in k
    big = o(3, [-1, -1, 0], [2, 2, 0.74])  # same top as `high`, larger footprint
    k = keys([big, high, cup])
    assert (9, "on", 2) in k and (9, "on", 3) not in k
    twin = o(1, [0, 0, 0], [1, 1, 0.74])   # identical to `high`: smaller id wins
    k = keys([twin, high, cup])
    assert (9, "on", 1) in k and (9, "on", 2) not in k


def test_in_tie_break_volume_then_id():
    item = o(9, [0.9, 0.9, 0.9], [1.0, 1.0, 1.0])
    outer, inner = o(1, [0, 0, 0], [2, 2, 2]), o(2, [0.5, 0.5, 0.5], [1.5, 1.5, 1.5])
    k = keys([outer, inner, item])
    assert (9, "in", 2) in k and (9, "in", 1) not in k
    twin = o(3, [0.5, 0.5, 0.5], [1.5, 1.5, 1.5])
    k = keys([twin, inner, item])
    assert (9, "in", 2) in k and (9, "in", 3) not in k


def test_pruned_on_still_blocks_next_to():
    """The cup qualifies `on` both same-height tables; one is pruned, yet no next_to with it either."""
    t1, t2 = o(1, [0, 0, 0], [1, 1, 0.75]), o(2, [1, 0, 0], [2, 1, 0.75])
    cup = o(9, [0.8, 0.4, 0.73], [1.2, 0.6, 0.9])  # sunk 2 cm: u-overlap > 0, footprint distance 0
    k = keys([t1, t2, cup])
    assert (9, "on", 1) in k and (9, "on", 2) not in k
    assert (2, "next_to", 9) not in k and (1, "next_to", 9) not in k
