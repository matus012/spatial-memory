"""HSSD canonical vocabulary (task 003 step 1)."""
import pytest

from spatialmem.gt.relations import WALL_MOUNTED
from spatialmem.hssd import vocab


def test_frozen_map_properties():
    t = vocab.table()
    assert vocab.canonical("shelf") == vocab.canonical("shelves") == "shelves"
    assert vocab.canonical("wall_lamp") == "wall_lamp" and vocab.canonical("ceiling_lamp") == "ceiling_lamp"
    assert vocab.canonical("not_a_label") == "not_a_label"
    for k, v in t.items():
        assert (k in WALL_MOUNTED) == (v in WALL_MOUNTED)
    assert len(vocab.vocabulary()) == 236


def test_build_rules():
    counts = {"a": 1, "b": 5, "c": 5, "picture": 3, "x": 2}
    assert vocab.build([["a", "b", "c"]], counts) == {"a": "b", "b": "b", "c": "b"}   # most frequent, tie alphabetical
    with pytest.raises(AssertionError):
        vocab.build([["a", "b"], ["b", "c"]], counts)          # label in two groups
    with pytest.raises(AssertionError):
        vocab.build([["picture", "x"]], counts)                # mixes wall-mounted and other
    with pytest.raises(AssertionError):
        vocab.build([["a", "zz"]], counts)                     # not a GT label
