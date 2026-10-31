"""HSSD GT label rule — frozen 2026-10-27b before any HSSD number (docs/CONTRACT.md "HSSD ground truth").

label = objects.csv main_category, else the template's lexicon class (object_config semantic_id), else the
objects.csv WordNet synset head ("showerhead.n.01" -> "showerhead"), else "undefined" (excluded from GT objects,
kept in all_objects).
"""
from __future__ import annotations

RULE = "main_category > lexicon(semantic_id) > wnsynset head > undefined (spatialmem/hssd/labels.py)"
EXCLUDED = frozenset({"undefined", "unknown"})


def label(rec: dict) -> str:
    for k in ("category_main", "category_lexicon"):
        v = (rec.get(k) or "").strip()
        if v and v not in EXCLUDED:
            return v
    syn = (rec.get("wnsynset") or "").strip()
    if syn:
        return syn.split(".")[0]
    return "undefined"


def group_label(members: list[dict]) -> str:
    """GT object label: a lone instance -> label(); decomposed parts -> the parent's main_category (members carry it),
    else the most common member label (ties: alphabetical)."""
    if len(members) == 1 and not members[0].get("parent"):
        return label(members[0])
    mc = (members[0].get("category_main") or "").strip()
    if mc and mc not in EXCLUDED:
        return mc
    from collections import Counter
    c = Counter(label(m) for m in members)
    return min(c, key=lambda k: (-c[k], k))
