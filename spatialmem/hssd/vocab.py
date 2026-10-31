"""HSSD canonical vocabulary (configs/hssd_synonyms.yaml; docs/CONTRACT.md 2026-10-27e).

`canonical(label)` maps a GT / predicted HSSD label to its class name. Built once from the frozen groups and the GT
label counts recorded in configs/hssd_vocab_counts.json (canonical name = most frequent member, ties alphabetical).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import yaml

from spatialmem.gt.relations import WALL_MOUNTED

ROOT = Path(__file__).resolve().parents[2]
SYNONYMS = ROOT / "configs" / "hssd_synonyms.yaml"
COUNTS = ROOT / "configs" / "hssd_vocab_counts.json"


def build(groups: list[list[str]], counts: dict[str, int]) -> dict[str, str]:
    seen: dict[str, int] = {}
    m: dict[str, str] = {}
    for gi, g in enumerate(groups):
        for x in g:
            assert x not in seen, f"{x} in groups {seen[x]} and {gi}"
            assert x in counts, f"{x} is not a GT label"
            seen[x] = gi
        wm = {x in WALL_MOUNTED for x in g}
        assert len(wm) == 1, f"group {g} mixes wall-mounted and other labels"
        canon = min(g, key=lambda x: (-counts[x], x))
        for x in g:
            m[x] = canon
    return m


@lru_cache(maxsize=1)
def table() -> dict[str, str]:
    return build(yaml.safe_load(SYNONYMS.read_text())["groups"], json.loads(COUNTS.read_text()))


def canonical(label: str) -> str:
    return table().get(label, label)


def vocabulary() -> list[str]:
    """Sorted canonical class names over the 35-scene GT (the HSSD label-assignment vocabulary)."""
    return sorted({canonical(x) for x in json.loads(COUNTS.read_text())})
