"""Answer parser (P1B_PROTOCOL §5). Fixed, no per-model change. Each parser returns the value or None (invalid)."""
from __future__ import annotations

import re
from collections.abc import Iterable

from spatialmem.p1b.questions import spoken

NUMBER_WORDS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
                "sixteen seventeen eighteen nineteen twenty").split()
_NUM = re.compile(r"\b(\d+|" + "|".join(NUMBER_WORDS) + r")\b")


def norm(r: str) -> str:
    return " ".join(r.lower().replace("_", " ").replace("-", " ").split())


def parse_existence(r: str) -> str | None:
    m = re.search(r"\b(yes|no)\b", norm(r))
    return m.group(1) if m else None


def parse_counting(r: str) -> int | None:
    m = _NUM.search(norm(r))
    if m is None:
        return None
    w = m.group(1)
    return int(w) if w.isdigit() else NUMBER_WORDS.index(w)


def parse_relation(r: str, labels: Iterable[str]) -> str | None:
    """Candidate whose whole-phrase match starts earliest wins; ties -> the longer candidate. Returns the label."""
    n = norm(r)
    best: tuple[int, int, str] | None = None
    for L in labels:
        s = spoken(L)
        m = re.search(rf"\b{re.escape(s)}\b", n)
        if m and (best is None or (m.start(), -len(s)) < best[:2]):
            best = (m.start(), -len(s), L)
    return best[2] if best else None


def parse_find(r: str, ids: set[int]) -> int | None:
    """First `obj <digits>`, else the first bare integer; must exist in the given map."""
    n = norm(r)
    m = re.search(r"obj (\d+)", n) or re.search(r"\b(\d+)\b", n)
    return int(m.group(1)) if m and int(m.group(1)) in ids else None
