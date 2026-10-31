"""Byte-stable JSON for results/p1b: floats rounded to 6, keys sorted, trailing newline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def rnd(x):
    if isinstance(x, (float, np.floating)):
        return round(float(x), 6)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, dict):
        return {k: rnd(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v) for v in x]
    return x


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(rnd(obj), sort_keys=True, indent=1) + "\n").encode())
