"""Prompts (P1B_PROTOCOL §4). `python -m spatialmem.p1b.prompt --row gt7b|img7b|gt72b --stream-root R --out F.jsonl`."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spatialmem.p1b.gtmap import RUNS, eval_scenes, load, vocabulary
from spatialmem.p1b.questions import read, spoken
from spatialmem.p1b.serialise import serialise, serialise_v11

ROWS = ("gt7b", "img7b", "gt72b")
N_FRAMES = 8
SYS_MAP = ("You answer questions about one house using only the map of the house given below. "
           "Reply with the answer only, in the requested format, with no explanation.")
SYS_IMG = ("You answer questions about one house using only the photos of the house given above. "
           "Reply with the answer only, in the requested format, with no explanation.")
FMT = {"existence": "yes or no", "counting": "a single integer",
       "relation": "one object label, exactly as written in the map",
       "find": "one object id (the number), exactly as written in the map"}


def frame_indices(n: int) -> list[int]:
    return [int(i) for i in np.round(np.linspace(0, n - 1, N_FRAMES))]


def build(q: dict, row: str, map_text: str | None, vocab: list[str], n_frames: int, stream_root: str) -> dict:
    """One prompt record. Map rows: `map_text` is prepended; images row: 8 frame paths, relation answers from `vocab`."""
    if row == "img7b":
        fmt = "one label from this list: " + ", ".join(sorted(spoken(L) for L in vocab)) if q["type"] == "relation" else FMT[q["type"]]
        imgs = [f"{stream_root}/{q['scene']}/rgb/{i:06d}.jpg" for i in frame_indices(n_frames)]
        return {"qid": q["qid"], "row": row, "system": SYS_IMG, "user_text": f"Question: {q['question']}\nAnswer format: {fmt}", "images": imgs}
    return {"qid": q["qid"], "row": row, "system": SYS_MAP,
            "user_text": f"{map_text}\n\nQuestion: {q['question']}\nAnswer format: {FMT[q['type']]}", "images": []}


def build_all(row: str, stream_root: str, version: str = "v11") -> list[dict]:
    assert row in ROWS, row
    scenes = eval_scenes()
    maps = {s: load(s) for s in scenes}
    vocab = vocabulary(list(maps.values()))
    ser = {"v11": serialise_v11, "v12": serialise}[version]
    text = {s: ser(m) for s, m in maps.items()}
    nf = {s: json.loads((RUNS / s / "meta.json").read_text())["n_frames"] for s in scenes}   # == poses.txt lines == PERUN jpgs
    return [build(q, row, text[q["scene"]], vocab, nf[q["scene"]], stream_root)
            for q in read() if not (row == "img7b" and q["type"] == "find")]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", required=True, choices=ROWS)
    ap.add_argument("--stream-root", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--version", default="v11", choices=("v11", "v12"), help="map text version (img7b has no map)")
    a = ap.parse_args()
    ps = build_all(a.row, a.stream_root.rstrip("/"), a.version)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes("".join(json.dumps(p) + "\n" for p in ps).encode())
    print(f"{a.out}: {len(ps)} prompts")
