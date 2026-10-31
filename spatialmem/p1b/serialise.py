"""Map <-> text (P1B_PROTOCOL §3). `serialise(parse(t)) == t` byte for byte; no trailing newline."""
from __future__ import annotations

import json

import numpy as np

from spatialmem.p1b.gtmap import Map, MapObj

OBJ_HEAD11 = "OBJECTS: [id,label,[cx,cy,cz],[sx,sy,sz]]"
REL_HEAD11 = "RELATIONS: [subject id,relation,object id]"
HOUSE_PRE, HOUSE_POST = "HOUSE ", " (metres, y up)"


def _f(v: float) -> str:
    s = f"{v:.1f}"
    return "0.0" if s == "-0.0" else s


def _v(a: np.ndarray) -> str:
    return "[" + ",".join(_f(float(x)) for x in a) + "]"


def serialise_v11(m: Map) -> str:
    lines = [f"{HOUSE_PRE}{m.scene}{HOUSE_POST}", OBJ_HEAD11]
    for o in sorted(m.objects, key=lambda o: o.id):
        lines.append(f"[{o.id},{json.dumps(o.label)},{_v(o.center)},{_v(o.size)}]")
    lines.append(REL_HEAD11)
    for s, p, o in sorted(m.relations):
        lines.append(f'[{s},"{p}",{o}]')
    return "\n".join(lines)


def parse_v11(text: str) -> Map:
    lines = text.split("\n")
    assert lines[0].startswith(HOUSE_PRE) and lines[0].endswith(HOUSE_POST), lines[0]
    scene = lines[0][len(HOUSE_PRE):-len(HOUSE_POST)]
    assert lines[1] == OBJ_HEAD11, lines[1]
    k = lines.index(REL_HEAD11)
    objs = []
    for ln in lines[2:k]:
        i, label, c, sz = json.loads(ln)
        objs.append(MapObj(i, label, np.array(c, dtype=np.float64), np.array(sz, dtype=np.float64)))
    rels = []
    for ln in lines[k + 1:]:
        s, p, o = json.loads(ln)
        rels.append((s, p, o))
    return Map(scene, objs, rels)


OBJ_HEAD = "OBJECTS: [id,label,[cx,cy,cz]]"
REL_HEAD = "RELATIONS {}: [subject id,subject label,object id,object label]"
PREDS = ("on", "in", "next_to")


def serialise(m: Map) -> str:
    lab = {o.id: o.label for o in m.objects}
    lines = [f"{HOUSE_PRE}{m.scene}{HOUSE_POST}", OBJ_HEAD]
    for o in sorted(m.objects, key=lambda o: o.id):
        lines.append(f"[{o.id},{json.dumps(o.label)},{_v(o.center)}]")
    for p in PREDS:
        lines.append(REL_HEAD.format(p))
        for s, _, o in sorted(r for r in m.relations if r[1] == p):
            lines.append(f"[{s},{json.dumps(lab[s])},{o},{json.dumps(lab[o])}]")
    return "\n".join(lines)


def parse(text: str) -> Map:
    """Inverse of `serialise`. Sizes are not carried: parsed objects get a zero size vector."""
    lines = text.split("\n")
    assert lines[0].startswith(HOUSE_PRE) and lines[0].endswith(HOUSE_POST), lines[0]
    assert lines[1] == OBJ_HEAD, lines[1]
    heads = [lines.index(REL_HEAD.format(p)) for p in PREDS]
    assert heads == sorted(heads)
    objs = []
    for ln in lines[2:heads[0]]:
        i, label, c = json.loads(ln)
        objs.append(MapObj(i, label, np.array(c, dtype=np.float64), np.zeros(3)))
    lab = {o.id: o.label for o in objs}
    rels = []
    for p, a, b in zip(PREDS, heads, heads[1:] + [len(lines)]):
        for ln in lines[a + 1:b]:
            s, ls, o, lo = json.loads(ln)
            assert lab[s] == ls and lab[o] == lo, ln
            rels.append((s, p, o))
    return Map(lines[0][len(HOUSE_PRE):-len(HOUSE_POST)], objs, rels)


def parse_any(text: str) -> Map:
    return parse_v11(text) if text.split("\n")[1] == OBJ_HEAD11 else parse(text)
