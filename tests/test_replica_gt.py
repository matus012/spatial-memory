"""GT extraction on a synthetic Replica-format scene (quad PLY with face.object_id + info_semantic.json)."""
import json

import numpy as np
from plyfile import PlyData, PlyElement

import spatialmem.gt.replica as R


def cube_quads(lo, hi, base):
    """8 vertices + 6 quads of an axis-aligned box."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    v = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    q = np.array([[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]) + base
    return v, q


def make_scene(tmp_path, scene="room0"):
    # floor (class 40), ceiling (31), table (80), a vase (91) on it, a book (13) next to the vase,
    # and an object of class 1 (backpack) that the 51-class protocol does not map
    parts = [(1, 40, [0, 0, -0.05], [4, 4, 0.0]), (2, 31, [0, 0, 2.6], [4, 4, 2.65]),
             (3, 80, [1, 1, 0], [2, 2, 0.75]), (4, 91, [1.2, 1.2, 0.74], [1.4, 1.4, 1.0]),
             (5, 13, [1.5, 1.2, 0.75], [1.7, 1.3, 0.95]), (6, 1, [3, 3, 0], [3.3, 3.3, 0.4])]
    V, Q, O = [], [], []
    for oid, _, lo, hi in parts:
        v, q = cube_quads(lo, hi, sum(len(x) for x in V))
        V.append(v), Q.append(q), O.append(np.full(len(q), oid))
    V, Q, O = np.concatenate(V), np.concatenate(Q), np.concatenate(O)
    vert = np.array([tuple(p) for p in V], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    face = np.empty(len(Q), dtype=[("vertex_indices", "i4", (4,)), ("object_id", "i4")])
    face["vertex_indices"], face["object_id"] = Q, O
    d = tmp_path / "replica_v1" / R.v1_name(scene) / "habitat"
    d.mkdir(parents=True)
    PlyData([PlyElement.describe(vert, "vertex"), PlyElement.describe(face, "face")], text=False).write(
        str(d / "mesh_semantic.ply"))
    names = {40: "floor", 31: "ceiling", 80: "table", 91: "vase", 13: "book", 1: "backpack"}
    info = {"classes": [{"id": k, "name": v} for k, v in names.items()],
            "objects": [{"id": oid, "class_id": cid} for oid, cid, _, _ in parts]}
    (d / "info_semantic.json").write_text(json.dumps(info))
    return parts


def test_build_gt_synthetic(tmp_path, monkeypatch):
    parts = make_scene(tmp_path)
    monkeypatch.setattr(R, "DATA", tmp_path)
    gt, extra = R.build_gt("room0", with_vertices=False)
    assert extra["up_axis"] == 2
    assert extra["skipped"] == {"ignored_or_unmapped": 1, "structural": 2, "missing_class": 0}
    by_id = {o.id: o for o in gt.objects}
    assert set(by_id) == {3, 4, 5}
    assert by_id[3].label == "table" and by_id[4].label == "vase" and by_id[5].label == "book"
    for oid, _, lo, hi in parts:
        if oid in by_id:
            assert np.allclose(by_id[oid].aabb_min, lo, atol=1e-6) and np.allclose(by_id[oid].aabb_max, hi, atol=1e-6)
    rel = {r.key() for r in gt.relations}
    assert (4, "on", 3) in rel and (5, "on", 3) in rel and (4, "next_to", 5) in rel


def test_protocol_reduce():
    p = R.protocol()
    assert p.num_classes == 51 and p.names[0] == "wall" and p.names[50] == "switch"
    orig = np.array([93, 31, 40, 79, -1, -2, 256, 0, 1, 2])
    assert p.reduce(orig).tolist() == [0, 1, 2, 50, -1, -1, -1, -1, -1, -1]


def test_up_axis_rejects_flipped(tmp_path, monkeypatch):
    make_scene(tmp_path)
    monkeypatch.setattr(R, "DATA", tmp_path)
    verts, faces, fo = R.read_semantic_mesh("room0")
    info = R.read_info("room0")
    oc, cn = R.object_classes(info)
    vs = R.object_vertex_sets(faces, fo)
    assert R.up_axis(verts, vs, oc, cn) == 2
    try:
        R.up_axis(-verts, vs, oc, cn)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
