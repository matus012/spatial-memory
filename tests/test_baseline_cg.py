"""ConceptGraphs adapter (spatialmem.baselines.cg) on a tiny synthetic CG export."""
from pathlib import Path

import numpy as np

from spatialmem.baselines import cg
from spatialmem.eval.contract import Prediction


def _write(tmp: Path) -> tuple[Path, Path]:
    classes = ["wall", "table", "book", "lamp"]
    emb = np.eye(4, 8, dtype=np.float32)                       # class k = basis vector k
    np.savez(tmp / "text.npz", classes=np.array(classes), emb=emb)
    table = np.array([[0, 0, 0], [1, 1, 0.7]], dtype=np.float32)          # top at z = 0.7
    book = np.array([[0.4, 0.4, 0.7], [0.6, 0.6, 0.75], [0.5, 0.5, 0.72]], dtype=np.float32)   # resting on it
    wall = np.array([[-1, -1, 0], [-0.99, 3, 2.5]], dtype=np.float32)
    ft = np.zeros((3, 8), np.float32)
    ft[0, 1], ft[0, 3] = 2.0, 1.0                              # table (unnormalised, cosine still picks table)
    ft[1, 2], ft[1, 1] = 0.9, 0.3                              # book
    ft[2, 0] = 1.0                                             # wall
    np.savez(tmp / "objects.npz", points=np.concatenate([table, book, wall]), counts=np.array([2, 3, 2]),
             clip_ft=ft, num_detections=np.array([7, 4, 9]))
    return tmp / "objects.npz", tmp / "text.npz"


def test_prediction(tmp_path: Path) -> None:
    o, t = _write(tmp_path)
    p = cg.prediction("s", o, t, up=2, semantic=True)
    assert [x.label for x in p.objects] == ["table", "book", "wall"]
    np.testing.assert_allclose(p.objects[1].aabb_min, [0.4, 0.4, 0.7], atol=1e-6)
    np.testing.assert_allclose(p.objects[1].aabb_max, [0.6, 0.6, 0.75], atol=1e-6)
    assert [x.score for x in p.objects] == [7.0, 4.0, 9.0]
    assert {(r.subj, r.pred, r.obj) for r in p.relations} == {(1, "on", 0)}    # wall (structural) never an endpoint
    assert p.sem_labels.tolist() == [1, 1, 2, 2, 2, 0, 0] and p.sem_points.shape == (7, 3)
    p.to_json(tmp_path / "s.json")
    q = Prediction.from_json(tmp_path / "s.json")
    assert len(q.objects) == 3 and q.sem_labels.tolist() == p.sem_labels.tolist()


def test_prompt_and_frame_check(tmp_path: Path) -> None:
    assert cg.prompt("tv-screen") == "a photo of a tv screen." and cg.prompt("wall_lamp") == "a photo of a wall lamp."
    assert "tv-screen" in cg.vocabulary("replica") and "wall_lamp" in cg.vocabulary("hssd")   # both vocabularies hit the rule
    o, t = _write(tmp_path)
    p = cg.prediction("s", o, t, up=2, semantic=False)
    assert p.sem_points is None
    gt = np.array([[[0, 0, 0], [1, 1, 0.7]], [[5, 5, 0], [6, 6, 1]]], dtype=float)
    med, inside = cg.frame_check(p, gt)
    assert abs(med - 0.025) < 1e-6 and abs(inside - 2 / 3) < 1e-9   # table in box 0, book 2.5 cm above it, wall outside
    shifted = gt + 10.0
    med, inside = cg.frame_check(p, shifted)
    assert med > 5 and inside == 0.0


def test_runtime(tmp_path: Path) -> None:
    w = tmp_path / "hssd" / "s1"
    w.mkdir(parents=True)
    (w / "time_det.tsv").write_text("det\t1000\t1100\t2048\t4\t10\n")
    (w / "time_map.tsv").write_text("map\t2000\t2050\t0\t8\t12\n")
    (w / "DONE").write_text("500 100 12\n")
    v = tmp_path / "replica" / "r0"
    v.mkdir(parents=True)
    (v / "time_det.tsv").write_text("det\t0\t30\tNA\t4\t10\treconstructed\n")
    (v / "time_map.tsv").write_text("map\t40\t50\t0\t4\t10\n")
    (v / "DONE").write_text("20 4 10\n")
    cg.runtime(tmp_path, tmp_path / "rt.csv")
    lines = (tmp_path / "rt.csv").read_text().splitlines()
    assert lines[1].split(",") == ["hssd", "s1", "500", "100", "100", "50", "1.5", "2.0", "4", "8", "10/12", ""]
    assert lines[2].split(",") == ["replica", "r0", "20", "4", "30", "10", "10.0", "NA", "4", "4", "10", "reconstructed"]
