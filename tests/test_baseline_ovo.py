"""OVO adapter: ckpt-equivalent npz -> CONTRACT Prediction (synthetic, no torch)."""
import numpy as np

from spatialmem.baselines import ovo
from spatialmem.eval.contract import Prediction


def _write(tmp_path, dataset="replica"):
    classes = ovo.vocabulary(dataset)
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(len(classes), 8)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    c_a, c_b = 3, 7
    # instance 5: box [0,1]^3 -> class c_a; instance 9: 30 pts -> c_b, on top of 5; instance 2: 4 pts (dropped); -1 = none
    p5 = rng.uniform(0, 1, (50, 3)); p5[0] = 0; p5[1] = 1
    p9 = rng.uniform(0, 0.2, (30, 3)) + np.array([0.4, 0.4, 1.0]); p9[0] = [0.4, 0.4, 1.0]; p9[1] = [0.6, 0.6, 1.2]
    p2 = rng.uniform(5, 6, (4, 3)); pn = rng.uniform(8, 9, (6, 3))
    xyz = np.concatenate([p5, p9, p2, pn]).astype(np.float32)
    oid = np.array([5] * 50 + [9] * 30 + [2] * 4 + [-1] * 6)
    feats = np.stack([emb[c_a] * 3 + 0.01, emb[c_b] * 2, emb[0]]).astype(np.float32)   # unnormalised on purpose
    np.savez(tmp_path / "s.npz", xyz=xyz, obj_ids=oid, ids=np.array([5, 9, 2]), feats=feats)
    np.savez(tmp_path / "t.npz", classes=np.array(classes), emb=emb, template=np.array(ovo.TEMPLATE))
    return classes, c_a, c_b


def test_predict_replica(tmp_path):
    classes, c_a, c_b = _write(tmp_path)
    pred, st = ovo.predict("s", "replica", tmp_path / "s.npz", tmp_path / "t.npz")
    assert st["instances"] == 3 and st["kept"] == 2 and st["dropped_lt_10_pts"] == 1
    objs = {o.id: o for o in pred.objects}
    assert set(objs) == {5, 9} and objs[5].label == classes[c_a] and objs[9].label == classes[c_b]
    assert np.allclose(objs[5].aabb_min, 0) and np.allclose(objs[5].aabb_max, 1)
    assert pred.sem_points.shape == (90, 3)
    assert (pred.sem_labels[:50] == c_a).all() and (pred.sem_labels[50:80] == c_b).all() and (pred.sem_labels[80:] == -1).all()
    # round trip through the contract JSON (+ sidecar)
    pred.to_json(tmp_path / "s.json")
    back = Prediction.from_json(tmp_path / "s.json")
    assert len(back.objects) == 2 and back.sem_labels.shape == (90,)


def test_predict_hssd_no_sem(tmp_path):
    classes, c_a, _ = _write(tmp_path, "hssd")
    pred, _ = ovo.predict("s", "hssd", tmp_path / "s.npz", tmp_path / "t.npz")
    assert pred.sem_points is None and {o.label for o in pred.objects} == {classes[3], classes[7]}


def test_prompt_rule():
    assert ovo.prompt("wall_lamp") == "a photo of a wall lamp."
    assert ovo.prompt("tv-screen") == "a photo of a tv screen."
    assert ovo.prompt("chair") == "a photo of a chair."
