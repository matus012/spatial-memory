"""ours v1: prompt rule, Prediction schema, stream loaders on synthetic data, --set overrides."""
import json

import numpy as np
from PIL import Image

from spatialmem.baselines.ovo import prompt
from spatialmem.eval.contract import Prediction
from spatialmem.ours import run
from spatialmem.ours.association import MapObject
from spatialmem.render import proof


def test_prompt_rule():
    assert prompt("wall_lamp") == "a photo of a wall lamp."
    assert prompt("t-shirt") == "a photo of a t shirt."


def toy_objects() -> list[MapObject]:
    rng = np.random.default_rng(0)
    f = np.eye(3)
    return [MapObject(rng.uniform(0, 1, (50, 3)), [(10, f[0])], 2),
            MapObject(rng.uniform(0, 1, (50, 3)) * (1, 1, 0.3) + (0, 0, 1.0), [(10, f[1])], 2),
            MapObject(rng.uniform(0, 1, (50, 3)) + (3, 0, 0), [(10, f[2])], 3)]


def test_prediction_schema_valid(tmp_path):
    classes = ["table", "lamp", "chair"]
    for ds in ("hssd", "replica"):
        pred = run.to_prediction("toy", ds, toy_objects(), np.eye(3), classes)
        f = tmp_path / f"{ds}.json"
        pred.to_json(f)
        back = Prediction.from_json(f)
        assert [o.label for o in back.objects] == classes
        assert all((o.aabb_max >= o.aabb_min).all() for o in back.objects)
        assert len({o.id for o in back.objects}) == 3
        assert Prediction.from_json(json.loads(f.read_text())).scene == "toy"
        if ds == "replica":
            assert back.sem_points.shape == (150, 3) and list(np.bincount(back.sem_labels)) == [50, 50, 50]
        else:
            assert back.sem_points is None


def test_replica_loader_synthetic(tmp_path, monkeypatch):
    d = tmp_path / "nice" / "Replica" / "roomX" / "results"
    d.mkdir(parents=True)
    T = np.eye(4)
    T[:3, 3] = (1, 2, 3)
    np.savetxt(d.parent / "traj.txt", np.stack([np.eye(4).ravel(), T.ravel()]))
    for i in range(2):
        Image.fromarray(np.full((680, 1200, 3), 10 * i, np.uint8)).save(d / f"frame{i:06d}.jpg")
        Image.fromarray(np.full((680, 1200), 6553 * (i + 1), np.uint16)).save(d / f"depth{i:06d}.png")
    monkeypatch.setattr(proof, "DATA", tmp_path)
    st = run.replica_stream("roomX")
    assert st.n == 2 and st.intr == proof.REPLICA_INTR
    assert np.allclose(st.poses[1], T)
    assert np.allclose(st.depth(1), 2 * 6553 / 6553.5) and st.rgb(0).shape == (680, 1200, 3)


def test_hssd_loader_synthetic(tmp_path):
    D = tmp_path / "s"
    (D / "rgb").mkdir(parents=True)
    (D / "depth").mkdir()
    intr = {"W": 8, "H": 6, "fx": 4.0, "fy": 4.0, "cx": 3.5, "cy": 2.5}
    (D / "meta.json").write_text(json.dumps({"intrinsics": intr, "depth_scale": 6553.5, "n_frames": 1}))
    np.savetxt(D / "poses.txt", np.eye(4).reshape(1, 16))
    Image.fromarray(np.zeros((6, 8, 3), np.uint8)).save(D / "rgb" / "000000.jpg")
    Image.fromarray(np.full((6, 8), 13107, np.uint16)).save(D / "depth" / "000000.png")
    st = run.hssd_stream("s", tmp_path)
    assert st.n == 1 and st.intr.fx == 4.0 and np.allclose(st.depth(0), 2.0) and st.poses.shape == (1, 4, 4)


def test_config_set_override(tmp_path):
    c = tmp_path / "c.yaml"
    c.write_text("lam: 0.5\nstride: 5\n")
    cfg = run.load_config(c, ["lam=1", "stride=10"])
    assert cfg == {"lam": 1.0, "stride": 10} and isinstance(cfg["lam"], float)
