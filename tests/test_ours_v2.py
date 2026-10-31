"""ours v2: vote rule, encoder registry, mask + desc cache round trip, Replica-dev loader, v2 -> v1 reduction."""
import collections
import json

import numpy as np
import yaml
from PIL import Image

from spatialmem.gt.replica import ROOT
from spatialmem.ours import encoders, run, v2
from spatialmem.ours.association import MapObject
from spatialmem.render import proof
from tests.test_ours_cache import CFG, SHA, stream

V2 = yaml.safe_load((ROOT / "configs" / "ours_v2.yaml").read_text())


def unit(*v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


# ------------------------------------------------------------------ label rule
TEXT = np.eye(3)


def obj(views: list[np.ndarray]) -> MapObject:
    return MapObject(np.zeros((1, 3)), [(100 - i, f) for i, f in enumerate(views)], len(views))


def test_vote_majority_beats_mean():
    # 2 views say class 1 (weakly), 1 view says class 0 strongly: mean -> 0, vote -> 1
    o = obj([unit(1, 0, 0), unit(0.45, 0.55, 0), unit(0.45, 0.55, 0)])
    assert v2.labels([o], TEXT, "mean")[0] == 0
    assert v2.labels([o], TEXT, "vote")[0] == 1


def test_vote_tie_falls_back_to_mean():
    o = obj([unit(1, 0, 0), unit(0, 1, 0), unit(0.1, 0, 1), unit(0, 0.1, 1)])     # 1 x c0, 1 x c1, 2 x c2
    assert v2.labels([o], TEXT, "vote")[0] == 2
    o = obj([unit(1, 0, 0), unit(0, 0.6, 0.5), unit(0.4, 0, 1), unit(0, 1, 0)])    # c0, c1, c2, c1
    v = (np.stack([f for _, f in o.views]) @ TEXT.T).argmax(1)
    assert list(v) == [0, 1, 2, 1] and v2.labels([o], TEXT, "vote")[0] == 1
    tie = obj([unit(0.2, 1, 0), unit(1, 0.9, 0), unit(1, 0, 0.1), unit(0, 1, 0)])  # c1, c0, c0, c1 -> tie
    assert list((np.stack([f for _, f in tie.views]) @ TEXT.T).argmax(1)) == [1, 0, 0, 1]
    mean = v2.labels([tie], TEXT, "mean")[0]
    assert v2.labels([tie], TEXT, "vote")[0] == mean == 1
    tie3 = obj([unit(1, 0, 0), unit(0, 1, 0), unit(0, 0, 1)])                     # three-way tie -> mean
    assert v2.labels([tie3], TEXT, "vote")[0] == v2.labels([tie3], TEXT, "mean")[0]


def test_mean_rule_is_v1_argmax():
    rng = np.random.default_rng(0)
    objs = [obj([unit(*rng.normal(size=3)) for _ in range(3)]) for _ in range(20)]
    v1 = (np.stack([o.descriptor() for o in objs]) @ TEXT.T).argmax(1)
    assert np.array_equal(v2.labels(objs, TEXT, "mean"), v1)
    assert len(v2.labels([], TEXT, "vote")) == 0


# ------------------------------------------------------------------ encoder registry
def test_encoder_registry(monkeypatch):
    assert set(encoders.ENCODERS) == {"siglip2_base", "siglip2_so400m", "clip_vith14_laion2b"}
    assert encoders.weights("siglip2_base") == ROOT / V2["siglip"] / "model.safetensors"   # = v1's SigLIP 2 base
    assert encoders.OpenClip.masks is encoders.Siglip.masks          # same blend / crop descriptor rule
    got = []
    monkeypatch.setattr(encoders, "Siglip", lambda p: got.append(("siglip", p)) or "S")
    monkeypatch.setattr(encoders, "OpenClip", lambda p: got.append(("open_clip", p)) or "O")
    assert [encoders.load(n) for n in ("siglip2_base", "siglip2_so400m", "clip_vith14_laion2b")] == ["S", "S", "O"]
    assert got[1][1].endswith("siglip2-so400m-patch14-384")
    assert got[2][1].endswith("open_clip_model.safetensors")


def test_v2_config_is_v1_plus_knobs():
    v1 = yaml.safe_load((ROOT / "configs" / "ours_v1.yaml").read_text())
    assert set(V2) == set(v1) | {"encoder", "label"}
    assert {k for k in v1 if v1[k] != V2[k]} <= {"descriptor", "tau"}          # the v2 search space (objective)
    assert V2["encoder"] in encoders.ENCODERS and V2["label"] in ("mean", "vote")


# ------------------------------------------------------------------ caches
def masker(rgb):
    """Frame-dependent masks so frames differ; one mask lifts to nothing (sky: depth 0 there)."""
    k = int(rgb[0, 0, 0])
    a, b, c = (np.zeros((30, 40), bool) for _ in range(3))
    a[2:14, 2 + k:18 + k] = True
    b[16:28, 22:38] = True
    b[20, 22] = False
    c[0:2, 0:3] = True
    return [a, b, c]


def depth(k):
    d = np.full((30, 40), 2.0)
    d[0:2, 0:3] = 0.0
    return d


class ContentEnc:
    """Descriptor computed from the mask's own pixels: equal outputs iff equal masks (and equal batching)."""

    def masks(self, rgb, masks, pad):
        f = np.stack([unit(m.sum(), *np.argwhere(m).mean(0), 1.0, rgb[0, 0, 0] + 1) for m in masks]).astype(np.float32)
        return {"blend": f, "crop": f[:, ::-1].copy()}


def synth(tmp_path) -> run.Stream:
    st = stream(tmp_path)
    st.depth = depth
    return st


def test_mask_cache_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    ms = [rng.random((30, 40)) < p for p in (0.05, 0.5, 0.95)] + [np.pad(np.ones((1, 1), bool), ((7, 22), (39, 0)))]
    rec = v2.MaskRecorder(ContentEnc())
    rec.masks(np.zeros((30, 40, 3), np.uint8), ms, 0)
    p = tmp_path / "masks.npz"
    v2.save_masks(p, "k", (30, 40), rec.box, rec.bits)
    z = v2.load(p)
    assert v2.key_of(p) == "k" and v2.key_of(tmp_path / "none.npz") is None
    assert all(np.array_equal(v2.mask(z, i), m) for i, m in enumerate(ms))


def test_desc_cache_round_trip_and_reduction(tmp_path):
    """Detections + masks recorded in the v1 detect loop; descriptors rebuilt from the masks cache == the ones
    computed live; v2 (label mean) prediction == v1 prediction, byte for byte, for both descriptors."""
    st, cfg = synth(tmp_path), {**CFG, **{k: V2[k] for k in ("encoder", "label")}, "label": "mean"}
    key = run.cache_key(cfg, SHA, st)
    rec, frames, nm = v2.MaskRecorder(ContentEnc()), [], []
    for n, dets in run.detect(st, cfg, masker, rec, collections.Counter()):
        frames.append(dets)
        nm.append(n)
    assert nm == [3, 3, 3] and [len(f) for f in frames] == [2, 2, 2] and len(rec.box) == 6
    text, classes = np.eye(4, 5, dtype=np.float32), list("abcd")
    run.save_cache(tmp_path / "detections.npz", key, frames, nm, text, classes)
    v2.save_masks(tmp_path / "masks.npz", key, (30, 40), rec.box, rec.bits)
    det = v2.load(tmp_path / "detections.npz")
    blend, crop = v2.descriptors(st, cfg, det["frame_ndet"], v2.load(tmp_path / "masks.npz"), ContentEnc(),
                                 collections.Counter())
    assert np.array_equal(blend, det["feat_blend"]) and np.array_equal(crop, det["feat_crop"])
    dk = v2.desc_key(key, "siglip2_base", "sha")
    assert dk != v2.desc_key(key, "siglip2_so400m", "sha") and dk != v2.desc_key("x", "siglip2_base", "sha")
    v2.save_desc(tmp_path / "desc.npz", dk, blend, crop, text, classes)
    dz = v2.load(tmp_path / "desc.npz")
    assert v2.key_of(tmp_path / "desc.npz") == dk and [str(c) for c in dz["classes"]] == classes
    for desc in ("blend", "crop"):
        c = {**cfg, "descriptor": desc}
        objs1, _ = run.map_scene(run.detect(st, c, masker, ContentEnc(), collections.Counter()), 3, c,
                                 collections.Counter())
        assert len(objs1) >= 1
        run.to_prediction("s", "replica", objs1, text, classes).to_json(tmp_path / f"v1_{desc}.json")
        _, pred, _ = v2.associate("s", "replica", v2.load(tmp_path / "detections.npz"), dz, c, classes, 3,
                                  collections.Counter())
        pred.to_json(tmp_path / f"v2_{desc}.json")
        assert (tmp_path / f"v1_{desc}.json").read_bytes() == (tmp_path / f"v2_{desc}.json").read_bytes(), desc


def test_associate_uses_encoder_descriptors(tmp_path):
    """The encoder's descriptors replace the v1 ones: other text space -> other labels."""
    st, cfg = synth(tmp_path), {**CFG, "label": "mean", "descriptor": "crop"}
    rec, frames, nm = v2.MaskRecorder(ContentEnc()), [], []
    for n, dets in run.detect(st, cfg, masker, rec, collections.Counter()):
        frames.append(dets)
        nm.append(n)
    run.save_cache(tmp_path / "d.npz", "k", frames, nm, np.eye(4, 5, dtype=np.float32), list("abcd"))
    det = v2.load(tmp_path / "d.npz")
    n = len(det["area"])
    f = np.tile(np.eye(3, dtype=np.float32)[2], (n, 1))
    dz = {"feat_blend": f, "feat_crop": f, "text": np.eye(3, dtype=np.float32)}
    _, pred, _ = v2.associate("s", "hssd", det, dz, cfg, list("xyz"), 3, collections.Counter())
    assert pred.objects and {o.label for o in pred.objects} == {"z"}


# ------------------------------------------------------------------ Replica-dev loader
def write_nice(root, scene, n=2):
    d = root / scene / "results"
    d.mkdir(parents=True)
    T = np.eye(4)
    T[:3, 3] = (1, 2, 3)
    np.savetxt(root / scene / "traj.txt", np.stack([np.eye(4).ravel(), T.ravel()][:n]))
    for i in range(n):
        Image.fromarray(np.full((680, 1200, 3), 10 * i, np.uint8)).save(d / f"frame{i:06d}.jpg")
        Image.fromarray(np.full((680, 1200), 6553 * (i + 1), np.uint16)).save(d / f"depth{i:06d}.png")
    return T


def test_replica_dev_loader_synthetic(tmp_path):
    T = write_nice(tmp_path, "apartment_9")
    st = v2.replica_dev_stream("apartment_9", tmp_path)
    assert st.n == 2 and st.intr == proof.REPLICA_INTR and st.poses_file == tmp_path / "apartment_9" / "traj.txt"
    assert np.allclose(st.poses[1], T) and np.allclose(st.depth(1), 2 * 6553 / 6553.5)
    assert st.rgb(1).shape == (680, 1200, 3) and st.rgb(1).dtype == np.uint8
    (tmp_path / "cam_params.json").write_text(json.dumps(
        {"camera": {"w": 1200, "h": 680, "fx": 610.0, "fy": 600.0, "cx": 599.5, "cy": 339.5, "scale": 1000.0}}))
    st = v2.replica_dev_stream("apartment_9", tmp_path)
    assert st.intr.fx == 610.0 and np.allclose(st.depth(0), 6553 / 1000.0)
    (tmp_path / "apartment_9" / "cam_params.json").write_text(json.dumps(          # per-scene file wins
        {"camera": {"w": 1200, "h": 680, "fx": 620.0, "fy": 600.0, "cx": 599.5, "cy": 339.5, "scale": 6553.5}}))
    st = v2.replica_dev_stream("apartment_9", tmp_path)
    assert st.intr.fx == 620.0 and np.allclose(st.depth(1), 2 * 6553 / 6553.5)
    assert v2.VOCAB["replica_dev"] == "replica"
