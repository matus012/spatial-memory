"""ours v1 detection cache: miss -> save -> hit replays byte-identical association inputs; key covers upstream only."""
import collections

import numpy as np
import yaml

from spatialmem.gt.replica import ROOT
from spatialmem.ours import run
from spatialmem.render.raycast import Intrinsics

CFG = yaml.safe_load((ROOT / "configs" / "ours_v1.yaml").read_text())
CFG.update(stride=2, dbscan_min_points=5, merge_every=2)
SHA = {"sam2": "a", "siglip": "b"}


def stream(tmp_path) -> run.Stream:
    """Flat floor 2 m in front of the camera, camera moving 1 cm per frame along x; 6 frames."""
    P = np.stack([np.eye(4)] * 6)
    P[:, 0, 3] = np.arange(6) * 0.01
    f = tmp_path / "poses.txt"
    np.savetxt(f, P.reshape(6, 16))
    return run.Stream(6, Intrinsics(W=40, H=30, fx=20.0, fy=20.0, cx=19.5, cy=14.5),
                      lambda k: np.full((30, 40, 3), k, np.uint8), lambda k: np.full((30, 40), 2.0), P, f)


def masker(rgb):
    a, b = np.zeros((30, 40), bool), np.zeros((30, 40), bool)
    a[2:14, 2:18] = True
    b[16:28, 22:38] = True
    return [a, b]


class FakeSiglip:
    def masks(self, rgb, masks, pad):
        f = np.stack([np.eye(4)[i % 4] for i in range(len(masks))]).astype(np.float32)
        return {"blend": f, "crop": f[:, ::-1].copy()}


def run_miss(st, cfg):
    t, rec, rec_m = collections.Counter(), [], []
    objs, stats = run.map_scene(run.detect(st, cfg, masker, FakeSiglip(), t), 3, cfg, t, rec, rec_m)
    return objs, stats, rec, rec_m


def same(a, b) -> bool:
    return len(a) == len(b) and all(x.n_det == y.n_det and np.array_equal(x.points, y.points)
                                    and np.array_equal(x.descriptor(), y.descriptor()) for x, y in zip(a, b))


def test_cache_round_trip(tmp_path):
    st = stream(tmp_path)
    key = run.cache_key(CFG, SHA, st)
    objs, stats, rec, rec_m = run_miss(st, CFG)
    assert stats["detections"] == 6 and len(objs) == 2
    path = tmp_path / "cache" / "hssd" / "s" / "detections.npz"
    text = np.eye(4, dtype=np.float32)
    run.save_cache(path, key, rec, rec_m, text, ["a", "b", "c", "d"])
    c = run.load_cache(path, key)
    assert c is not None and np.array_equal(c["text"], text) and list(c["frame_nmasks"]) == [2, 2, 2]
    for desc in ("blend", "crop"):
        cfg = {**CFG, "descriptor": desc}
        hit, _ = run.map_scene(run.replay(c), 3, cfg, collections.Counter())
        assert same(hit, run_miss(st, cfg)[0])
    assert np.allclose(c["aabb"][0], [c["points"][: c["pt_off"][1]].min(0), c["points"][: c["pt_off"][1]].max(0)])


def test_cache_key_upstream_only(tmp_path):
    st = stream(tmp_path)
    key = run.cache_key(CFG, SHA, st)
    for k, v in dict(descriptor="crop", lam=1.0, tau=0.9, tau_c=0.5, merge_use_cos=False, min_detections=3).items():
        assert run.cache_key({**CFG, k: v}, SHA, st) == key, k
    for k, v in dict(stride=5, voxel=0.02, dbscan_eps=0.2, crop_pad_px=10, min_mask_frac=0.01, seed=1).items():
        assert run.cache_key({**CFG, k: v}, SHA, st) != key, k
    assert run.cache_key(CFG, {**SHA, "sam2": "z"}, st) != key


def test_stale_cache_not_reused(tmp_path):
    st = stream(tmp_path)
    _, _, rec, rec_m = run_miss(st, CFG)
    path = tmp_path / "detections.npz"
    run.save_cache(path, run.cache_key(CFG, SHA, st), rec, rec_m, np.eye(4, dtype=np.float32), list("abcd"))
    assert run.load_cache(path, run.cache_key({**CFG, "voxel": 0.02}, SHA, st)) is None
    np.savetxt(st.poses_file, (st.poses + 0.001).reshape(6, 16))       # stream changed on disk
    assert run.load_cache(path, run.cache_key(CFG, SHA, st)) is None
