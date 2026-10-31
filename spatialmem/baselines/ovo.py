"""OVO-mapping baseline (tberriel/OVO @ e0d16f7, `slam_module: vanilla`, GT poses) -> CONTRACT `Prediction`.

Two halves, split on the torch boundary:
- inside the container on PERUN (torch, open_clip, OVO importable; cwd /opt/OVO):
    probe                      G1: CUDA + SigLIP + SAM2 on device
    convert CKPT NPZ           ovo_map.ckpt -> npz (xyz, obj_ids, ids, feats)
    textemb VOCAB NPZ          SigLIP text embeddings of "a photo of a {c}." (VOCAB = replica | hssd)
- on lpc (numpy only): `predict` — npz + text-embedding npz -> data/baselines/ovo/<ds>/<scene>.json (+ .sem.npz)

Label (docs/CONTRACT.md 2026-10-27e): argmax cosine of the instance's own `ins3d_{id}_clip_feature` vs the text
embeddings over the vocabulary (Replica 51 protocol classes; HSSD 236 canonical classes).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from spatialmem.eval.boxes import from_points
from spatialmem.eval.contract import STRUCTURAL, PredObject, Prediction
from spatialmem.gt.relations import compute_relations

TEMPLATE = "a photo of a {}."
MIN_POINTS = 10           # instances with fewer map points are dropped (reported)
UP = {"replica": 2, "hssd": 1}
SIGLIP_CARD = "SigLIP-384"   # OVO model_card (hf-hub:timm/ViT-SO400M-14-SigLIP-384)


def prompt(c: str) -> str:
    """CONTRACT 2026-10-27e (2): "a photo of a {c}." with _ and - in c replaced by spaces."""
    return TEMPLATE.format(c.replace("_", " ").replace("-", " "))


def vocabulary(vocab: str) -> list[str]:
    if vocab == "replica":
        from spatialmem.gt.replica import protocol
        return list(protocol().names)
    from spatialmem.hssd.vocab import vocabulary as hssd_vocab
    return hssd_vocab()


# ------------------------------------------------------------------ lpc half (numpy only)
def predict(scene: str, dataset: str, npz: Path, textemb: Path) -> tuple[Prediction, dict]:
    m, t = np.load(npz), np.load(textemb, allow_pickle=False)
    classes = [str(c) for c in t["classes"]]
    assert classes == vocabulary(dataset), "text-embedding class list != vocabulary"
    xyz, oid, ids, feats = m["xyz"].astype(np.float64), m["obj_ids"], m["ids"], m["feats"].astype(np.float64)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True).clip(1e-12)
    label_idx = (feats @ t["emb"].astype(np.float64).T).argmax(1)           # cosine argmax
    objects: list[PredObject] = []
    inst_label = np.full(len(xyz), -1, dtype=np.int64)
    dropped = 0
    for k, i in enumerate(ids):
        sel = oid == i
        if sel.sum() < MIN_POINTS:
            dropped += 1
            continue
        box = from_points(xyz[sel])
        objects.append(PredObject(id=int(i), label=classes[label_idx[k]], aabb_min=box[0], aabb_max=box[1]))
        inst_label[sel] = label_idx[k]
    rels = compute_relations([o for o in objects if o.label not in STRUCTURAL], up=UP[dataset])
    pred = Prediction(scene=scene, objects=objects, relations=rels)
    if dataset == "replica":
        pred.sem_points, pred.sem_labels = xyz.astype(np.float32), inst_label
    return pred, {"scene": scene, "instances": int(len(ids)), "kept": len(objects), "dropped_lt_10_pts": dropped,
                  "points": int(len(xyz)), "points_without_kept_instance": int((inst_label < 0).sum()),
                  "relations": len(rels)}


def cmd_predict(a: argparse.Namespace) -> None:
    out = Path(a.out) / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    stats = []
    for f in sorted(Path(a.npz_dir).glob("*.npz")):
        pred, st = predict(f.stem, a.dataset, f, Path(a.textemb))
        pred.to_json(out / f"{f.stem}.json")
        stats.append(st)
        print(st)
    (out / "_stats.json").write_text(json.dumps(stats, indent=1))


# ------------------------------------------------------------------ container half (torch)
def _np(x):
    import torch
    return torch.as_tensor(x).detach().cpu().numpy() if x is not None else None


def cmd_probe(_: argparse.Namespace) -> None:
    import torch
    print("cuda_available", torch.cuda.is_available(), "device", torch.cuda.get_device_name(0), "torch", torch.__version__)
    assert torch.cuda.is_available()
    from ovo.utils import clip_utils, io_utils, segment_utils
    model, tok, _, dim = clip_utils.load_clip_model(SIGLIP_CARD, False)
    model = model.cuda()
    print("siglip_param_device", next(model.parameters()).device, "dim", dim)
    cfg = io_utils.load_config("data/working/configs/ovo.yaml")["semantic"]["sam"]
    mg = segment_utils.load_sam(cfg, device="cuda")
    print("sam2_param_device", next(mg.predictor.model.parameters()).device)
    with torch.no_grad():
        e = model.encode_text(tok(["a photo of a chair."]).cuda())
    print("siglip_text_embed", tuple(e.shape), e.device, "max_mem_alloc_MiB", torch.cuda.max_memory_allocated() // 2**20)


def cmd_convert(a: argparse.Namespace) -> None:
    import torch
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    mp, op = ck["map_params"], ck["ovo_map_params"]
    ids = _np(op["ins_3d_ids"]).astype(np.int64)
    feats = np.stack([_np(op[f"ins3d_{i}_clip_feature"]).reshape(-1) for i in ids]) if len(ids) else np.zeros((0, 1152))
    np.savez_compressed(a.npz, xyz=_np(mp["xyz"]).astype(np.float32), obj_ids=_np(mp["obj_ids"]).reshape(-1).astype(np.int64),
                        ids=ids, feats=feats.astype(np.float32))
    print("converted", a.ckpt, "points", len(_np(mp["xyz"])), "instances", len(ids))


def cmd_textemb(a: argparse.Namespace) -> None:
    import torch
    from ovo.utils import clip_utils
    model, tok, _, _ = clip_utils.load_clip_model(SIGLIP_CARD, False)
    model = model.cuda()
    classes = vocabulary(a.vocab)
    with torch.no_grad():
        e = model.encode_text(torch.cat([tok(prompt(c)) for c in classes]).cuda()).float()
        e = e / e.norm(dim=-1, keepdim=True)
    np.savez(a.npz, classes=np.array(classes), prompts=np.array([prompt(c) for c in classes]), emb=e.cpu().numpy(), template=np.array(TEMPLATE))
    print("textemb", a.vocab, e.shape)


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(required=True)
    sp.add_parser("probe").set_defaults(f=cmd_probe)
    c = sp.add_parser("convert"); c.add_argument("ckpt"); c.add_argument("npz"); c.set_defaults(f=cmd_convert)
    t = sp.add_parser("textemb"); t.add_argument("vocab", choices=["replica", "hssd"]); t.add_argument("npz"); t.set_defaults(f=cmd_textemb)
    p = sp.add_parser("predict")
    p.add_argument("--dataset", required=True, choices=["replica", "hssd"])
    p.add_argument("--npz-dir", required=True); p.add_argument("--textemb", required=True)
    p.add_argument("--out", default="data/baselines/ovo"); p.set_defaults(f=cmd_predict)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
