"""v2 label/descriptor encoders (task 006): image features for mask crops + text features for the CONTRACT 2026-10-27e
prompts, both from the same model, L2-normalised, fp16 on cuda. Each exposes `masks(rgb, masks, pad)` -> {"blend",
"crop"} (descriptor rule of `models.Siglip`) and `text(classes)`.

Weights live under data/models/ (git-ignored); checksums in THIRD_PARTY.md and run_meta.json.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from spatialmem.baselines.ovo import prompt
from spatialmem.gt.replica import ROOT
from spatialmem.ours.models import Siglip

# name -> (family, weights dir, weights file)
ENCODERS: dict[str, tuple[str, str, str]] = {
    "siglip2_base": ("siglip", "data/models/siglip2-base-patch16-224", "model.safetensors"),
    "siglip2_so400m": ("siglip", "data/models/siglip2-so400m-patch14-384", "model.safetensors"),
    "clip_vith14_laion2b": ("open_clip", "data/models/CLIP-ViT-H-14-laion2B-s32B-b79K", "open_clip_model.safetensors"),
}


def weights(name: str) -> Path:
    _, d, f = ENCODERS[name]
    return ROOT / d / f


class OpenClip(Siglip):
    """open_clip ViT-H-14 / laion2b_s32b_b79k (the ConceptGraphs encoder), fp16; its own preprocess (resize shortest
    side 224 + centre crop) and tokenizer (context 77). Descriptor rule inherited from `Siglip.masks`."""

    def __init__(self, ckpt: str):
        import open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained=ckpt, precision="fp16", device="cuda")
        self.model.eval()
        self.tok = open_clip.get_tokenizer("ViT-H-14")

    @torch.inference_mode()
    def _img(self, crops: list[np.ndarray]) -> torch.Tensor:
        x = torch.stack([self.preprocess(Image.fromarray(c)) for c in crops]).to("cuda", torch.float16)
        f = self.model.encode_image(x).float()
        return f / f.norm(dim=-1, keepdim=True)

    @torch.inference_mode()
    def text(self, classes: list[str]) -> np.ndarray:
        f = self.model.encode_text(self.tok([prompt(c) for c in classes]).to("cuda")).float()
        return (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()


def load(name: str) -> Siglip:
    fam, d, f = ENCODERS[name]
    return Siglip(str(ROOT / d)) if fam == "siglip" else OpenClip(str(ROOT / d / f))
