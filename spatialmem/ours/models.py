"""GPU half of "ours" v1: SAM 2.1 hiera-small masks + SigLIP 2 base image / text embeddings."""
from __future__ import annotations

import numpy as np
import torch

from spatialmem.baselines.ovo import prompt


class Masker:
    """SAM2AutomaticMaskGenerator, class-agnostic, fp16 autocast; masks < min_frac of the image dropped."""

    def __init__(self, cfg_name: str, ckpt: str, points_per_batch: int, min_frac: float):
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2
        self.model = build_sam2(cfg_name, ckpt, device="cuda")
        self.gen = SAM2AutomaticMaskGenerator(self.model, points_per_batch=points_per_batch)
        self.min_frac = min_frac

    def __call__(self, rgb: np.ndarray) -> list[np.ndarray]:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            out = self.gen.generate(rgb)
        min_area = self.min_frac * rgb.shape[0] * rgb.shape[1]
        return [m["segmentation"] for m in out if m["area"] >= min_area]


def padded_box(mask: np.ndarray, pad: int) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    H, W = mask.shape
    return max(ys.min() - pad, 0), min(ys.max() + pad + 1, H), max(xs.min() - pad, 0), min(xs.max() + pad + 1, W)


class Siglip:
    """google/siglip2-base-patch16-224 (fixed-resolution SigLIP 2; transformers loads it as SiglipModel), fp16."""

    def __init__(self, path: str):
        from transformers import AutoModel, AutoProcessor
        self.model = AutoModel.from_pretrained(path, dtype=torch.float16).cuda().eval()
        self.proc = AutoProcessor.from_pretrained(path)

    @torch.inference_mode()
    def _img(self, crops: list[np.ndarray]) -> torch.Tensor:
        x = self.proc(images=crops, return_tensors="pt")["pixel_values"].to("cuda", torch.float16)
        f = self.model.get_image_features(pixel_values=x).float()
        return f / f.norm(dim=-1, keepdim=True)

    @torch.inference_mode()
    def masks(self, rgb: np.ndarray, masks: list[np.ndarray], pad: int) -> dict[str, np.ndarray]:
        """Both descriptors per mask, L2-normalised. "blend" (Design v1): w * f(masked crop, black fill) + (1 - w) *
        f(crop), w = mask fraction of the padded box. "crop": f(crop) only."""
        crops, masked, w = [], [], []
        for m in masks:
            y0, y1, x0, x1 = padded_box(m, pad)
            c, mm = rgb[y0:y1, x0:x1], m[y0:y1, x0:x1]
            crops.append(c)
            masked.append(c * mm[..., None])
            w.append(mm.mean())
        wt = torch.tensor(w, device="cuda")[:, None]
        fm, fc = self._img(masked), self._img(crops)
        f = wt * fm + (1 - wt) * fc
        return {"blend": (f / f.norm(dim=-1, keepdim=True)).cpu().numpy(), "crop": fc.cpu().numpy()}

    @torch.inference_mode()
    def text(self, classes: list[str]) -> np.ndarray:
        """CONTRACT 2026-10-27e (2) prompts; SigLIP text processing as on the model card (max_length padding, 64)."""
        t = self.proc(text=[prompt(c) for c in classes], padding="max_length", max_length=64, return_tensors="pt").to("cuda")
        f = self.model.get_text_features(**t).float()
        return (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()
