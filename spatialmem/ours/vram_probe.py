"""VRAM probe (task 006 G2): SAM 2.1-s + one v2 encoder resident together, one real frame through both.

  uv run python -m spatialmem.ours.vram_probe ENCODER IMAGE [IMAGE ...] [--empty-cache]

Prints one JSON line per image: torch max_memory_reserved / allocated (GiB) and the nvidia-smi rise over the idle
reading taken before any model is loaded. Run one encoder per process so the rise is that encoder's alone.
--empty-cache: torch.cuda.empty_cache() after the text pass and between SAM and the encoder (returns SAM's cached blocks to the driver).
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch
import yaml
from PIL import Image

from spatialmem.baselines.ovo import vocabulary
from spatialmem.gt.replica import ROOT
from spatialmem.ours import run
from spatialmem.ours.encoders import load
from spatialmem.ours.models import Masker
from spatialmem.ours.v2 import smi_now


def main() -> None:
    empty = "--empty-cache" in sys.argv
    enc_name, images = sys.argv[1], [a for a in sys.argv[2:] if a != "--empty-cache"]
    cfg = yaml.safe_load((ROOT / "configs" / "ours_v2.yaml").read_text())
    assert torch.cuda.is_available(), "CPU fallback: STOP"
    idle = smi_now()
    poll = run.SmiPoll(0.2)
    poll.start()
    masker = Masker(cfg["sam_cfg"], str(ROOT / cfg["sam_ckpt"]), cfg["points_per_batch"], cfg["min_mask_frac"])
    enc = load(enc_name)
    dev = {"sam2": str(next(masker.model.parameters()).device), enc_name: str(next(enc.model.parameters()).device)}
    assert all(v.startswith("cuda") for v in dev.values()), dev
    text = enc.text(vocabulary("hssd"))                       # the larger vocabulary (236 prompts)
    after_text = torch.cuda.memory_reserved() / 2**30
    if empty:
        torch.cuda.empty_cache()
    for img in images:
        torch.cuda.reset_peak_memory_stats()
        at_reset = torch.cuda.memory_reserved() / 2**30
        rgb = np.asarray(Image.open(img).convert("RGB"))
        t0 = time.perf_counter()
        masks = masker(rgb)
        if empty:
            torch.cuda.empty_cache()
        after_sam = torch.cuda.memory_reserved() / 2**30
        t1 = time.perf_counter()
        f = enc.masks(rgb, masks, cfg["crop_pad_px"])          # every mask >= min_frac in one batch (upper bound)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        time.sleep(0.5)
        print(json.dumps({"encoder": enc_name, "image": img, "hw": rgb.shape[:2], "masks": len(masks), "empty_cache": empty,
                          "dim": int(f["crop"].shape[1]), "text": list(text.shape), "devices": dev,
                          "reserved_after_text_gib": round(after_text, 3), "reserved_at_reset_gib": round(at_reset, 3),
                          "reserved_after_sam_gib": round(after_sam, 3), "max_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
                          "max_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                          "smi_idle_mib": idle, "smi_peak_mib": poll.peak, "smi_rise_mib": poll.peak - idle,
                          "sam_s": round(t1 - t0, 2), "encoder_s": round(t2 - t1, 2)}), flush=True)
    poll.stop.set()


if __name__ == "__main__":
    main()
