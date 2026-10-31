"""Task 002 G4 negative controls on a pulled dev stream (lpc, no Habitat).

The same StreamCheck on every 5th frame with (a) the stored OpenCV c2w poses, (b) OpenGL camera axes (c2w @ diag(1,-1,-1,1)),
(c) a 5 cm x shift of the camera. (a) must pass G4, (b) and (c) must fail it.
  uv run python scripts/hssd_g4_negative_control.py --root data/hssd_runs/dev --scene 102816627
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from spatialmem.hssd.check import StreamCheck


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/hssd_runs/dev"))
    ap.add_argument("--scene", default="102816627")
    ap.add_argument("--every", type=int, default=5)
    a = ap.parse_args()
    D = a.root / a.scene
    g = json.loads((D / "gt.json").read_text())
    m = json.loads((D / "meta.json").read_text())
    P = np.loadtxt(D / "poses.txt").reshape(-1, 4, 4)
    shift = np.zeros((4, 4))
    shift[0, 3] = 0.05
    variants = (("as stored (OpenCV c2w)", lambda T: T), ("negative control: OpenGL axes", lambda T: T @ np.diag([1.0, -1, -1, 1])),
                ("negative control: 5 cm x shift", lambda T: T + shift))
    for name, f in variants:
        c = StreamCheck(g["gt_groups"], m["intrinsics"])
        for i in range(0, len(P), a.every):
            d = np.asarray(Image.open(D / "depth" / f"{i:06d}.png"), float) / m["depth_scale"]
            k = np.asarray(Image.open(D / "inst" / f"{i:06d}.png")).astype(np.int64)
            c.add(d, k, f(P[i]))
        r = c.result()
        fr = [v["inside"] for v in r["per_object"].values() if v["n_pts"] >= 500]
        print(f"{a.scene} every {a.every}th frame ({r['frames']}): {name}: G4 objects {r['G4_objects']} median "
              f"{r['G4_median_inside']:.3f} ge0.90 {r['G4_frac_objects_ge_0.90']:.3f} p05 {np.percentile(fr, 5):.3f}")


if __name__ == "__main__":
    main()
