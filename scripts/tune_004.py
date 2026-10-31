"""Task 004 tuning driver: association-only runs from the detection cache on HSSD dev 5, scored by
score_baselines. Objective = mean over the 5 dev scenes of object F1@0.25 (task 004 step 3)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEV = ["102816627", "104348253_171513237", "104348328_171513363", "105515211_173104185", "108294558_176710095"]
EVID = ROOT / "data" / "tune_004"   # outputs (git-ignored)


def run(name: str, sets: list[str]) -> dict:
    tag = f"ours_dev_{name}"
    out = EVID / name
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "spatialmem.ours.run", "--config", "configs/ours_v1.yaml", "--dataset", "hssd",
           "--scenes", *DEV, "--tag", tag, "--set", *sets]
    with open(out / "run.log", "w") as f:
        subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
    with open(out / "score.log", "w") as f:
        subprocess.run([sys.executable, "-m", "spatialmem.eval.score_baselines", "--methods", tag, "--out", str(out)],
                       cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
    sc = json.load(open(out / f"scores_{tag}.json"))["hssd_dev"]
    per = {s: sc["scenes"][s] for s in DEV}
    n = len(DEV)
    labels = [o["label"] for s in DEV for o in json.load(open(ROOT / "data/baselines" / tag / "hssd" / f"{s}.json"))["objects"]]
    row = {
        "config": name, "sets": sets,
        "mean_f1_25": sum(p["obj@0.25"]["f1"] for p in per.values()) / n,
        "mean_f1_50": sum(p["obj@0.5"]["f1"] for p in per.values()) / n,
        "mean_label_acc": sum(p["obj@0.25"]["label_acc"] for p in per.values()) / n,
        "mean_rel_f1": sum(p["rel"]["f1"] for p in per.values()) / n,
        "per_scene_f1_25": {s: per[s]["obj@0.25"]["f1"] for s in DEV},
        "n_pred": sum(p["obj@0.25"]["n_pred"] for p in per.values()),
        "object__outside_share": sum(l == "object__outside" for l in labels) / max(len(labels), 1),
    }
    json.dump(row, open(out / "row.json", "w"), indent=1)
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("sets", nargs="*")
    r = run(ap.parse_args().name, ap.parse_args().sets)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "per_scene_f1_25"}))
