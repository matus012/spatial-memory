"""HSSD (hssd/hssd-hab, CC BY-NC 4.0, public on HF) — metadata fetch, seeded scene selection, per-scene objects.

Runs on the PERUN login node (anonymous HF access, `gated: false`), data under $PROJ/data/hssd-hab.
  python scripts/hssd_fetch.py meta    --root <dir>               # configs, scenes/, stages/*.json, semantics, metadata
  python scripts/hssd_fetch.py select  --root <dir> --out configs/hssd_scenes.yaml
  python scripts/hssd_fetch.py objects --root <dir> --scenes configs/hssd_scenes.yaml   # stages + object assets of those scenes
  python scripts/hssd_fetch.py manifest --root <dir> --scenes configs/hssd_scenes.yaml > hssd_manifest.txt

Selection (task 002 step 2, fixed before any number): pool = scenes whose scene_instance has >= MIN_OBJECTS
object_instances, sorted by id; order = Random(SEED).sample(pool, all); dev = first 5; eval = next 30 not sharing a dev floorplan id.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import yaml
from huggingface_hub import HfApi, snapshot_download

REPO = "hssd/hssd-hab"
REVISION = "4369cb9876214c7fbebcf552eb532380e4d287e4"   # HF sha at selection time (2026-10-27)
SEED, MIN_OBJECTS, N_DEV, N_EVAL = 114, 30, 5, 30


def fetch(root: Path, patterns: list[str]) -> None:
    snapshot_download(REPO, repo_type="dataset", revision=REVISION, local_dir=root, allow_patterns=patterns,
                      max_workers=16)


def meta(root: Path) -> None:
    fetch(root, ["*.scene_dataset_config.json", "scene_splits.yaml", "README.md", "scenes/*.scene_instance.json",
                 "stages/*.stage_config.json", "semantics/*", "semantics/scenes/*", "metadata/*"])


def scene_objects(root: Path, scene: str) -> list[dict]:
    return json.loads((root / "scenes" / f"{scene}.scene_instance.json").read_text())["object_instances"]


def floorplan(scene: str) -> str:
    return scene.split("_")[0]


def select(root: Path, out: Path) -> None:
    scenes = sorted(p.name.split(".")[0] for p in (root / "scenes").glob("*.scene_instance.json"))
    counts = {s: len(scene_objects(root, s)) for s in scenes}
    pool = [s for s in scenes if counts[s] >= MIN_OBJECTS]
    order = random.Random(SEED).sample(pool, len(pool))
    dev = order[:N_DEV]
    # dev / eval disjoint by floorplan (the id before "_"): two layouts of one house must not straddle the split
    dev_fp = {floorplan(s) for s in dev}
    ev = [s for s in order[N_DEV:] if floorplan(s) not in dev_fp][:N_EVAL]
    dev, ev = sorted(dev), sorted(ev)
    assert len(ev) == N_EVAL and not {floorplan(s) for s in dev} & {floorplan(s) for s in ev}
    doc = {"source": f"{REPO}@{REVISION}", "rule": f"seed {SEED}; pool = scenes with >= {MIN_OBJECTS} object_instances "
           f"({len(pool)} of {len(scenes)}), sorted; order = random.Random({SEED}).sample(pool, len(pool)); dev = first "
           f"{N_DEV}; eval = next {N_EVAL} whose floorplan id (before '_') is not a dev floorplan", "dev": dev, "eval": ev,
           "object_instances": {s: counts[s] for s in dev + ev}}
    out.write_text(yaml.safe_dump(doc, sort_keys=False))
    print(f"pool {len(pool)}/{len(scenes)}; dev {dev}; eval {len(ev)} -> {out}")


def scene_list(path: Path) -> list[str]:
    d = yaml.safe_load(path.read_text())
    return d["dev"] + d["eval"]


def asset_patterns(root: Path, scenes: list[str]) -> list[str]:
    """Stage GLB + every object template the scenes reference (objects/<c>/<hash>.* and decomposed parts)."""
    api = HfApi()
    files = {f for f in api.list_repo_files(REPO, repo_type="dataset", revision=REVISION) if f.startswith("objects/")}
    by_stem: dict[str, list[str]] = {}
    for f in files:
        stem = Path(f).name.split(".")[0]   # <hash>, <hash>_part_<k> (objects/decomposed/), <n>-<k> (objects/openings/)
        by_stem.setdefault(stem, []).append(f)
    pats, missing = [], set()
    for s in scenes:
        d = json.loads((root / "scenes" / f"{s}.scene_instance.json").read_text())
        stage = d["stage_instance"]["template_name"].split("/")[-1]
        pats += [f"stages/{stage}.glb", f"stages/{stage}.stage_config.json"]
        for o in d["object_instances"]:
            t = o["template_name"].split("/")[-1]
            if t in by_stem:
                pats += by_stem[t]
            else:
                missing.add(t)
    if missing:
        print(f"WARNING {len(missing)} templates not found under objects/: {sorted(missing)[:10]}")
    return sorted(set(pats))


def objects(root: Path, scenes_yaml: Path) -> None:
    pats = asset_patterns(root, scene_list(scenes_yaml))
    print(f"{len(pats)} asset files")
    fetch(root, pats)


def manifest(root: Path, scenes_yaml: Path) -> None:
    pats = ["scenes/" + f"{s}.scene_instance.json" for s in scene_list(scenes_yaml)] + asset_patterns(root, scene_list(scenes_yaml))
    tot = 0
    for p in pats:
        f = root / p
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        tot += f.stat().st_size
        print(f"{h}  {p}")
    print(f"# {len(pats)} files, {tot} bytes, {REPO}@{REVISION}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["meta", "select", "objects", "manifest"])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("configs/hssd_scenes.yaml"))
    ap.add_argument("--scenes", type=Path, default=Path("configs/hssd_scenes.yaml"))
    a = ap.parse_args()
    {"meta": lambda: meta(a.root), "select": lambda: select(a.root, a.out),
     "objects": lambda: objects(a.root, a.scenes), "manifest": lambda: manifest(a.root, a.scenes)}[a.cmd]()


if __name__ == "__main__":
    main()
