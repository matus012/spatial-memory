"""Generate RESULTS.md from the saved score files in results/scores/.

Run: uv run python -m spatialmem.report.results

Deterministic: no timestamps, fixed float formats, fixed row order. Reads only results/scores/*,
results/published.yaml. Score files are merged per method (JSON "method") and per split: the first
file (sorted by name) with a non-empty split wins, so a later run on a new split is dropped in as an
extra file (e.g. scores_ours_v2_hssd_eval.json) without touching the others.
"""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCORES = ROOT / "results" / "scores"
PUBLISHED = ROOT / "results" / "published.yaml"
OUT = ROOT / "RESULTS.md"

DASH = "—"
Scores = dict[str, dict[str, dict]]  # method -> split -> split dict


def load_scores(d: Path = SCORES) -> Scores:
    out: Scores = {}
    for f in sorted(d.glob("scores*.json")):
        j = json.loads(f.read_text(encoding="utf-8"))
        m = out.setdefault(j["method"], {})
        for split, v in j.items():
            if isinstance(v, dict) and v.get("scenes") and split not in m:
                m[split] = v
    return out


def load_published() -> dict:
    return yaml.safe_load(PUBLISHED.read_text(encoding="utf-8"))


def _read_csv(name: str) -> list[dict[str, str]]:
    with (SCORES / name).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def f3(x: float) -> str:
    return f"{x:.3f}"


def pct(x: float) -> str:
    return f"{100 * x:.2f}"


def table(head: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------- object tables

HSSD_HEAD = ["method", "P@.25", "R@.25", "F1@.25", "P@.5", "R@.5", "F1@.5", "label acc@.25",
             "rel P", "rel R", "rel F1", "mean-scene F1@.25"]


def object_row(name: str, s: dict | None, missing: str) -> list[str]:
    if s is None:
        return [name] + [missing] * (len(HSSD_HEAD) - 1)
    p = s["pooled"]
    a, b, r = p["obj@0.25"], p["obj@0.5"], p["rel"]
    return [name, f3(a["p"]), f3(a["r"]), f3(a["f1"]), f3(b["p"]), f3(b["r"]), f3(b["f1"]),
            f3(a["label_acc"]), f3(r["p"]), f3(r["r"]), f3(r["f1"]), f3(p["mean_scene_f1@0.25"])]


# (label, method key, text shown when the split has no data for it)
METHOD_ROWS = [
    ("ConceptGraphs (map module)", "cg", DASH),
    ("OVO", "ovo", DASH),
    ("ours v1 (attempt)", "ours_v1", DASH),
    ("ours v2 (attempt, K2 fired)", "ours_v2", "pending"),
]


def object_table(sc: Scores, split: str, v1_missing: str = DASH) -> str:
    rows = []
    for label, key, miss in METHOD_ROWS:
        s = sc.get(key, {}).get(split)
        rows.append(object_row(label, s, miss if key == "ours_v2" else v1_missing if key == "ours_v1" else miss))
    return table(HSSD_HEAD, rows)


def n_scenes(sc: Scores, split: str) -> int:
    return len(sc["cg"][split]["scenes"])


# ---------------------------------------------------------------- semantic table

def sem_row(name: str, s: dict, note: str = "") -> list[str]:
    p = s["sem_pooled"]
    per = st.mean(v["sem"]["miou"] for v in s["scenes"].values())
    return [name, pct(p["miou"]), pct(p["macc"]), pct(p["fmiou"]), pct(per), note]


def published_val(x: float | None, pm: float | None = None) -> str:
    if x is None:
        return DASH
    return f"{x:.1f}" + (f" ± {pm:.1f}" if pm is not None else "")


def semantic_table(sc: Scores, pub: dict) -> str:
    head = ["method", "mIoU", "mAcc", "f-mIoU", "per-scene mean mIoU", "note"]
    rows: list[list[str]] = []
    for r in pub["published"]:
        rows.append([f"{r['method']} (published)", published_val(r["miou"]),
                     published_val(r.get("macc")), published_val(r.get("fmiou")), DASH, "OVO Table II"])
    o = pub["own_eval"]["ovo"]
    rows.append(["OVO, own eval (our re-run)", f"{o['miou']:.2f}", f"{o['macc']:.2f}", f"{o['fmiou']:.2f}", DASH,
                 "OVO's `run_eval.py`; f-mIoU at the log's 0.1 precision"])
    rows.append(sem_row("ConceptGraphs (our re-run)", sc["cg"]["replica"], "world-frame pose patch (note 1)"))
    rows.append(sem_row("OVO (our re-run, our label rule)", sc["ovo"]["replica"], "CONTRACT 2026-10-27e labels"))
    rows.append(sem_row("ours v1 (attempt, lpc)", sc["ours_v1"]["replica"], "004, frozen c11"))
    rows.append(sem_row("ours v1, PERUN caches (reference)", sc["ours_v1ref_perun"]["replica"], "same caches as v2"))
    rows.append(sem_row("ours v2 (attempt, K2 fired)", sc["ours_v2"]["replica"], "006, frozen 8eed799"))
    k = sc["cg"]["replica"]["sem_k5_ceiling_pooled"]
    rows.append(["*k=5 ceiling (GT labels through the transfer)*", pct(k["miou"]), pct(k["macc"]), pct(k["fmiou"]),
                 DASH, "upper bound of the transfer step, not a method"])
    return table(head, rows)


# ---------------------------------------------------------------- runtime

def _fmt_s(x: float) -> str:
    return f"{x:.2f}" if x >= 1 else f"{x:.3f}"


def _summ(sec: list[float], vram: list[float]) -> tuple[str, str]:
    s = f"{_fmt_s(st.median(sec))} [{_fmt_s(min(sec))}–{_fmt_s(max(sec))}]"
    v = f"{st.median(vram):.1f} ({max(vram):.1f})" if vram else DASH
    return s, v


def baseline_runtime(sc: Scores) -> str:
    sets = {"Replica 8": None, "HSSD eval 30": set(sc["cg"]["hssd_eval"]["scenes"]),
            "HSSD dev 5": set(sc["cg"]["hssd_dev"]["scenes"])}
    rows = []
    for name, fname, scol in (("OVO", "runtime_ovo.csv", "s_per_frame"), ("CG", "runtime_cg.csv", "s_per_frame")):
        data = _read_csv(fname)
        for sname, scenes in sets.items():
            sel = [r for r in data if (r["dataset"].lower() == "replica") == (scenes is None)
                   and (scenes is None or r["scene"] in scenes)]
            sec = [float(r[scol]) for r in sel]
            vram = [float(r["peak_vram_gb"]) for r in sel if r["peak_vram_gb"] not in ("", "NA")]
            s, v = _summ(sec, vram)
            rows.append([name, f"{sname} (n={len(sel)})", s, v])
    return table(["method", "set", "s/frame median [range]", "peak VRAM GB median (max)"], rows)


def ours_runtime() -> str:
    data = _read_csv("ours_v2_r8_build_runtime.csv")
    groups = [("A: SAM 2.1-s masks + lift + siglip2-base (v1 detections)", "A", "siglip2_base"),
              ("B: CLIP-H descriptors from cached masks (v2 labels)", "B", "clip_vith14_laion2b"),
              ("B: siglip2-base descriptors", "B", "siglip2_base")]
    rows = []
    per: dict[str, dict[str, float]] = {}
    for label, ph, enc in groups:
        sel = [r for r in data if r["phase"] == ph and r["encoder"] == enc]
        sec = [float(r["s_per_frame"]) for r in sel]
        per[label[0] + enc] = {r["scene"]: float(r["s_per_frame"]) for r in sel}
        rows.append([label, f"{_fmt_s(st.mean(sec))} [{_fmt_s(min(sec))}–{_fmt_s(max(sec))}]",
                     f"{max(float(r['peak_alloc_gb']) for r in sel):.2f}",
                     f"{max(float(r['peak_reserved_gb']) for r in sel):.2f}"])
    a, b = per["Asiglip2_base"], per["Bclip_vith14_laion2b"]
    e2e = st.mean(a[k] + b[k] for k in sorted(a))
    rows.append(["A + B (CLIP-H), end to end", _fmt_s(e2e), DASH, DASH])
    return table(["phase (Replica 8, PERUN H200)", "s/frame mean [range]", "peak allocated GB (max)",
                  "peak reserved GB (max)"], rows)


# ---------------------------------------------------------------- document

P1B = ROOT / "results" / "p1b"
TYPES = ("existence", "relation", "counting", "find")
REPAIRS = ("M", "label", "extra", "miss", "box", "label+extra", "label+extra+miss", "all")
REAL = ("cg", "ovo", "ours_v2")
AXES = ("drop", "label", "position", "extra", "relations")


def _j(name: str) -> dict:
    return json.loads((P1B / name).read_text(encoding="utf-8"))


def p1(x: float) -> str:
    return f"{100 * x:.1f}"


def _level(axis: str, v: float) -> str:
    return "0 (gt)" if v == 0 else (f"{v:.2f} m" if axis == "position" else f"{v:g} %")


def p1b_section() -> str:
    ceil, summ, runs, feat, over, rep = (_j("ceiling/ceiling_008.json"), _j("summary.json"), _j("runs.json"), _j("features.json"),
                                          _j("overflow.json"), _j("repair.json"))
    rows = [("gt72b v1.2 (gate row)", "gt72b_v12"), ("gt7b v1.2 (gate row)", "gt7b_v12"), ("gt72b v1.1 (pre-fix)", "gt72b_v11"),
            ("gt7b v1.1 (pre-fix)", "gt7b_v11"), ("img7b (7B, 8 frames)", "img7b")]
    a = []
    for lab, k in rows:
        for rule in ("dist", "label+dist"):
            r = ceil[k][rule]
            a.append([lab, rule, p1(r["accuracy"]), *(p1(r["per_type"][t]["accuracy"]) for t in TYPES)])
    r = runs["img"]
    a.append(["img (72B, 8 frames; 009)", "label+dist", p1(r["accuracy"]), *(p1(r["per_type"][t]["accuracy"]) for t in TYPES)])
    cur = summ["curves"]
    b = []
    for ax in AXES:
        c = cur[ax]
        for i, v in enumerate(c["levels"]):
            o = c["overall"]
            b.append([ax, _level(ax, v), f"{p1(o['mean'][i])} [{p1(o['min'][i])}–{p1(o['max'][i])}]", *(p1(c[t]["mean"][i]) for t in TYPES)])
    h, rb = summ["headline"], summ["robustness"]
    co = h["coef"]
    n_over = sum(any(r["removed"] > 0 for r in o.values()) for o in over.values())
    pr = lambda s: [[m, p1(s["real"][m]["predicted"]), p1(s["real"][m]["actual"]), f"{s['real'][m]['abs_error_pp']:.1f}"] for m in REAL]
    ft = [[m, f3(feat[m]["f1"]), f3(feat[m]["label_acc"]), f3(feat[m]["rel_f1"])] for m in ("gt", *REAL)]
    acc = rep["accuracy"]
    d = [[m, *(p1(acc[m if k == "M" else f"{m}__{k}"]["overall"]) for k in REPAIRS)] for m in REAL]
    gains, dom = [], []
    for m in REAL:
        o = rep["maps"][m]["overall"]
        gains.append([m, *(f"{o['gains_pp'][k]:+.1f}" for k in REPAIRS[1:5]), *(f"{o['cumulative_gain_pp'][k]:+.1f}" for k in REPAIRS[5:])])
        dom.append([m, o["dominant"]["reading"], f"{o['dominant']['top']} − {o['dominant']['second']}", f"{o['dominant']['margin_pp']:.1f}",
                    *(rep["maps"][m][t]["dominant"]["reading"] for t in TYPES)])
    g2 = [[m, p1(acc[f"{m}__all"]["overall"]), f"{rep['maps'][m]['g2']['delta_pp']:+.1f}", "ok" if rep["maps"][m]["g2"]["ok"] else "FLAG"] for m in REAL]
    fb = [(v, s) for v, hh in rep["overflow"].items() for s, r in hh.items() if r.get("fallback")]
    parts = [
        "# P1b results\n",
        "Generated from `results/p1b/` (`ceiling/ceiling_008.json`, `summary.json`, `runs.json`, `features.json`, `overflow.json`, "
        "`repair.json`). Planner: Qwen2.5-VL-72B-Instruct (7B where stated), temperature 0, 479 questions on the 30 HSSD eval houses "
        "(images-only: 359 asked, find counted wrong). Percent unless stated; pp = percentage points.\n",
        "### (a) Planner ceiling on the GT map (task 008)\n",
        table(["row", "find rule", "overall", *TYPES], a),
        "\n- Protocol: `docs/P1B_PROTOCOL.md` (v1.2 map text = gate rows; v1.1 rows are the pre-fix diagnosis). Find rule `dist` = within 1.0 m "
        "of the GT object (008); `label+dist` = returned object's label is the asked label as well (deviation 16, used for every 009/010 row).\n"
        "- Deviations 14 (map text v1.1), 15 (v1.2, the one pre-registered prompt fix), 16 (find rule) in `DEVIATIONS.md`.\n",
        "### (b) Sweep: planner accuracy vs degradation (task 009, 72B, find rule label+dist)\n",
        table(["axis", "level", "overall mean [min–max]", *(f"{t} (mean)" for t in TYPES)], b),
        "\n- Levels: drop / label / extra / relations in percent of objects (relations: of relation lines), position = sigma in m; level 0 = the "
        "`gt` re-run; mean and range over 3 seeds. Definitions: `docs/P1B_SWEEP.md` §2.\n",
        "### (c) Does map quality predict planner success? (009)\n",
        f"OLS accuracy ~ intercept + object F1 + label acc + relation F1 on {h['n_fit']} points (60 degraded + `gt`): intercept {co['intercept']:.3f}, "
        f"F1 {co['f1']:.3f}, label acc {co['label_acc']:.3f}, relation F1 {co['rel_f1']:.3f}; R² {h['r2']:.3f}.\n",
        table(["map", "predicted", "actual", "abs error (pp)"], pr(h)),
        f"\n**Pre-registered reading (all 30 houses): {h['reading']}** (holds if all errors ≤ 5.0 pp, fails if any > 10.0 pp).\n",
        f"Robustness, {len(rb['houses'])} houses where no variant triggered the overflow rule (refit on that subset):\n",
        table(["map", "predicted", "actual", "abs error (pp)"], pr(rb)),
        f"\n**Reading (subset): {rb['reading']}**; R² {rb['r2']:.3f}. Images-only row (72B): {p1(runs['img']['accuracy'])} %.\n",
        "Map-quality features of the real maps (P1a scorer, pooled over 30 houses, before the overflow rule):\n",
        table(["map", "F1@.25", "label acc", "rel F1"], ft),
        f"\n- {n_over} of 64 map variants trigger the 30,000-token overflow rule on at least one house (`results/p1b/overflow.json`). "
        "Reading and robustness rule: `docs/P1B_SWEEP.md` §5; deviation 16.\n",
        "### (d) Repair attribution on the real maps (task 010)\n",
        "Planner accuracy (overall) of each real map and of the map after each repair (matching at IoU 0.25; relations recomputed):\n",
        table(["map", *REPAIRS], d),
        "\nGain over the unrepaired map (pp): four single repairs, then the cumulative combinations.\n",
        table(["map", "label", "extra", "miss", "box", "label+extra", "label+extra+miss", "all"], gains),
        "\nDominant error (largest single gain, if ahead of the second by ≥ 5.0 pp), overall and per question type:\n",
        table(["map", "overall", "top − second", "margin (pp)", *TYPES], dom),
        "\nSanity (all repairs ≈ the GT map; must be within ±3.0 pp of 85.4 %):\n",
        table(["map", "acc(all)", "Δ vs 85.4 (pp)", "G2"], g2),
        "\n- Definitions and readings: `docs/P1B_REPAIR.md`. Deviation 17: overflow fallback, used once"
        + (f" (`{fb[0][0]}`, house {fb[0][1]})" if len(fb) == 1 else f" ({len(fb)} cases)") + ".\n",
    ]
    return "\n".join(parts)


def render() -> str:
    sc, pub = load_scores(), load_published()
    n_eval, n_dev = n_scenes(sc, "hssd_eval"), n_scenes(sc, "hssd_dev")
    v2_eval = "hssd_eval" in sc["ours_v2"]
    parts = [
        "# P1a results\n",
        "Generated by `uv run python -m spatialmem.report.results` from `results/scores/` (provenance: "
        "`results/scores/MANIFEST.md`) and `results/published.yaml`. Do not edit by hand.\n",
        "Scorer: `spatialmem/eval`, labels per CONTRACT 2026-10-27e. Object P / R / F1 are pooled over all scenes of the "
        "set (Hungarian match on 3D AABB IoU at the given threshold); label acc = label accuracy among the matched "
        "pairs at IoU 0.25; rel = relation P / R / F1; mean-scene F1@.25 = mean of the per-scene F1 values.\n",
        f"## HSSD eval {n_eval}\n",
        object_table(sc, "hssd_eval"),
        "\n- ConceptGraphs is the map module (K2, below).\n"
        "- ours v1: no run on this set (" + DASH + ").\n"
        + ("- ours v2: scored on this set.\n" if v2_eval else "- ours v2: pending (run in progress; file not yet in `results/scores/`).\n"),
        f"## HSSD dev {n_dev}\n",
        "Dev scenes were used to tune ours v1 / v2; CG and OVO were not tuned on them.\n",
        object_table(sc, "hssd_dev"),
        "\n- ours v1 here is the lpc run (004, frozen c11).\n",
        "## Replica 8, OVO protocol (semantic mIoU / mAcc / f-mIoU, pooled confusion over 8 scenes, percent)\n",
        semantic_table(sc, pub),
        "\nNotes:\n"
        "1. ConceptGraphs ran with the world-frame pose patch (`hpc/cg_patches/02_world_frame_poses.patch`), so its "
        "numbers may differ from CG in its default relative frame (published 16.7; our 003 reproduction gate allowed ± 3.0).\n"
        "2. OVO's weights-predictor licence is unstated (THIRD_PARTY.md); OVO is used locally for evaluation only.\n"
        "3. K2: ours is below ConceptGraphs after two pre-registered attempts (v1 13.03, v2 14.65 vs CG 15.80 pooled "
        "mIoU), so ConceptGraphs is the map module.\n"
        "4. Published rows are as printed in OVO Table II (HOV-SG was not re-run); no f-mIoU is published there.\n",
        "## Runtime and VRAM (PERUN H200)\n",
        "OVO and ConceptGraphs (`runtime_ovo.csv`, `runtime_cg.csv`):\n",
        baseline_runtime(sc),
        "\n- OVO s/frame = wall time / input frames (all frames of the stream, no stride).\n"
        "- CG s/frame = (detection wall + mapping wall) / frames used (stride 5). Detection ran 4 scenes per GPU and "
        "mapping 8 per node, so CG numbers are inflated by sharing. HSSD rows marked `NA` in the CSV (reconstructed "
        "timings) have no VRAM and are left out of the VRAM column.\n",
        "\nours v2 (`ours_v2_r8_build_runtime.csv`, frozen build, Replica 8):\n",
        ours_runtime(),
        "\n- s/frame = phase wall time / frames used (400 of 2,000, stride 5). Phase A ran 3 scenes per H200, so its "
        "time is inflated by sharing; phase B is the cached-mask descriptor pass.\n"
        "- Association and labelling are CPU work and are not in this table.\n"
        "- The three s/frame definitions differ (all frames vs used frames); compare orders of magnitude, not decimals.\n",
    ]
    return "\n".join(parts).rstrip("\n") + "\n\n" + p1b_section().rstrip("\n") + "\n"


def main() -> None:
    OUT.write_bytes(render().encode("utf-8"))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
