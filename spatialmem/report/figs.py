"""Figures 2 and 3 for the P1a write-up, from the same loaded numbers as RESULTS.md.

Run: uv run python -m spatialmem.report.figs   -> reports/figs/007_fig2_pipeline.png, 007_fig3_results.png
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from spatialmem.report.results import ROOT, load_published, load_scores

FIGS = ROOT / "reports" / "figs"
TOPDOWN = FIGS / "003_102816627_topdown.png"

# one colour per method, same in every panel; published = grey
C = {"cg": "#1f77b4", "ovo": "#e07b00", "ours": "#7a3e9d", "pub": "#9a9a9a"}
INK = "#222222"


def _box(ax, x, y, w, h, text, fc, fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=INK, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.3)


def _arrow(ax, p, q, **kw):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, lw=1.6, color=INK, shrinkA=0, shrinkB=0, **kw))


def fig2_pipeline(path=FIGS / "007_fig2_pipeline.png") -> None:
    fig, ax = plt.subplots(figsize=(15.5, 6.2))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    # data streams
    _box(ax, 0.2, 3.9, 2.9, 1.4, "Habitat-sim HSSD streams\nRGB-D + GT poses\n(rendered on PERUN)", "#e8eef7")
    _box(ax, 0.2, 1.9, 2.9, 1.4, "Replica\nNICE-SLAM RGB-D frames\n+ GT poses", "#e8eef7")
    # mapper group
    ax.add_patch(FancyBboxPatch((4.0, 1.3), 3.5, 4.4, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="#f6f6f6", ec=INK, lw=1.0, ls="--"))
    ax.text(5.75, 5.45, "mappers", ha="center", va="center", fontsize=11, fontweight="bold", color=INK)
    _box(ax, 4.25, 3.95, 3.0, 1.1, "ConceptGraphs\n(map module)", "#cfe2f3", bold=True)
    _box(ax, 4.25, 2.75, 3.0, 1.0, "OVO\n(baseline)", "#fbe0c0")
    _box(ax, 4.25, 1.5, 3.0, 1.0, "ours v1 / v2\n(attempts, K2 fired)", "#e3d3ee")
    # contract, scorer, tables
    _box(ax, 8.1, 2.9, 2.0, 1.6, "Prediction\ncontract\n(docs/CONTRACT.md)", "#fff2cc")
    _box(ax, 10.7, 2.4, 2.4, 2.6, "SCORER\n\nobject P / R / F1 @ IoU\nlabel accuracy\nrelation P / R / F1\nsemantic mIoU", "#d9ead3", fs=9.5)
    _box(ax, 13.7, 3.0, 1.6, 1.4, "tables\n(RESULTS.md)", "#ffffff")
    _box(ax, 9.7, 0.3, 4.4, 1.2, "ground truth from sim assets\n(HSSD scene instances,\nReplica semantic meshes)", "#ffffff", fs=9.5)
    _arrow(ax, (3.2, 4.6), (4.0, 4.2))
    _arrow(ax, (3.2, 2.6), (4.0, 2.9))
    _arrow(ax, (7.5, 3.7), (8.1, 3.7))
    _arrow(ax, (10.1, 3.7), (10.7, 3.7))
    _arrow(ax, (13.1, 3.7), (13.7, 3.7))
    _arrow(ax, (11.9, 1.5), (11.9, 2.4))
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _bars(ax, labels, vals, colors, fmt, hatches=None, errs=None):
    xs = range(len(labels))
    for i, v in enumerate(vals):
        if v is None:
            ax.text(i, 0.02 * ax.get_ylim()[1], "no run" if "v1" in labels[i] else "pending", ha="center",
                    va="bottom", fontsize=9, color="#555555", rotation=90)
            continue
        ax.bar(i, v, color=colors[i], ec=INK, lw=0.8, hatch=(hatches[i] if hatches else None),
               yerr=(errs[i] if errs and errs[i] else None), capsize=4)
        ytxt = v + 0.015 * ax.get_ylim()[1]
        if 14 < v < 15.8:  # keep the label off the dotted K2 line
            ytxt = 16.4
        ax.text(i, ytxt, fmt(v), ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig3_results(path=FIGS / "007_fig3_results.png") -> None:
    sc, pub = load_scores(), load_published()
    fig = plt.figure(figsize=(14, 11.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.3, wspace=0.18, width_ratios=[1, 1.5])

    # (a) HSSD eval 30 object F1@0.25
    ax = fig.add_subplot(gs[0, 0])
    ax.set_ylim(0, 0.6)
    names = ["ConceptGraphs\n(map module)", "OVO", "ours v1", "ours v2"]
    vals, cols = [], [C["cg"], C["ovo"], C["ours"], C["ours"]]
    for key in ("cg", "ovo", "ours_v1", "ours_v2"):
        s = sc.get(key, {}).get("hssd_eval")
        vals.append(s["pooled"]["obj@0.25"]["f1"] if s else None)
    _bars(ax, names, vals, cols, lambda v: f"{v:.3f}")
    ax.set_ylabel("object F1 @ IoU 0.25 (pooled)")
    ax.set_title("(a) HSSD eval 30", loc="left", fontweight="bold")

    # (b) Replica 8 mIoU, published vs our re-runs
    ax = fig.add_subplot(gs[0, 1])
    ax.set_ylim(0, 36)
    short = {"ConceptGraphs": "CG"}
    rows = [(f"{short.get(r['method'], r['method'])}\n(publ.)", r["miou"], C["pub"], "//", None) for r in pub["published"]]
    o = pub["own_eval"]["ovo"]
    rows += [("CG\nours", 100 * sc["cg"]["replica"]["sem_pooled"]["miou"], C["cg"], None, None),
             ("OVO\nown eval", o["miou"], C["ovo"], None, None),
             ("OVO\nour lbl", 100 * sc["ovo"]["replica"]["sem_pooled"]["miou"], C["ovo"], None, None),
             ("ours v1", 100 * sc["ours_v1"]["replica"]["sem_pooled"]["miou"], C["ours"], None, None),
             ("ours v2", 100 * sc["ours_v2"]["replica"]["sem_pooled"]["miou"], C["ours"], None, None)]
    _bars(ax, [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], lambda v: f"{v:.1f}" if v in (16.7, 22.5, 27.0) else f"{v:.2f}",
          hatches=[r[3] for r in rows], errs=[r[4] for r in rows])
    cg = 100 * sc["cg"]["replica"]["sem_pooled"]["miou"]
    ax.axhline(cg, color=C["cg"], lw=1, ls=":", label="K2 bar: CG 15.80 (our re-run)")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.set_ylabel("mIoU (%, pooled, OVO protocol)")
    ax.set_title("(b) Replica 8: published (hatched) vs our re-runs", loc="left", fontweight="bold")
    ax.tick_params(axis="x", labelsize=8.5)

    # (c) per-method top-down, HSSD dev 102816627
    ax = fig.add_subplot(gs[1, :])
    img = mpimg.imread(TOPDOWN)[..., :3]
    ys, xs = (img.min(axis=2) < 0.97).nonzero()
    img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    blank = (img.min(axis=2) >= 0.97).all(axis=1)  # fully white rows
    r = 0
    while not blank[r]:
        r += 1
    top = r
    while blank[r]:
        r += 1
    ax.imshow(np.vstack([img[:top + 6], img[r - 6:]]))
    ax.axis("off")
    ax.set_title("(c) HSSD dev 102816627, GT vs OVO and ConceptGraphs predictions (top-down; 003)", loc="left", fontweight="bold")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---- P1b figures (task 011) ---------------------------------------------------------------------------------------------
P1B = ROOT / "results" / "p1b"
KW = dict(dpi=150, metadata={"Software": None})
HOUSE = "102815859"
P1B_AXES = ("drop", "label", "position", "extra", "relations")
P1B_REPAIRS = ("M", "label", "extra", "miss", "box", "label+extra", "label+extra+miss", "all")
P1B_TYPES = ("existence", "relation", "counting", "find")


def _pj(name: str) -> dict:
    import json
    return json.loads((P1B / name).read_text(encoding="utf-8"))


def p1b_fig1_what(path=FIGS / "011_p1b_fig1_what.png", house: str = HOUSE) -> bool:
    """One house: GT top-down | map text v1.2 excerpt | one question per type with key and 72B correct. Needs data/."""
    import json
    from matplotlib.patches import Rectangle
    from spatialmem.p1b.gtmap import RUNS, load
    from spatialmem.p1b.questions import OUT as QFILE
    from spatialmem.p1b.serialise import serialise
    if not ((RUNS / house / "gt.json").exists() and QFILE.exists()):
        return False
    m = load(house)
    qs = [q for q in map(json.loads, QFILE.read_text().splitlines()) if q["scene"] == house]
    ok = {r[0]: r[2] for r in _pj("per_question/gt.json")}
    fig = plt.figure(figsize=(17, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.5, 1.25], wspace=0.06)
    ax = fig.add_subplot(gs[0])
    for o in m.objects:
        x, z = o.center[0], o.center[2]
        ax.add_patch(Rectangle((x - o.size[0] / 2, z - o.size[2] / 2), o.size[0], o.size[2], fc="#9ecae1", ec="#3b6a8f", lw=0.4, alpha=0.7))
    for o in sorted(m.objects, key=lambda o: -o.size[0] * o.size[2])[:9]:
        ax.text(o.center[0], o.center[2], o.label.replace("_", " "), fontsize=6.5, ha="center", va="center", color=INK)
    xs = [o.center[0] for o in m.objects]
    zs = [o.center[2] for o in m.objects]
    ax.set_xlim(min(xs) - 1.5, max(xs) + 1.5)
    ax.set_ylim(min(zs) - 1.5, max(zs) + 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"GT map, top-down\n(house {house}, {len(m.objects)} objects)", loc="left", fontweight="bold", fontsize=10)
    lines = serialise(m).split("\n")
    k = [i for i, ln in enumerate(lines) if ln.startswith("RELATIONS")]
    ex = lines[:14] + ["..."] + lines[k[0]:k[0] + 4] + ["..."] + lines[k[1]:k[1] + 3] + ["..."]
    ax = fig.add_subplot(gs[1])
    ax.axis("off")
    ax.text(0, 1, "\n".join(ln if len(ln) < 84 else ln[:81] + "..." for ln in ex), family="monospace", fontsize=7.2, va="top", color=INK,
            linespacing=1.35)
    ax.set_title("Map text v1.2 exactly as the planner reads it (excerpt)", loc="left", fontweight="bold", fontsize=10)
    ax = fig.add_subplot(gs[2])
    ax.axis("off")
    y = 1.0
    for t in P1B_TYPES:
        q = next(q for q in qs if q["type"] == t and ok[q["qid"]] == 1)
        key = q["answer"] if t != "find" else f"obj_{q['answer_id']}"
        ax.text(0, y, f"[{t}]", fontsize=9, fontweight="bold", va="top", color=INK)
        ax.text(0, y - 0.045, q["question"], fontsize=8.5, va="top", color=INK)
        ax.text(0, y - 0.10, f"GT answer: {key}     72B: correct ✓", fontsize=8.5, va="top", color="#2e7d32")
        y -= 0.24
    ax.set_title("One question per type (72B, 009 `gt` row)", loc="left", fontweight="bold", fontsize=10)
    fig.savefig(path, bbox_inches="tight", **KW)
    plt.close(fig)
    return True


def p1b_fig2_how(path=FIGS / "011_p1b_fig2_how.png") -> None:
    fig, ax = plt.subplots(figsize=(15.5, 6.6))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 6.6)
    ax.axis("off")
    _box(ax, 0.2, 2.6, 2.7, 1.8, "house map\nGT / degraded / real\n(cg, ovo, ours v2) / repaired\n30 HSSD eval houses", "#e8eef7", fs=9.5)
    _box(ax, 3.6, 2.6, 2.7, 1.8, "map text v1.2\nobjects + relations\n(overflow rule: next_to\ntrimmed to 30,000 tokens)", "#fff2cc", fs=9.5)
    _box(ax, 7.0, 2.6, 2.4, 1.8, "Qwen2.5-VL-72B\nvLLM, temperature 0\nmax 32 tokens", "#cfe2f3", fs=9.5, bold=True)
    _box(ax, 10.0, 2.6, 1.6, 1.8, "raw answer\ntext", "#ffffff", fs=9.5)
    _box(ax, 12.2, 2.6, 3.0, 1.8, "parser + scorer\nvs GT answer key\n(find: label + 1.0 m)", "#d9ead3", fs=9.5)
    for a, b in (((2.9, 3.5), (3.6, 3.5)), ((6.3, 3.5), (7.0, 3.5)), ((9.4, 3.5), (10.0, 3.5)), ((11.6, 3.5), (12.2, 3.5))):
        _arrow(ax, a, b)
    _box(ax, 6.0, 5.2, 4.4, 1.1, "479 questions: existence / relation /\ncounting / find (templated from GT)", "#ffffff", fs=9.5)
    _arrow(ax, (8.2, 5.2), (8.2, 4.4))
    _box(ax, 12.2, 0.6, 3.0, 1.1, "planner accuracy\nper variant, per type", "#ffffff", fs=9.5, bold=True)
    _arrow(ax, (13.7, 2.6), (13.7, 1.7))
    _box(ax, 0.2, 0.4, 3.6, 1.4, "degradation (009)\n5 axes x 4 levels x 3 seeds\ndrop / label / position / extra / relations", "#fbe0c0", fs=9)
    _box(ax, 4.4, 0.4, 3.6, 1.4, "repair (010)\nlabel / extra / miss / box\nfix one real-map error with GT", "#e3d3ee", fs=9)
    _arrow(ax, (1.5, 1.8), (1.5, 2.6))
    _arrow(ax, (6.2, 1.8), (2.4, 2.6))
    ax.text(8.6, 1.0, "map-quality features (F1@.25, label acc,\nrelation F1) -> OLS predictor of accuracy (009)", fontsize=9, color=INK, va="center")
    fig.tight_layout(pad=0.5)
    fig.savefig(path, **KW)
    plt.close(fig)


def p1b_fig3_proof(path=FIGS / "011_p1b_fig3_proof.png") -> None:
    s, runs, rep = _pj("summary.json"), _pj("runs.json"), _pj("repair.json")
    fig = plt.figure(figsize=(18, 5.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1.25], wspace=0.25)
    ax = fig.add_subplot(gs[0])
    for i, a in enumerate(P1B_AXES):
        c = s["curves"][a]
        lv = "/".join(f"{v:g}" for v in c["levels"][1:])
        ax.plot(range(5), [100 * v for v in c["overall"]["mean"]], "-o", color=f"C{i}", ms=4, label=f"{a} ({lv})")
        ax.fill_between(range(5), [100 * v for v in c["overall"]["min"]], [100 * v for v in c["overall"]["max"]], color=f"C{i}", alpha=0.15)
    ax.axhline(85.4, color="k", ls="--", lw=1)
    ax.axhline(24.0, color="0.4", ls="--", lw=1)
    ax.text(0.05, 86.2, "GT 85.4", fontsize=8)
    ax.text(2.6, 25.5, "images-only 24.0", fontsize=8)
    for m, col in zip(("cg", "ovo", "ours_v2"), ("#1f77b4", "#e07b00", "#7a3e9d")):
        ax.plot(4.45, 100 * runs[m]["accuracy"], "D", color=col, ms=7, clip_on=False, label=f"{m} (real)")
    ax.set(xlim=(-0.2, 4.7), ylim=(0, 100), xticks=range(5), xlabel="level index (0 = GT map; levels in legend)", ylabel="planner accuracy (%)")
    ax.set_title("(a) accuracy vs degradation (72B)", loc="left", fontweight="bold")
    ax.legend(fontsize=7.5, loc="lower left")
    ax.grid(alpha=0.3)
    ax = fig.add_subplot(gs[1])
    h = s["headline"]
    x = np.linspace(0, 100, 2)
    for d, al in ((10, 0.10), (5, 0.18)):
        ax.fill_between(x, x - d, x + d, color="C0", alpha=al, lw=0)
    ax.plot(x, x, "k-", lw=1)
    fp = h["fit_points"]
    ax.scatter([100 * p["predicted"] for p in fp.values()], [100 * p["actual"] for p in fp.values()], c="0.6", s=16, label="fit (61)")
    off = {"cg": (-8, 8, "right"), "ours_v2": (8, -14, "left"), "ovo": (8, 6, "left")}
    for v, r in h["real"].items():
        ax.scatter(100 * r["predicted"], 100 * r["actual"], c="C3", s=40, zorder=3)
        ax.annotate(f"{v} ({r['abs_error_pp']:.1f} pp)", (100 * r["predicted"], 100 * r["actual"]), textcoords="offset points",
                    xytext=off[v][:2], ha=off[v][2], fontsize=8)
    ax.set(xlabel="predicted accuracy (%)", ylabel="actual accuracy (%)", xlim=(0, 100), ylim=(0, 100))
    ax.set_title(f"(b) predictor: {h['reading']} (R2 {h['r2']:.2f})\nbands +-5 and +-10 pp", loc="left", fontweight="bold", fontsize=10)
    ax.grid(alpha=0.3)
    ax = fig.add_subplot(gs[2])
    w = 0.1
    for i, b in enumerate(P1B_REPAIRS):
        ys = [100 * rep["accuracy"][m if b == "M" else f"{m}__{b}"]["overall"] for m in ("cg", "ovo", "ours_v2")]
        ax.bar(np.arange(3) + (i - 3.5) * w, ys, w, color=f"C{i}", label=b)
    ax.axhline(85.4, color="k", ls="--", lw=1)
    ax.axhline(24.0, color="0.4", ls="--", lw=1)
    ax.set(xticks=range(3), xticklabels=("cg", "ovo", "ours v2"), ylim=(0, 100), ylabel="planner accuracy (%)")
    ax.set_title("(c) repair one error at a time (010)", loc="left", fontweight="bold")
    ax.legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(path, bbox_inches="tight", **KW)
    plt.close(fig)


def p1b_figs() -> list[str]:
    done = []
    if p1b_fig1_what():
        done.append("011_p1b_fig1_what.png")
    p1b_fig2_how()
    p1b_fig3_proof()
    return done + ["011_p1b_fig2_how.png", "011_p1b_fig3_proof.png"]


def main() -> None:
    fig2_pipeline()
    fig3_results()
    print("wrote 007_fig2_pipeline.png, 007_fig3_results.png, " + ", ".join(p1b_figs()))


if __name__ == "__main__":
    main()
