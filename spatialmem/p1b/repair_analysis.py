"""P1B_REPAIR §3-4: attribution of the real-map error from results/p1b/ (009 rows + results/p1b/repair/ rows).
`uv run python -m spatialmem.p1b.repair_analysis [--results results/p1b] [--figs reports/figs]`; byte-identical regeneration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spatialmem.hssd.vocab import ROOT
from spatialmem.p1b.degrade import REAL
from spatialmem.p1b.jsonio import dump
from spatialmem.p1b.questions import TYPES
from spatialmem.p1b.repair import KINDS

SINGLE = ("label", "extra", "miss", "box")
COMBO = ("label+extra", "label+extra+miss", "all")
GT_PCT, IMG_PCT = 85.4, 24.0         # reference lines (P1B_REPAIR §3)
DOMINANT_PP, G2_PP = 5.0, 3.0


def acc(pq: list) -> dict[str, float]:
    return {"overall": float(np.mean([r[2] for r in pq])),
            **{t: float(np.mean([r[2] for r in pq if r[1] == t])) for t in TYPES}}


def dominant(gains: dict[str, float]) -> dict:
    """Largest single-repair gain, if it beats the second by >= 5.0 points."""
    order = sorted(SINGLE, key=lambda k: (-gains[k], k))
    margin = gains[order[0]] - gains[order[1]]
    return {"margin_pp": margin, "top": order[0], "second": order[1],
            "reading": order[0] if margin >= DOMINANT_PP else "no single dominant error"}


def analyse(res: Path) -> dict:
    rd = res / "repair"
    j = lambda p: json.loads(p.read_text())
    a = {m: acc(j(res / "per_question" / f"{m}.json")) for m in REAL}
    for m in REAL:
        for k in KINDS:
            a[f"{m}__{k}"] = acc(j(rd / "per_question" / f"{m}__{k}.json"))
    out: dict = {"features": j(rd / "features.json"), "overflow": j(rd / "overflow.json"), "accuracy": a,
                 "per_question": {f"{m}__{k}": j(rd / "per_question" / f"{m}__{k}.json") for m in REAL for k in KINDS},
                 "reference_pct": {"gt": GT_PCT, "images_only": IMG_PCT}, "maps": {}}
    for m in REAL:
        g2 = 100 * a[f"{m}__all"]["overall"]
        r = {"g2": {"all_pct": g2, "delta_pp": g2 - GT_PCT, "ok": abs(g2 - GT_PCT) <= G2_PP}}
        for scope in ("overall", *TYPES):
            gains = {k: 100 * (a[f"{m}__{k}"][scope] - a[m][scope]) for k in SINGLE}
            r[scope] = {"gains_pp": gains, "dominant": dominant(gains),
                        "cumulative_gain_pp": {k: 100 * (a[f"{m}__{k}"][scope] - a[m][scope]) for k in COMBO}}
        out["maps"][m] = r
    return out


def figure(s: dict, figs: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs.mkdir(parents=True, exist_ok=True)
    bars = ("M", *SINGLE, *COMBO)
    fig, ax = plt.subplots(figsize=(12, 5))
    w = 0.1
    for i, b in enumerate(bars):
        ys = [100 * s["accuracy"][m if b == "M" else f"{m}__{b}"]["overall"] for m in REAL]
        ax.bar(np.arange(3) + (i - 3.5) * w, ys, w, color=f"C{i}", label=b)
    ax.axhline(s["reference_pct"]["gt"], color="k", ls="--", lw=1, label="GT 85.4")
    ax.axhline(s["reference_pct"]["images_only"], color="0.4", ls="--", lw=1, label="images-only 24.0")
    ax.set(xticks=range(3), xticklabels=REAL, ylabel="planner accuracy (%)", ylim=(0, 100), title="Repair attribution (72B, label+dist)")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs / "010_repair.png", dpi=120, metadata={"Software": None})
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=ROOT / "results" / "p1b")
    ap.add_argument("--figs", type=Path, default=ROOT / "reports" / "figs")
    a = ap.parse_args()
    s = analyse(a.results)
    dump(a.results / "repair.json", s)
    figure(s, a.figs)
    for m, r in s["maps"].items():
        o = r["overall"]
        print(f"{m:8s} acc(M) {100 * s['accuracy'][m]['overall']:.1f}  gains " + "  ".join(f"{k} {v:+.1f}" for k, v in o["gains_pp"].items()),
              f"| dominant: {o['dominant']['reading']} (margin {o['dominant']['margin_pp']:.1f}) | all {r['g2']['all_pct']:.1f} G2 {'ok' if r['g2']['ok'] else 'FLAG'}")


if __name__ == "__main__":
    main()
