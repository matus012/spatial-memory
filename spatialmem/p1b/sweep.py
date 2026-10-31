"""P1B_SWEEP §5 analysis from results/p1b/ only: curves, OLS predictor, pre-registered reading, robustness, figures.
`uv run python -m spatialmem.p1b.sweep [--results results/p1b] [--figs reports/figs]`; regenerates byte-identically."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spatialmem.hssd.vocab import ROOT
from spatialmem.p1b.degrade import AXES, LEVELS, REAL, degraded_names
from spatialmem.p1b.jsonio import dump
from spatialmem.p1b.questions import TYPES

REF_GT72B = 409 / 479          # 008 gt72b v1.2 on the GT map, 85.4 %
FEATS = ("f1", "label_acc", "rel_f1")
OK_PP, FAIL_PP = 5.0, 10.0
READING = {"holds": "map scores predict planner success", "fails": "fails", "partial": "partial"}


def load(res: Path) -> dict:
    j = lambda n: json.loads((res / n).read_text())
    return {"features": j("features.json"), "overflow": j("overflow.json"), "runs": j("runs.json"),
            "pq": {p.stem: json.loads(p.read_text()) for p in sorted((res / "per_question").glob("*.json"))}}


def pooled(houses: dict) -> dict:
    """Pooled F1@.25 / label acc / relation F1 from per-house TP / n_pred / n_gt (score_baselines.pool arithmetic)."""
    def f1(tp, npr, ngt):
        p, r = tp / max(npr, 1), tp / max(ngt, 1)
        return 2 * p * r / (p + r) if p + r > 0 else 0.0
    s = {k: sum(h[k] for h in houses.values()) for k in next(iter(houses.values()))}
    return {"f1": f1(s["obj_tp"], s["obj_n_pred"], s["obj_n_gt"]),
            "label_acc": s["label_correct"] / s["obj_tp"] if s["obj_tp"] else float("nan"),
            "rel_f1": f1(s["rel_tp"], s["rel_n_pred"], s["rel_n_gt"])}


def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    ss_res, ss_tot = float(((y - X @ beta) ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    return beta, 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def reading(errs: dict[str, float]) -> str:
    if all(e <= OK_PP for e in errs.values()):
        return READING["holds"]
    return READING["fails"] if any(e > FAIL_PP for e in errs.values()) else READING["partial"]


def predictor(feat: dict[str, dict], acc: dict[str, dict[str, float]], fit: list[str], metric: str) -> dict:
    """OLS acc[metric] ~ 1 + f1 + label_acc + rel_f1 on `fit`; predicted vs actual for the real maps."""
    X = lambda vs: np.array([[1.0, *(feat[v][k] for k in FEATS)] for v in vs])
    y = np.array([acc[v][metric] for v in fit])
    beta, r2 = ols(X(fit), y)
    pred = X(fit) @ beta
    real = {v: {"predicted": float((X([v]) @ beta)[0]), "actual": acc[v][metric]} for v in REAL}
    for r in real.values():
        r["abs_error_pp"] = abs(r["predicted"] - r["actual"]) * 100
    return {"coef": dict(zip(("intercept", *FEATS), map(float, beta))), "r2": r2, "n_fit": len(fit), "real": real,
            "fit_points": {v: {"predicted": float(p), "actual": float(a)} for v, p, a in zip(fit, pred, y)}}


def accuracies(runs: dict) -> dict[str, dict[str, float]]:
    return {v: {"overall": r["accuracy"], **{t: r["per_type"][t]["accuracy"] for t in TYPES}} for v, r in runs.items()}


def curves(acc: dict) -> dict:
    out = {}
    for ax in AXES:
        lv = LEVELS[ax]
        cols = {"overall": [], **{t: [] for t in TYPES}}
        res = {m: {"mean": [acc["gt"][m]], "min": [acc["gt"][m]], "max": [acc["gt"][m]]} for m in cols}   # level 0 = gt
        for l in lv:
            names = [f"{ax}_{l}_s{s}" for s in range(3)]
            for m in cols:
                v = [acc[n][m] for n in names]
                res[m]["mean"].append(float(np.mean(v)))
                res[m]["min"].append(min(v))
                res[m]["max"].append(max(v))
        out[ax] = {"levels": [0.0] + [float(l) for l in lv], **res}
    return out


def analyse(res: Path) -> dict:
    d = load(res)
    feat, runs, over = d["features"], d["runs"], d["overflow"]
    acc = accuracies(runs)
    fit = ["gt"] + degraded_names()
    head = predictor(feat, acc, fit, "overall")
    head["reading"] = reading({v: r["abs_error_pp"] for v, r in head["real"].items()})
    clean = sorted(s for s in next(iter(over.values())) if all(o[s]["removed"] == 0 for o in over.values()))
    rob = None
    if clean:
        keep = set(clean)
        feat2 = {v: pooled({s: h for s, h in feat[v]["houses"].items() if s in keep}) for v in feat}
        acc2 = {v: {"overall": float(np.mean([q[2] for q in d["pq"][v] if q[0].split(":")[0] in keep]))} for v in feat}
        rob = predictor(feat2, acc2, fit, "overall")
        rob["reading"] = reading({v: r["abs_error_pp"] for v, r in rob["real"].items()})
        rob["houses"] = clean
    return {
        "headline": head,
        "robustness": rob,
        "per_type": {t: predictor(feat, acc, fit, t) for t in TYPES},
        "curves": curves(acc),
        "reference_gt72b_008": REF_GT72B,
        "images_row": runs.get("img"),
        "points": {v: {"features": {k: feat[v][k] for k in FEATS}, "features_post_overflow": feat[v]["post"],
                       "accuracy": acc[v], "houses_overflow": sorted(s for s, r in over[v].items() if r["removed"] > 0)}
                   for v in feat},
    }


def figures(s: dict, figs: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs.mkdir(parents=True, exist_ok=True)
    fkw = dict(dpi=120, metadata={"Software": None})
    lines = ("overall", *TYPES)
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    for ax, name in zip(axs.flat, AXES):
        c = s["curves"][name]
        for i, m in enumerate(lines):
            col = f"C{i}"
            ax.plot(c["levels"], [100 * x for x in c[m]["mean"]], "-o", color=col, lw=2.2 if m == "overall" else 1.2, ms=3, label=m)
            ax.fill_between(c["levels"], [100 * x for x in c[m]["min"]], [100 * x for x in c[m]["max"]], color=col, alpha=0.15)
        ax.axhline(100 * s["reference_gt72b_008"], color="k", ls=":", lw=1)
        ax.set(title=name, xlabel="sigma (m)" if name == "position" else "level (%)", ylabel="planner accuracy (%)", ylim=(0, 100))
        ax.grid(alpha=0.3)
    axs.flat[5].axis("off")
    h, l = axs.flat[0].get_legend_handles_labels()
    axs.flat[5].legend(h, l, loc="center", title="mean (band = min-max over 3 seeds)\ndotted = 008 gt72b 85.4 %")
    fig.tight_layout()
    fig.savefig(figs / "009_curves.png", **fkw)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    x = np.linspace(0, 100, 2)
    for d, a in ((10, 0.10), (5, 0.18)):
        ax.fill_between(x, x - d, x + d, color="C0", alpha=a, lw=0)
    ax.plot(x, x, "k-", lw=1)
    fp = s["headline"]["fit_points"]
    ax.scatter([100 * p["predicted"] for p in fp.values()], [100 * p["actual"] for p in fp.values()], c="0.6", s=18, label="fit (61)")
    for v, r in s["headline"]["real"].items():
        ax.scatter(100 * r["predicted"], 100 * r["actual"], c="C3", s=45, zorder=3)
        off = {"cg": (-8, 8), "ours_v2": (8, -14)}.get(v, (8, 4))   # cg and ours_v2 sit close: keep their labels apart
        ax.annotate(f"{v} ({r['abs_error_pp']:.1f} pp)", (100 * r["predicted"], 100 * r["actual"]), textcoords="offset points",
                    xytext=off, ha="right" if off[0] < 0 else "left")
    ax.set(xlabel="predicted accuracy (%)", ylabel="actual accuracy (%)", xlim=(0, 100), ylim=(0, 100),
           title=f"R2 {s['headline']['r2']:.3f}; reading: {s['headline']['reading']}\n(bands: +-5, +-10 pp)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs / "009_predictor.png", **fkw)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=ROOT / "results" / "p1b")
    ap.add_argument("--figs", type=Path, default=ROOT / "reports" / "figs")
    a = ap.parse_args()
    s = analyse(a.results)
    dump(a.results / "summary.json", s)
    figures(s, a.figs)
    h = s["headline"]
    print("coef", {k: round(v, 4) for k, v in h["coef"].items()}, "R2", round(h["r2"], 4))
    for v, r in h["real"].items():
        print(f"{v:8s} predicted {100 * r['predicted']:5.1f}  actual {100 * r['actual']:5.1f}  abs err {r['abs_error_pp']:.1f} pp")
    print("reading:", h["reading"])
    if s["robustness"]:
        print("robustness (", len(s["robustness"]["houses"]), "houses ):", s["robustness"]["reading"],
              {v: round(r["abs_error_pp"], 1) for v, r in s["robustness"]["real"].items()})


if __name__ == "__main__":
    main()
