"""Task 001 demo figures.

reports/figs/001_room0_strip.png   — 3 frames: RGB | dataset depth | rendered depth | rendered instance IDs
reports/figs/001_room0_gt_topdown.png — GT AABBs (horizontal footprint) + labels + relation edges

Run: uv run python -m spatialmem.viz [--scene room0]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from spatialmem.gt.replica import ROOT  # noqa: E402
from spatialmem.render.proof import dataset_depth, dataset_rgb, intrinsics, poses, semantic_raycaster  # noqa: E402

FIGS = ROOT / "reports" / "figs"


def instance_colors(ids: np.ndarray) -> np.ndarray:
    """Deterministic pseudo-random colour per id; misses black."""
    h = (ids.astype(np.int64) * 2654435761) % (2**32)
    rgb = np.stack([(h >> 0) & 255, (h >> 8) & 255, (h >> 16) & 255], -1).astype(np.uint8)
    rgb[ids < 0] = 0
    return rgb


def strip(scene: str, frames: tuple[int, ...] = (0, 900, 1800)) -> None:
    intr, scale = intrinsics(scene)
    T = poses(scene)
    rc, _, _ = semantic_raycaster(scene)
    fig, ax = plt.subplots(len(frames), 4, figsize=(20, 3.1 * len(frames)))
    for r, i in enumerate(frames):
        d_ren, inst = rc.render(intr, T[i])
        d_ds = dataset_depth(scene, i, scale)
        vmax = float(np.percentile(d_ds[d_ds > 0], 99))
        valid = (d_ds > 0) & (d_ren > 0)
        med = float(np.median(np.abs(d_ren - d_ds)[valid]) * 100)
        panels = [(dataset_rgb(scene, i), f"frame {i}: RGB"),
                  (d_ds, "dataset depth (png / 6553.5)"),
                  (d_ren, f"rendered depth, mesh_semantic (median |diff| {med:.2f} cm)"),
                  (instance_colors(inst), f"rendered instance IDs ({len(np.unique(inst[inst >= 0]))} objects)")]
        for c, (img, title) in enumerate(panels):
            a = ax[r, c]
            if img.ndim == 2:
                a.imshow(img, cmap="viridis", vmin=0, vmax=vmax)
            else:
                a.imshow(img)
            a.set_title(title, fontsize=10)
            a.axis("off")
    fig.suptitle(f"Replica {scene}: dataset vs raycast of habitat/mesh_semantic.ply at NICE-SLAM poses", fontsize=13)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"001_{scene}_strip.png", dpi=80)
    plt.close(fig)


def topdown(scene: str) -> None:
    d = json.loads((ROOT / "gt" / f"{scene}.json").read_text())
    up = d["up_axis"]
    h = [k for k in range(3) if k != up]
    objs = {o["id"]: o for o in d["objects"]}
    fig, ax = plt.subplots(figsize=(14, 12))
    # structural context: floor footprint from all_objects
    for o in d["all_objects"]:
        if o["class_name"] == "floor":
            lo, hi = np.array(o["aabb_min"]), np.array(o["aabb_max"])
            ax.add_patch(Rectangle((lo[h[0]], lo[h[1]]), hi[h[0]] - lo[h[0]], hi[h[1]] - lo[h[1]],
                                   fc="#eeeeee", ec="#999999", lw=1, zorder=0))
    order = sorted(objs.values(), key=lambda o: -np.prod(np.subtract(o["aabb_max"], o["aabb_min"])[h]))
    for o in order:
        lo, hi = np.array(o["aabb_min"]), np.array(o["aabb_max"])
        col = instance_colors(np.array([o["id"]]))[0] / 255
        ax.add_patch(Rectangle((lo[h[0]], lo[h[1]]), hi[h[0]] - lo[h[0]], hi[h[1]] - lo[h[1]],
                               fc=(*col, 0.25), ec=col, lw=1.2, zorder=1))
    cen = {i: (np.array(o["aabb_min"]) + np.array(o["aabb_max"]))[h] / 2 for i, o in objs.items()}
    style = {"on": ("tab:red", "-", 1.6), "in": ("tab:blue", "-", 1.6), "next_to": ("tab:gray", ":", 0.8)}
    for r in d["relations"]:
        a, b = cen[r["subj"]], cen[r["obj"]]
        c, ls, lw = style[r["pred"]]
        if r["pred"] == "next_to":
            ax.plot([a[0], b[0]], [a[1], b[1]], color=c, ls=ls, lw=lw, zorder=2)
        else:
            ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color=c, lw=lw), zorder=3)
    nv = {o["id"]: o["n_vertices"] for o in d["all_objects"]}
    big = sorted(objs.values(), key=lambda o: -nv[o["id"]])[:45]
    for o in big:
        ax.text(*cen[o["id"]], o["label"], fontsize=7, ha="center", va="center", zorder=4,
                bbox=dict(fc="white", ec="none", alpha=0.6, pad=0.5))
    n = {k: sum(r["pred"] == k for r in d["relations"]) for k in style}
    ax.plot([], [], color="tab:red", label=f"on ({n['on']}), arrow A->B")
    ax.plot([], [], color="tab:blue", label=f"in ({n['in']}), arrow A->B")
    ax.plot([], [], color="tab:gray", ls=":", label=f"next_to ({n['next_to']})")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    ax.autoscale_view()
    axes = "xyz"
    ax.set_xlabel(f"{axes[h[0]]} [m]")
    ax.set_ylabel(f"{axes[h[1]]} [m]")
    ax.set_title(f"Replica {scene} GT: {len(objs)} objects (AABB footprints, 51-class protocol, structural "
                 f"excluded), {len(d['relations'])} relations (docs/RELATIONS.md), up = {axes[up]}")
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"001_{scene}_gt_topdown.png", dpi=90)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="room0")
    ap.add_argument("--hssd-root", type=Path, default=None, help="data/hssd_runs/dev -> 002 HSSD figures for --scene")
    a = ap.parse_args()
    if a.hssd_root:
        print("wrote", hssd_strip(a.scene, a.hssd_root).name, hssd_topdown(a.scene, a.hssd_root).name)
        return
    strip(a.scene)
    topdown(a.scene)
    print("wrote", sorted(p.name for p in FIGS.glob(f"001_{a.scene}_*.png")))



# ---------------------------------------------------------------- task 002 (HSSD)
def hssd_strip(scene: str, root: Path, frames: tuple[int, ...] | None = None) -> Path:
    """reports/figs/002_<scene>_strip.png — RGB | depth | instance IDs for 3 frames of the coverage stream."""
    from PIL import Image
    D = root / scene
    meta = json.loads((D / "meta.json").read_text())
    n = meta["n_frames"]
    frames = frames or (n // 6, n // 2, 5 * n // 6)
    fig, ax = plt.subplots(len(frames), 3, figsize=(15, 3.9 * len(frames)))
    for r, i in enumerate(frames):
        d = np.asarray(Image.open(D / "depth" / f"{i:06d}.png"), float) / meta["depth_scale"]
        inst = np.asarray(Image.open(D / "inst" / f"{i:06d}.png")).astype(np.int64)
        panels = [(np.asarray(Image.open(D / "rgb" / f"{i:06d}.jpg")), f"frame {i}/{n}: RGB"),
                  (d, f"depth (m), valid {np.mean(d > 0):.0%}"),
                  (instance_colors(np.where(inst > 0, inst, -1)), f"instance IDs ({len(np.unique(inst[inst > 0]))} objects; black = stage)")]
        for c, (img, title) in enumerate(panels):
            a = ax[r, c]
            a.imshow(img, cmap="viridis") if img.ndim == 2 else a.imshow(img)
            a.set_title(title, fontsize=10)
            a.axis("off")
    fig.suptitle(f"HSSD {scene}: Habitat-sim 0.3.3 on H200 (EGL), coverage stream", fontsize=13)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / f"002_{scene}_strip.png"
    fig.savefig(out, dpi=70)
    plt.close(fig)
    return out


def hssd_topdown(scene: str, root: Path) -> Path:
    """reports/figs/002_<scene>_gt_topdown.png — GT AABB footprints (x-z, up = +y) + relations + camera path."""
    D = root / scene
    g = json.loads((D / "gt.json").read_text())
    objs = {o["id"]: o for o in g["objects"]}
    h = [0, 2]
    P = np.loadtxt(D / "poses.txt").reshape(-1, 4, 4)[:, :3, 3]
    seen = {int(k) for k, v in json.loads((D / "check.json").read_text())["per_object"].items() if v["max_px"] >= 200}
    fig, ax = plt.subplots(figsize=(14, 12))
    for o in sorted(objs.values(), key=lambda o: -np.prod(np.subtract(o["aabb_max"], o["aabb_min"])[h])):
        lo, hi = np.array(o["aabb_min"]), np.array(o["aabb_max"])
        col = instance_colors(np.array([o["id"]]))[0] / 255
        ax.add_patch(Rectangle((lo[0], lo[2]), hi[0] - lo[0], hi[2] - lo[2], fc=(*col, 0.22), ec=col,
                               lw=1.0 if o["id"] in seen else 2.0, ls="-" if o["id"] in seen else "--", zorder=1))
    cen = {i: (np.array(o["aabb_min"]) + np.array(o["aabb_max"]))[h] / 2 for i, o in objs.items()}
    style = {"on": ("tab:red", 1.4), "in": ("tab:blue", 1.4)}
    for r in g["relations"]:
        if r["pred"] in style:
            c, lw = style[r["pred"]]
            ax.annotate("", xy=cen[r["obj"]], xytext=cen[r["subj"]], arrowprops=dict(arrowstyle="->", color=c, lw=lw), zorder=3)
    ax.plot(P[:, 0], P[:, 2], color="black", lw=0.6, alpha=0.6, zorder=2)
    ax.scatter(P[::8, 0], P[::8, 2], s=4, color="black", zorder=2)
    nv = {g_["id"]: len(g_["members"]) for g_ in g["gt_groups"]}
    big = sorted(objs.values(), key=lambda o: -np.prod(np.subtract(o["aabb_max"], o["aabb_min"])))[:40]
    for o in big:
        ax.text(*cen[o["id"]], o["label"] + ("*" if nv.get(o["id"], 1) > 1 else ""), fontsize=7, ha="center", va="center",
                zorder=4, bbox=dict(fc="white", ec="none", alpha=0.6, pad=0.5))
    n = {k: sum(r["pred"] == k for r in g["relations"]) for k in ("on", "in", "next_to")}
    ax.plot([], [], color="tab:red", label=f"on ({n['on']}), arrow A->B")
    ax.plot([], [], color="tab:blue", label=f"in ({n['in']}), arrow A->B")
    ax.plot([], [], color="black", label=f"camera path ({len(P)} frames)")
    ax.plot([], [], color="gray", ls="--", label=f"dashed box = not seen (< 200 px), {len(objs) - len(seen & set(objs))} objects")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.invert_yaxis()
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(f"HSSD {scene} GT: {len(objs)} objects (* = merged decomposed parts), {len(g['relations'])} relations "
                 f"(next_to {n['next_to']} not drawn), docs/RELATIONS.md v2")
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / f"002_{scene}_gt_topdown.png"
    fig.savefig(out, dpi=80)
    plt.close(fig)
    return out

# ---------------------------------------------------------------- task 003 (baselines)
def baselines_topdown(scene: str, methods: tuple[str, ...] = ("ovo", "cg"), tau: float = 0.25,
                      out_name: str | None = None) -> Path:
    """reports/figs/<out_name, default 003_<scene>_topdown.png> — per method: GT AABB footprints vs predicted, matched (green) /
    extra prediction (red) / missed GT (orange, dashed) at IoU >= tau, x-z plane (HSSD up = +y)."""
    from spatialmem.eval.contract import Prediction
    from spatialmem.eval.objects import score_objects
    from spatialmem.hssd.gt import load_hssd_gt
    from spatialmem.hssd.vocab import canonical
    gt, _ = load_hssd_gt(ROOT / "data" / "hssd_runs" / "all" / scene / "gt.json")
    fig, axs = plt.subplots(1, len(methods), figsize=(9 * len(methods), 8.5), squeeze=False)
    for ax, m in zip(axs[0], methods):
        pred = Prediction.from_json(ROOT / "data" / "baselines" / m / "hssd" / f"{scene}.json")
        for o in pred.objects:
            o.label = canonical(o.label)
        sc = score_objects(pred, gt, tau)
        mp, mg = {p for p, _, _ in sc.matches}, {g for _, g, _ in sc.matches}
        for o in gt.objects:
            lo, hi = o.aabb_min, o.aabb_max
            hit = o.id in mg
            ax.add_patch(Rectangle((lo[0], lo[2]), hi[0] - lo[0], hi[2] - lo[2], fc="none",
                                   ec="#999999" if hit else "tab:orange", lw=0.8 if hit else 1.4, ls="-" if hit else "--", zorder=1))
        for o in pred.objects:
            lo, hi = o.aabb_min, o.aabb_max
            ok = o.id in mp
            ax.add_patch(Rectangle((lo[0], lo[2]), hi[0] - lo[0], hi[2] - lo[2], fc=("tab:green" if ok else "tab:red", 0.15),
                                   ec="tab:green" if ok else "tab:red", lw=1.0, zorder=2))
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.invert_yaxis()
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")
        ax.set_title(f"{m}: {sc.n_pred} pred, {sc.n_gt} GT, TP {sc.tp} @ IoU {tau} -> P {sc.precision:.2f} R {sc.recall:.2f} "
                     f"F1 {sc.f1:.2f}, label acc {sc.label_acc:.2f}", fontsize=10)
        ax.plot([], [], color="tab:green", label=f"matched prediction ({sc.tp})")
        ax.plot([], [], color="tab:red", label=f"extra prediction ({sc.n_pred - sc.tp})")
        ax.plot([], [], color="tab:orange", ls="--", label=f"missed GT ({sc.n_gt - sc.tp})")
        ax.legend(loc="upper right", fontsize=9)
    fig.suptitle(f"HSSD {scene} (dev): GT boxes vs baselines (GT poses, CONTRACT labels)", fontsize=13)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / (out_name or f"003_{scene}_topdown.png")
    fig.savefig(out, dpi=75)
    plt.close(fig)
    return out


# ---------------------------------------------------------------- task 006 (ours v2, Replica-dev)
def replica_dev_topdown(scene: str, methods: tuple[str, ...], titles: tuple[str, ...], tau: float = 0.25) -> Path:
    """reports/figs/006_<scene>_topdown.png — per method: Replica-dev GT AABB footprints vs predicted, matched (green) /
    extra prediction (red) / missed GT (orange, dashed) at IoU >= tau, x-y plane (Replica up = +z); scored as the scorer does."""
    from spatialmem.eval.objects import score_objects
    from spatialmem.eval.score_baselines import load_pred
    from spatialmem.gt.replica import load_gt
    gt = load_gt(scene, with_vertices=False)
    fig, axs = plt.subplots(1, len(methods), figsize=(9 * len(methods), 8.5), squeeze=False)
    for ax, m, t in zip(axs[0], methods, titles):
        pred = load_pred(m, "replica_dev", scene, hssd=False)
        sc = score_objects(pred, gt, tau)
        mp, mg = {p for p, _, _ in sc.matches}, {g for _, g, _ in sc.matches}
        for o in gt.objects:
            lo, hi = o.aabb_min, o.aabb_max
            hit = o.id in mg
            ax.add_patch(Rectangle((lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1], fc="none",
                                   ec="#999999" if hit else "tab:orange", lw=0.8 if hit else 1.4, ls="-" if hit else "--", zorder=1))
        for o in pred.objects:
            lo, hi = o.aabb_min, o.aabb_max
            ok = o.id in mp
            ax.add_patch(Rectangle((lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1], fc=("tab:green" if ok else "tab:red", 0.15),
                                   ec="tab:green" if ok else "tab:red", lw=1.0, zorder=2))
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"{t}: {sc.n_pred} pred, {sc.n_gt} GT, TP {sc.tp} @ IoU {tau} -> P {sc.precision:.2f} R {sc.recall:.2f} "
                     f"F1 {sc.f1:.2f}, label acc {sc.label_acc:.2f}", fontsize=10)
        ax.plot([], [], color="tab:green", label=f"matched prediction ({sc.tp})")
        ax.plot([], [], color="tab:red", label=f"extra prediction ({sc.n_pred - sc.tp})")
        ax.plot([], [], color="tab:orange", ls="--", label=f"missed GT ({sc.n_gt - sc.tp})")
        ax.legend(loc="upper right", fontsize=9)
    fig.suptitle(f"Replica-dev {scene} (tuning scene, PTex render): GT boxes vs ours v1 / v2 (GT poses, CONTRACT labels)",
                 fontsize=13)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / f"006_{scene}_topdown.png"
    fig.savefig(out, dpi=75)
    plt.close(fig)
    return out


# ---------------------------------------------------------------- task 007 (P1a demo figure)
def house_map_graph(scene: str, data: Path | None = None, n_nodes: int = 25) -> Path:
    """reports/figs/007_fig1_house_map_graph.png — HSSD <scene>, ConceptGraphs map: (a) top-down points per object + boxes,
    (b) scene graph of the n_nodes best-connected objects (edges labelled by relation type), (c) GT vs map match colouring
    @ IoU 0.25 (matched green / extra red / missed GT orange). x-z plane, HSSD up = +y."""
    from spatialmem.demo import data_dir, load, statuses, subsample
    from spatialmem.eval.contract import STRUCTURAL
    data = data or data_dir()
    pred, gt, sc, npz = load(scene, data)
    ps, gs = statuses(pred, gt, sc)
    pts, off = subsample(npz, len(pred.objects), 250_000)
    own = np.repeat(np.arange(len(pred.objects)), np.diff(off))
    scored = [o for o in pred.objects if o.label not in STRUCTURAL]
    fig = plt.figure(figsize=(30, 8.6))
    gs_ = fig.add_gridspec(1, 3, width_ratios=[1, 1.25, 1], wspace=0.08)
    axa, axb, axc = (fig.add_subplot(gs_[0, i]) for i in range(3))

    def box(ax, o, **kw):
        ax.add_patch(Rectangle((o.aabb_min[0], o.aabb_min[2]), o.aabb_max[0] - o.aabb_min[0], o.aabb_max[2] - o.aabb_min[2], **kw))

    def frame(ax, title):
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.invert_yaxis()
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")
        ax.set_title(title, fontsize=14)

    # (a) points + boxes (structural objects' points and boxes left out: ceiling would hide the rooms)
    keep = np.array([pred.objects[i].label not in STRUCTURAL for i in own], bool)
    ax = axa
    ax.scatter(pts[keep, 0], pts[keep, 2], s=0.6, c=instance_colors(own[keep]) / 255, marker=".", linewidths=0, zorder=1)
    for o in scored:
        box(ax, o, fc="none", ec=instance_colors(np.array([o.id]))[0] / 255, lw=0.7, zorder=2)
    ax.set_xlim(min(o.aabb_min[0] for o in scored) - 0.5, max(o.aabb_max[0] for o in scored) + 0.5)
    ax.set_ylim(max(o.aabb_max[2] for o in scored) + 0.5, min(o.aabb_min[2] for o in scored) - 0.5)
    frame(ax, f"(a) ConceptGraphs map: {len(scored)} objects, points + boxes (per-object colour)")

    # (b) scene graph: the n_nodes objects with most on/in/next_to relations among non-structural objects (ties: larger)
    ids = {o.id for o in scored}
    rels = [r for r in pred.relations if r.subj in ids and r.obj in ids]
    vol = {o.id: float(np.prod(o.aabb_max - o.aabb_min)) for o in scored}
    nodes: list[int] = []   # on / in edges carry the structure: add their endpoints (largest first) until n_nodes
    for r in sorted((r for r in rels if r.pred != "next_to"), key=lambda r: -(vol[r.subj] + vol[r.obj])):
        for i in (r.subj, r.obj):
            if i not in nodes and len(nodes) < n_nodes:
                nodes.append(i)
    nodes += sorted(ids - set(nodes), key=lambda i: -vol[i])[:n_nodes - len(nodes)]
    nset = set(nodes)
    edges = [r for r in rels if r.subj in nset and r.obj in nset]
    import networkx as nx   # torch dependency
    gr = nx.Graph()
    gr.add_nodes_from(nodes)
    gr.add_edges_from((r.subj, r.obj) for r in edges)
    P_: dict[int, np.ndarray] = {}   # one spring layout per connected component, components packed on a grid
    comps = sorted(nx.connected_components(gr), key=len, reverse=True)
    ncol = 4
    for c, comp in enumerate(comps):
        sub = gr.subgraph(comp)
        loc = nx.spring_layout(sub, k=1.0 / np.sqrt(len(comp)), seed=114, iterations=200, scale=0.52) if len(comp) > 1 else {next(iter(comp)): np.zeros(2)}
        for i, v in loc.items():
            P_[i] = v + np.array([c % ncol, c // ncol]) * 1.2
    ix = {i: k for k, i in enumerate(nodes)}
    pos = np.array([P_[i] for i in nodes])
    style = {"on": "tab:red", "in": "tab:blue", "next_to": "tab:gray"}
    ax = axb
    seen: set = set()
    for r in edges:
        key = r.key()
        if key in seen:
            continue
        seen.add(key)
        a, b = pos[ix[r.subj]], pos[ix[r.obj]]
        ax.annotate("", xy=b, xytext=a, zorder=1, arrowprops=dict(arrowstyle="-" if r.pred == "next_to" else "-|>", color=style[r.pred],
                                                                    lw=1.6 if r.pred != "next_to" else 1.0, ls=":" if r.pred == "next_to" else "-",
                                                                    shrinkA=20, shrinkB=20, mutation_scale=16))
        ax.text(*((a + b) / 2), r.pred, fontsize=8.5, color=style[r.pred], ha="center", va="center", zorder=2,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.4))
    for i in nodes:
        st = ps[i][0]
        ax.scatter(*pos[ix[i]], s=1400, c="#d9f2dd" if st == "m" else "#fbdcdc", ec="tab:green" if st == "m" else "tab:red", lw=2, zorder=3)
        ax.text(*pos[ix[i]], pred.objects[i].label, fontsize=10.5, ha="center", va="center", zorder=4, fontweight="bold")
    ax.margins(0.06)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    n = {k: len({r.key() for r in edges if r.pred == k}) for k in style}
    for k, c in style.items():
        ax.plot([], [], color=c, ls=":" if k == "next_to" else "-", label=f"{k} ({n[k]})")
    ax.scatter([], [], s=200, c="#d9f2dd", ec="tab:green", label="object matched to GT")
    ax.scatter([], [], s=200, c="#fbdcdc", ec="tab:red", label="extra object (no GT match)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=11, frameon=False)
    ax.set_title(f"(b) scene graph: {len(nodes)} objects (largest on/in pairs first) of {len(scored)} objects, {len(edges)} of {len(rels)} relations "
                 f"(arrow subject -> object)", fontsize=14)

    # (c) GT vs map
    ax = axc
    for o in gt.objects:
        if o.label in STRUCTURAL:
            continue
        hit = gs[o.id][0] == "m"
        box(ax, o, fc="none", ec="#8a8a8a" if hit else "tab:orange", lw=0.9 if hit else 1.6, ls="-" if hit else "--", zorder=1)
    for o in scored:
        ok = ps[o.id][0] == "m"
        box(ax, o, fc=("tab:green" if ok else "tab:red", 0.18), ec="tab:green" if ok else "tab:red", lw=1.0, zorder=2)
    ax.set_xlim(axa.get_xlim())
    ax.set_ylim(axa.get_ylim())
    ax.plot([], [], color="tab:green", label=f"matched prediction ({sc.tp})")
    ax.plot([], [], color="tab:red", label=f"extra prediction ({sc.n_pred - sc.tp})")
    ax.plot([], [], color="tab:orange", ls="--", label=f"missed GT ({sc.n_gt - sc.tp})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, fontsize=12, frameon=False)
    frame(ax, f"(c) GT vs map @ IoU 0.25: P {sc.precision:.2f}  R {sc.recall:.2f}  F1 {sc.f1:.2f}  label acc {sc.label_acc:.2f}")
    fig.suptitle(f"HSSD {scene} (dev): ConceptGraphs map as a 3D object map + scene graph, scored against sim ground truth", fontsize=17)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.91, bottom=0.1)
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "007_fig1_house_map_graph.png"
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out


if __name__ == "__main__":
    main()
