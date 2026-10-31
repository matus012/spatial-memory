"""114 demo video + README GIF, from saved files only (task 013). HSSD only — no Replica imagery.

Segments (one caption each): title · (a) the camera walking the house · (b) the ConceptGraphs 3D map orbiting,
coloured matched / extra / missed vs GT · (c) three planner questions on that house (saved 009 outputs) · (d) closing card.

Inputs (under `<repo>/data` or $SPATIALMEM_DATA, git-ignored):
  hssd_streams/<scene>/rgb/*.jpg, poses.txt       the agent's RGB stream (002 renderer; eval houses: fetched from PERUN runs/hssd/<scene>)
  baselines/cg/hssd/<scene>.json + runs/.../objects.npz, hssd_runs/all/<scene>/gt.json   (as spatialmem.demo)
  media/114_demo_qa.json (committed)              the three questions with the planner's raw answers
Outputs: media/114_demo.mp4 (h264, 1280x720, 24 fps), media/114_demo.gif (800x330, 10 fps loop).

Run:
  uv run --group media python scripts/make_demo_video.py
  uv run --group media python scripts/make_demo_video.py --extract-qa <evidence>/2026-10-29-009/raw   # rebuild the qa json
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image, ImageDraw, ImageFont

from spatialmem.demo import data_dir, load, statuses, subsample
from spatialmem.eval.contract import STRUCTURAL

REPO = Path(__file__).resolve().parents[1]
MEDIA = REPO / "media"
QA = MEDIA / "114_demo_qa.json"
SCENE = "102815859"          # an HSSD eval house: the only houses with saved planner outputs (009)
QIDS = [f"{SCENE}:existence:0", f"{SCENE}:counting:1", f"{SCENE}:find:3"]
REPO_URL = "github.com/matus012/spatial-memory"
W, H, FPS = 1280, 720, 24
WALK_STEP = 3                # raw frames per video frame (~3x speed-up)
GIF_WALK_START = 540         # the README loop starts inside the house (the stream begins outdoors)
ORBIT_S = 15.0
CARD_S, Q_S, END_S = 4.0, 6.5, 7.0
GREEN, ORANGE, RED, GREY = (46, 160, 67), (230, 140, 20), (210, 40, 40), (150, 150, 150)
BG, FG, MUTED = (250, 250, 248), (25, 25, 25), (95, 95, 95)
FONT = font_manager.findfont("DejaVu Sans")
FONT_B = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_B if bold else FONT, size)


def canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("RGB", (W, H), BG)
    return im, ImageDraw.Draw(im)


def caption(d: ImageDraw.ImageDraw, tag: str, text: str) -> None:
    d.rectangle([0, H - 64, W, H], fill=(30, 30, 30))
    d.text((24, H - 46), tag, font=font(24, True), fill=(255, 210, 90))
    d.text((24 + d.textlength(tag, font=font(24, True)) + 14, H - 46), text, font=font(24), fill=(245, 245, 245))


# ---------------------------------------------------------------- qa extraction (once, from the 009 evidence)
def extract_qa(raw_dir: Path) -> None:
    raw = {}
    for f in sorted(glob.glob(str(raw_dir / "raw_chunk_*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r["qid"].startswith(SCENE + ":") and r["row"] in ("gt", "cg"):
                raw[(r["row"], r["qid"])] = r["raw_output"].strip()
    flags = {v: {q[0]: q[2] for q in json.loads((REPO / "results/p1b/per_question" / f"{v}.json").read_text())} for v in ("gt", "cg")}
    qs = {q["qid"]: q for q in map(json.loads, open(REPO / "data/p1b/questions_v1.jsonl", encoding="utf-8"))}
    pred, gt, _, _ = load(SCENE)
    lab = {"gt": {o.id: o.label for o in gt.objects}, "cg": {o.id: o.label for o in pred.objects}}

    def shown(row: str, qid: str) -> str:
        a = raw[(row, qid)]
        if qs[qid]["type"] == "find" and a.isdigit():
            return f"object {a} ({lab[row].get(int(a), '?').replace('_', ' ')})"
        return a.replace("_", " ")

    house = [k for k in flags["gt"] if k.startswith(SCENE + ":")]
    out = {"scene": SCENE, "source": "009 raw planner outputs (Qwen2.5-VL-72B, temperature 0) + results/p1b/per_question",
           "house_correct": {v: [sum(flags[v][k] for k in house), len(house)] for v in ("gt", "cg")}, "questions": []}
    for qid in QIDS:
        q = qs[qid]
        key = q["answer"] if q["type"] != "find" else f"object {q['answer'].removeprefix('obj_')} ({q['label'].replace('_', ' ')})"
        out["questions"].append({"qid": qid, "type": q["type"], "question": q["question"], "answer_key": key,
                                 **{f"{v}_answer": shown(v, qid) for v in ("gt", "cg")},
                                 **{f"{v}_correct": bool(flags[v][qid]) for v in ("gt", "cg")}})
    MEDIA.mkdir(exist_ok=True)
    QA.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", QA)


# ---------------------------------------------------------------- segment frames
def walk_frames(scene: str, start: int = 0):
    """(a) RGB stream left, top-down progress (GT footprints + path so far) right."""
    root = data_dir() / "hssd_streams" / scene
    files = sorted((root / "rgb").glob("*.jpg"))
    poses = np.loadtxt(root / "poses.txt").reshape(-1, 4, 4)
    assert len(files) == len(poses), f"{len(files)} frames vs {len(poses)} poses"
    _, gt, _, _ = load(scene)
    boxes = [(o.aabb_min[[0, 2]], o.aabb_max[[0, 2]]) for o in gt.objects if o.label not in STRUCTURAL]
    xy = poses[:, [0, 2], 3]
    lo = np.minimum(np.min([b[0] for b in boxes], 0), xy.min(0)) - 0.5
    hi = np.maximum(np.max([b[1] for b in boxes], 0), xy.max(0)) + 0.5
    fw, fh = 853, 640
    pw, ph = W - fw - 30, H - 64 - 60
    s = min(pw / (hi - lo)[0], ph / (hi - lo)[1])
    ox, oy = fw + 15 + (pw - s * (hi - lo)[0]) / 2, 40 + (ph - s * (hi - lo)[1]) / 2
    to_px = lambda p: (ox + s * (p[0] - lo[0]), oy + s * (p[1] - lo[1]))
    base = Image.new("RGB", (W, H), BG)
    bd = ImageDraw.Draw(base)
    for a, b in boxes:
        bd.rectangle([*to_px(a), *to_px(b)], outline=GREY, width=1)
    bd.text((fw + 15, 8), "top-down: GT objects (grey), path so far", font=font(18), fill=MUTED)
    idx = list(range(start, len(files), WALK_STEP))
    for k, i in enumerate(idx):
        im = base.copy()
        d = ImageDraw.Draw(im)
        im.paste(Image.open(files[i]).convert("RGB").resize((fw, fh), Image.BILINEAR), (0, 0))
        d.line([to_px(p) for p in xy[: i + 1]], fill=(40, 110, 200), width=2)
        c = to_px(xy[i])
        d.ellipse([c[0] - 6, c[1] - 6, c[0] + 6, c[1] + 6], fill=RED)
        d.text((12, 10), f"frame {i + 1} / {len(files)}  (x{WALK_STEP} speed)", font=font(18), fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        caption(d, "(a)", f"The agent walks HSSD house {scene} and records RGB-D frames (1,349 here).")
        yield np.asarray(im)


def orbit_frames(scene: str, n: int, size: tuple[int, int] = (W, H), with_caption: bool = True):
    """(b) CG map points coloured matched (green) / extra (red); missed GT objects as orange boxes (the 003/007 colours); camera orbits 360 deg."""
    pred, gt, sc, npz = load(scene)
    ps, gs = statuses(pred, gt, sc)
    pts, off = subsample(npz, len(pred.objects), budget=60_000)
    col = np.zeros((len(pts), 3))
    for o in pred.objects:
        st = ps[o.id][0]
        col[off[o.id]:off[o.id + 1]] = np.array(GREEN if st == "m" else RED if st == "x" else GREY) / 255
    keep = np.array([ps[o.id][0] != "i" for o in pred.objects for _ in range(off[o.id + 1] - off[o.id])], bool)
    pts, col = pts[keep], col[keep]
    P = lambda a: np.stack([a[..., 0], -a[..., 2], a[..., 1]], -1)          # y-up -> z-up (right-handed)
    q = P(pts)
    segs = []
    for o in gt.objects:
        if gs[o.id][0] != "u":
            continue
        a, b = o.aabb_min, o.aabb_max
        c = np.array([[x, y, z] for x in (a[0], b[0]) for y in (a[1], b[1]) for z in (a[2], b[2])])
        segs += [P(c[[i, j]]) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]
    n_m = sum(1 for v in ps.values() if v[0] == "m")
    n_x = sum(1 for v in ps.values() if v[0] == "x")
    n_u = sum(1 for v in gs.values() if v[0] == "u")
    dpi = 100
    pw, ph = (size[0], size[1] - (64 if with_caption else 0))
    fig = plt.figure(figsize=(pw / dpi, ph / dpi), dpi=dpi)
    fig.patch.set_facecolor(np.array(BG) / 255)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    ax.set_facecolor(np.array(BG) / 255)
    ax.scatter(q[:, 0], q[:, 1], q[:, 2], c=col, s=0.6, linewidths=0, depthshade=False)
    ax.add_collection3d(Line3DCollection(segs, colors=[np.array(ORANGE) / 255], linewidths=0.8))
    lo, hi = q.min(0), q.max(0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo, zoom=1.12 if with_caption else 1.2)
    ax.set_axis_off()
    for k in range(n):
        ax.view_init(elev=38, azim=-60 + 360 * k / n)
        fig.canvas.draw()
        im = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        if with_caption:
            full = Image.new("RGB", size, BG)
            full.paste(im, (0, 0))
            d = ImageDraw.Draw(full)
            y = 14
            for c, t in ((GREEN, f"matched to a GT object ({n_m})"), (RED, f"extra: no GT match ({n_x})"), (ORANGE, f"missed GT object, box ({n_u})")):
                d.rectangle([20, y + 4, 38, y + 22], fill=c)
                d.text((48, y), t, font=font(20), fill=FG)
                y += 30
            d.text((20, y + 6), f"object F1 @ IoU 0.25 = {sc.f1:.3f}", font=font(20), fill=MUTED)
            caption(d, "(b)", "The ConceptGraphs 3D object map of that house, scored against simulator ground truth.")
            im = full
        yield np.asarray(im)
    plt.close(fig)


def text_card(lines: list[tuple[str, int, bool, tuple]], tag: str | None = None, cap: str | None = None, y0: int = 110) -> np.ndarray:
    im, d = canvas()
    y = y0
    for t, size, bold, colr in lines:
        if t:
            d.text((80, y), t, font=font(size, bold), fill=colr)
        y += int(size * 1.55)
    if tag:
        caption(d, tag, cap or "")
    return np.asarray(im)


def question_cards(qa: dict) -> list[np.ndarray]:
    gc, cc = qa["house_correct"]["gt"], qa["house_correct"]["cg"]
    out = []
    for i, q in enumerate(qa["questions"], 1):
        mark = lambda ok: ("✓ correct", GREEN) if ok else ("✗ wrong", RED)
        g, c = mark(q["gt_correct"]), mark(q["cg_correct"])
        im, d = canvas()
        d.text((80, 70), f"Question {i} of 3  ·  {q['type']}", font=font(24), fill=MUTED)
        size = 40
        while d.textlength(q["question"], font=font(size, True)) > W - 160:
            size -= 2
        d.text((80, 112), q["question"], font=font(size, True), fill=FG)
        d.text((80, 190), f"answer key:  {q['answer_key']}", font=font(28), fill=FG)
        for y, who, ans, (m, mc) in ((280, "planner on the GT map", q["gt_answer"], g), (360, "planner on the ConceptGraphs map", q["cg_answer"], c)):
            d.text((80, y), who, font=font(26), fill=MUTED)
            d.text((640, y), ans, font=font(30, True), fill=FG)
            d.text((1000, y), m, font=font(30, True), fill=mc)
        d.text((80, 470), f"This house, all {gc[1]} questions: GT map {gc[0]}/{gc[1]} correct, ConceptGraphs map {cc[0]}/{cc[1]}.", font=font(24), fill=FG)
        d.text((80, 512), "Planner: Qwen2.5-VL-72B, temperature 0. Saved outputs from task 009, no new runs.", font=font(20), fill=MUTED)
        caption(d, "(c)", "The planner reads the map as text and answers templated questions.")
        out.append(np.asarray(im))
    return out


def closing_card() -> np.ndarray:
    im, d = canvas()
    d.text((80, 80), "Planner accuracy, 479 questions on 30 HSSD houses", font=font(34, True), fill=FG)
    for y, t, v, c, b in ((160, "perfect (GT) map", "85.4 %", GREEN, True),
                          (215, "real maps (ConceptGraphs / OVO / ours v2)", "29.2 / 33.6 / 28.6 %", RED, True),
                          (270, "8 photos, no map (find not asked, scored wrong)", "24.0 %", MUTED, False)):
        d.text((80, y), t, font=font(28, b), fill=c)
        d.text((860, y), v, font=font(28, b), fill=c)
    d.text((80, 350), "Real maps miss and mis-name objects; the errors compound. Box precision barely matters.", font=font(24), fill=FG)
    d.text((80, 420), REPO_URL, font=font(30, True), fill=(40, 110, 200))
    d.text((80, 470), "HSSD scenes: CC BY-NC 4.0 (hssd/hssd-hab). Research use only.", font=font(18), fill=MUTED)
    caption(d, "(d)", "The map makes the questions answerable, and real maps lose most of that.")
    return np.asarray(im)


def title_card() -> np.ndarray:
    return text_card([
        ("Spatial memory for a VLM planner", 48, True, FG),
        ("", 12, False, FG),
        ("An agent walks a simulated house (HSSD) and builds a 3D object map.", 28, False, FG),
        ("We score the map exactly against simulator ground truth,", 28, False, FG),
        ("then ask a VLM planner questions that only the map can answer.", 28, False, FG),
    ], y0=200)


# ---------------------------------------------------------------- writers
def write_mp4(path: Path, crf: int) -> None:
    import imageio_ffmpeg
    qa = json.loads(QA.read_text(encoding="utf-8"))
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = 0

    def put(fr: np.ndarray, reps: int = 1) -> None:
        nonlocal n
        b = np.ascontiguousarray(fr, dtype=np.uint8).tobytes()
        for _ in range(reps):
            p.stdin.write(b)
            n += 1

    put(title_card(), int(CARD_S * FPS))
    for fr in walk_frames(SCENE):
        put(fr)
    for fr in orbit_frames(SCENE, int(ORBIT_S * FPS)):
        put(fr)
    for fr in question_cards(qa):
        put(fr, int(Q_S * FPS))
    put(closing_card(), int(END_S * FPS))
    p.stdin.close()
    assert p.wait() == 0, "ffmpeg failed"
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB, {n} frames, {n / FPS:.1f} s)")


def write_gif(path: Path, n: int = 96, fps: int = 8) -> None:
    hw, hh = 360, 270
    walk = walk_frames(SCENE, start=GIF_WALK_START)
    orbit = orbit_frames(SCENE, n, size=(hw, hh), with_caption=False)
    frames = []
    for k in range(n):
        wf = next(walk)
        left = Image.fromarray(wf).crop((0, 0, 853, 640)).resize((hw, hh), Image.BILINEAR)
        im = Image.new("RGB", (2 * hw, hh + 30), BG)
        im.paste(left, (0, 0))
        im.paste(Image.fromarray(next(orbit)), (hw, 0))
        d = ImageDraw.Draw(im)
        d.text((8, hh + 6), "agent's RGB stream (HSSD)", font=font(15), fill=FG)
        x = hw + 8
        for t, c in (("CG map: ", FG), ("matched ", GREEN), ("extra ", RED), ("missed", ORANGE)):
            d.text((x, hh + 6), t, font=font(15, c != FG), fill=c)
            x += d.textlength(t, font=font(15, c != FG))
        frames.append(im)
    # one shared palette built from sample frames + the key colours, so red / green / orange survive quantisation
    swatch = Image.new("RGB", (2 * hw, 60), BG)
    for i, c in enumerate((GREEN, ORANGE, RED, FG, MUTED, GREY, (255, 255, 255))):
        swatch.paste(c, (i * 60, 0, i * 60 + 60, 60))
    sample = [frames[i] for i in range(0, n, max(n // 8, 1))] + [swatch] * 4
    mosaic = Image.new("RGB", (2 * hw, sum(f.height for f in sample)))
    y = 0
    for f in sample:
        mosaic.paste(f, (0, y))
        y += f.height
    pal = mosaic.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    frames = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB, {n} frames, {n / fps:.1f} s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-qa", type=Path, help="009 raw-output dir; rewrites media/114_demo_qa.json and exits")
    ap.add_argument("--crf", type=int, default=30)
    ap.add_argument("--only", choices=["mp4", "gif"])
    a = ap.parse_args()
    if a.extract_qa:
        return extract_qa(a.extract_qa)
    MEDIA.mkdir(exist_ok=True)
    if a.only != "gif":
        write_mp4(MEDIA / "114_demo.mp4", a.crf)
    if a.only != "mp4":
        write_gif(MEDIA / "114_demo.gif")


if __name__ == "__main__":
    main()
