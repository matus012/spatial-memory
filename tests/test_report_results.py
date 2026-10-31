"""RESULTS.md generator: runs, is byte-deterministic, and agrees with the numbers in the 005 / 006 reports."""
from __future__ import annotations

from spatialmem.report import results


def test_render_is_deterministic() -> None:
    assert results.render() == results.render()
    assert results.render().encode("utf-8") == results.render().encode("utf-8")


def test_numbers_agree_with_reports() -> None:
    t = results.render()
    # 005 report: HSSD eval 30 CG / OVO; 006 report: Replica 8 and HSSD dev 5
    for needle in ("0.341 | 0.500 | 0.406 | 0.226 | 0.332 | 0.269 | 0.409",
                   "0.253 | 0.405 | 0.312 | 0.175 | 0.279 | 0.215 | 0.523",
                   "| 15.80 | 37.83 | 30.91 | 16.21 |", "| 26.75 | 39.82 | 61.25 | 28.72 |",
                   "| 14.65 | 34.47 | 27.45 | 13.96 |", "| 13.03 | 27.82 |",
                   "0.341 | 0.392 | 0.365 | 0.224 | 0.257 | 0.239 | 0.454",
                   "| OVO | Replica 8 (n=8) | 0.087 [0.068–0.098] | 12.5 (13.8) |",
                   "| CG | HSSD eval 30 (n=30) | 7.28 [3.46–9.36] | 17.9 (17.9) |"):
        assert needle in t, needle
    assert "pending" in t or "ours v2 (attempt, K2 fired)" in t


def test_p1b_section() -> None:
    t = results.render()
    head, p1b = t.split("# P1b results\n")
    assert head.startswith("# P1a results") and "P1b" not in head
    for needle in ("| gt72b v1.2 (gate row) | label+dist | 85.4 | 100.0 | 84.0 | 76.7 | 80.8 |",
                   "| img (72B, 8 frames; 009) | label+dist | 24.0 |",
                   "| position | 1.00 m | 54.1 [53.7–54.3] |",
                   "**Pre-registered reading (all 30 houses): fails**", "**Reading (subset): partial**",
                   "| cg | 18.4 | 29.2 | 10.9 |",
                   "| ovo | miss | miss − label | 8.4 |", "| cg | no single dominant error | label − miss | 0.2 |",
                   "`ovo__miss`, house 107734158_175999998", "P1B_PROTOCOL.md", "P1B_SWEEP.md", "P1B_REPAIR.md", "Deviations 14"):
        assert needle in p1b, needle


def test_p1b_figs_deterministic(tmp_path) -> None:
    from spatialmem.report import figs
    for fn, name in ((figs.p1b_fig2_how, "a.png"), (figs.p1b_fig3_proof, "b.png")):
        fn(tmp_path / name)
        fn(tmp_path / ("2" + name))
        assert (tmp_path / name).read_bytes() == (tmp_path / ("2" + name)).read_bytes() and (tmp_path / name).stat().st_size > 5000
    if (figs.ROOT / "data" / "hssd_runs" / "all" / figs.HOUSE / "gt.json").exists():
        assert figs.p1b_fig1_what(tmp_path / "c.png") and figs.p1b_fig1_what(tmp_path / "c2.png")
        assert (tmp_path / "c.png").read_bytes() == (tmp_path / "c2.png").read_bytes()
