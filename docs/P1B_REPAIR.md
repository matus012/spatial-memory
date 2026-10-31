# P1b repair attribution v1 — which real-map error costs the planner most? (task 010, claim-grade)

Frozen before any 010 number. Everything not named here is as in 009 (`docs/P1B_SWEEP.md`, incl. the deviation-16 find
rule "label A + ≤ 1.0 m"): questions (sha256 d0d9ee72…), map text v1.2, prompt, parser, scorer, planner
Qwen2.5-VL-72B @ 89c86200… tp 2, temperature 0, max_tokens 32, seed 114, overflow rule (30,000 tokens, next_to farthest
first). 009's reading stands as reported; nothing here refits it. A change is a dated entry at the bottom and a row in
`DEVIATIONS.md`.

## 1. Maps
For each real map M ∈ {`cg`, `ovo`, `ours_v2`} (P1a HSSD-eval predictions, `score_baselines.load_pred(m, "hssd", s, True)`)
and each of the 30 eval houses, objects with a STRUCTURAL label are removed first (as `from_prediction` does), on both
sides. **Matching** = the P1a scorer's Hungarian match on 3D AABB IoU with a pair kept iff IoU ≥ 0.25
(`spatialmem.eval.objects.match` on the same IoU matrix `score_objects` builds), unchanged.

| repair | change to M (per house) |
|---|---|
| `label` | every matched predicted object takes its GT object's canonical label |
| `extra` | every unmatched predicted object is removed |
| `miss` | every unmatched GT object is added (GT box + GT label); new ids = max predicted id + 1, + 2, … in GT id order |
| `box` | every matched predicted object's box is replaced by its GT object's box |
| `label+extra` | `label` then `extra` |
| `label+extra+miss` | `label`, `extra`, then `miss` |
| `all` | `label`, `extra`, `miss`, then `box` (≈ the GT map up to ids) |

- Matching is computed once on the unrepaired M; every repair uses that matching (repairs never re-match).
- **Relations are always recomputed** from the repaired boxes and labels by `spatialmem.gt.relations.compute_relations(
  objects, up=1)` (RELATIONS v2). For the unrepaired M this reproduces its stored relations exactly (all three mappers
  used `compute_relations`; unit test), so **acc(M) = 009's row for M** (same prompt files; 009 G2 showed identical
  outputs on re-run) and M itself is not re-run.
- Map ids: predicted ids kept; GT-added ids as above.
- 3 maps × 7 repairs = **21 repaired variants**, named `<m>__<repair>` (e.g. `cg__label+extra`).

## 2. Features and overflow
Per variant, pooled over 30 houses: object F1 @ 0.25, label acc, relation F1 by the P1a scorer (as 009 §4, before the
overflow rule). Sanity: `<m>__all` features = 1.000 / 1.000 / 1.000 (unit test on all 30 houses). The 009 overflow rule
is applied to every variant and logged (houses trimmed, next_to removed, GT-relevant hits).

## 3. Readings (pre-registered, written before any 010 number; never revised by the result)
- **G2 sanity:** for each M, acc(`M__all`) within ± 3.0 points of 85.4 %. If not → instrument finding: diagnose before
  any attribution is read.
- **Attribution:** for each M, gain_i = acc(M + R_i) − acc(M) in points, for the four single repairs
  i ∈ {label, extra, miss, box}. The **dominant error** of M is the single repair with the largest gain **if** it exceeds
  the second-largest gain by ≥ 5.0 points; otherwise "no single dominant error".
- The same rule is applied per question type (existence / relation / counting / find; gains in points of that type's
  accuracy) — reported as secondary, no headline weight.
- Combined repairs (`label+extra`, `label+extra+miss`, `all`) are reported as cumulative gains; no reading on them.
- Images-only (24.0 %, 009) and GT (85.4 %) are reference lines.

## 4. Outputs
`results/p1b/repair.json` (features, overflow, per-question results, accuracies, gains, readings) and
`reports/figs/010_repair.png` (bars: accuracy per repair per map, with M, GT and images-only lines), both regenerated
byte-identically by `uv run python -m spatialmem.p1b.repair_analysis` from `results/p1b/`.

## Change log
- 2026-10-29: v1 written (task 010), before any 010 number.
- 2026-10-29 (amendment, before any 010 number; DEVIATIONS.md row 17): **overflow fallback.** The 009 rule
  (remove next_to farthest-first until ≤ 30,000 tokens) cannot finish when even all next_to lines removed leave the
  house above 30,000: found only for `ovo__miss`, house 107734158_175999998 (961 objects, 4,023 relations; 80,669 tokens;
  31,090 with every next_to removed), in a scan of all 21 variants × 30 houses. Rule for that case, same for every map:
  remove all next_to lines and keep the prompt if it is ≤ 32,736 tokens (the 32,768 native context minus max_tokens
  32); only if it is still longer, remove `in` then `on` lines farthest-first until ≤ 32,736. Objects never removed.
  The case is logged and reported; it never occurs in 009's maps (their max after the rule was 29,999).
