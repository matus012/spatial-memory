# P1b sweep v1 — planner success vs map quality (task 009, claim-grade)

Frozen before any 009 number. Inputs fixed by 008: questions `data/p1b/questions_v1.jsonl` (sha256 d0d9ee72…), map
text / prompt / parser / scorer = `docs/P1B_PROTOCOL.md` v1.2, planner = Qwen2.5-VL-72B-Instruct @ 89c86200…, vLLM
0.10.1.1, tp 2, temperature 0, max_tokens 32, seed 114 (`configs/p1b_v1.yaml`). A change is a dated entry at the bottom
and a row in `DEVIATIONS.md`; nothing above the change log is rewritten after a 009 number exists.

## 1. Maps
A **map variant** is one map per house for all 30 HSSD eval houses. Every variant is turned into planner text by the
v1.2 serialiser (STRUCTURAL objects and their relations removed, `spatialmem.p1b.gtmap.from_prediction`).

| variant | what |
|---|---|
| `gt` | the GT map (008), re-run in 009 (§6 G2) |
| `<axis>_<level>_s<seed>` | GT map degraded along one axis (§2); 5 axes × 4 levels × seeds 0, 1, 2 = 60 variants |
| `cg`, `ovo`, `ours_v2` | the P1a HSSD-eval predictions `data/baselines/<m>/hssd/<scene>.json` (`score_baselines.load_pred`), with the relations they carry (all three = `compute_relations` on their own boxes) |
| `img` | no map: 8 frames (008 `img7b` prompts, run on the 72B); find not asked → wrong |

## 2. Degradation axes (each alone, applied to the GT map of each house; N = non-structural GT objects of the house, M = its GT relations)
Randomness: `random.Random(f"009:{axis}:{level}:{seed}:{scene}")` (one stream per house and variant; draws in the order
written below; object lists sorted by id before every draw). Level 0 of every axis = the GT map exactly (unit test).

1. **drop** p ∈ {10, 25, 50, 75} %: remove exactly round(p·N) objects drawn without replacement (`rng.sample`); every
   relation touching a removed object goes with it. Other relations unchanged.
2. **label** q ∈ {10, 25, 50, 75} %: choose exactly round(q·N) objects (`rng.sample`); each gets a label drawn uniformly
   (`rng.choice`) from the eval-30 label vocabulary (008 §2, sorted) minus its current label. Boxes and relations unchanged.
3. **position** σ ∈ {0.10, 0.25, 0.50, 1.00} m: every object's box is translated by (dx, dy, dz), each `rng.gauss(0, σ)`
   (objects in id order, x then y then z); sizes unchanged. Relations recomputed from the noisy boxes by
   `spatialmem.gt.relations.compute_relations(objects, up=1)` (RELATIONS v2; on the GT boxes this reproduces the exported
   GT relations exactly — `selfcheck_hssd`, and a 009 unit test at σ = 0).
4. **extra** e ∈ {10, 25, 50, 100} %: add exactly round(e·N) fake objects. For each: label = label of a GT object drawn
   uniformly (`rng.choice`, i.e. the house's GT label distribution); size = size of a GT object drawn uniformly among
   those with that label; height = the centre height (y) of a GT object drawn uniformly (`rng.choice`); x, z centre =
   `rng.uniform` over the house's GT AABB (union of all non-structural GT boxes) on each horizontal axis. Fake ids =
   max GT id + 1, + 2, … in creation order. Relations recomputed on all objects (GT + fake) by `compute_relations`.
5. **relations** r ∈ {25, 50, 75, 100} %: remove exactly round(r·M) relation lines drawn without replacement
   (`rng.sample` over the sorted relation list); objects unchanged.

## 3. Overflow rule (008 proposal, adopted as written)
For each house of each variant: if the v1.2 prompt with the longest question+format text of that house (system prompt +
user text, Qwen2.5-VL chat template, generation prompt included) exceeds **30,000 tokens**, `next_to` lines are removed,
the largest centre-to-centre distance (3D, map centres) first, ties by (subject id, object id), until it fits. Objects,
`on` and `in` lines are never removed. The minimal number of removals is found by bisection (token count is monotone in
removals). The same rule for every variant (GT included; the GT map never triggers it). Reported per variant and house:
next_to lines removed; and the number of relation / find questions whose **GT-relevant relation was removed** — a
question's GT-relevant relations are the GT triples that make its answer true (relation: (a, p, b) for the unique
subject a and every b of the answer label; find: (answer object, p, b) for every b of label B); a removed map line
(s, p, o) hits one if s and o are matched to its GT subject and object by the scorer's object matching at IoU 0.25
(for GT-derived variants ids coincide with GT ids).

## 4. Map-quality features (per variant, pooled over the 30 houses; our scorer, unchanged)
Object F1 @ IoU 0.25, label accuracy (over matched pairs @ 0.25), relation F1 — `spatialmem.eval.objects`
(`score_objects`, `score_relations`, STRUCTURAL excluded), pooled exactly as `score_baselines` pools HSSD eval
(sums of TP / n_pred / n_gt over houses). **Computed on the map before the overflow rule** (the rule is a prompt-budget
artefact; §5 robustness covers it). Post-overflow features are reported as a secondary column only.

## 5. Analysis (`spatialmem/p1b/sweep.py`; inputs = `results/p1b/`, regenerated byte-identically)
- **Planner accuracy** of a variant = correct / 479, per type and overall (008 scorer; images-only: 479 with find wrong).
- **Curves:** accuracy (overall and per type) vs level per axis; level 0 = the 009 `gt` re-run; mean over seeds with the
  min–max range. 008's `gt72b` 85.4 % shown as a reference line.
- **Predictor:** ordinary least squares, accuracy (overall, fraction) ~ intercept + object F1 + label acc + relation F1,
  fitted on the 61 points (60 degraded variants + `gt`). Reported: coefficients, R² on the fit set; for `cg`, `ovo`,
  `ours_v2`: predicted, actual, abs error in percentage points.
- **Pre-registered reading** (written 2026-10-29 before any 009 number; never revised by the result):
  - "**map scores predict planner success**" holds if all three real-map abs errors ≤ 5.0 points;
  - "**fails**" if any abs error > 10.0 points;
  - otherwise "**partial**".
- **Robustness:** the same predictor and reading restricted to the houses where **no** variant (any of the 64 map
  variants) triggered the overflow rule: accuracy over those houses' questions only, features pooled over those houses
  only, refitted. Reported beside the headline, never substituted for it.
- Secondary (reported, no reading): per-type predictor errors; the images-only row; the 7B is not run in 009.

## 6. Gates (task 009)
G1 this file + generator + tests committed before any 009 number (git order in the report) · G2 `gt` re-run within
± 1.0 point of 85.4 % (else stop: nondeterminism → diagnose, report) · G3 60 + 3 + 1 (+ `gt`) planner runs complete, 479
(images: 359 asked) answers each, every prompt + raw output saved · G4 analysis regenerates byte-identically from saved
outputs · G5 figures looked at, one line each; the reading stated with the numbers · G6 committed, clean tree, both
ledgers.

## Change log
- 2026-10-29: v1 written (task 009), before any 009 number.
- 2026-10-29 (amendment, before any 009 number; DEVIATIONS.md row 16): **find scoring tightened for 009
  and every later P1b scoring.** A find answer is correct iff the returned object's canonical label equals the asked label
  A **and** its centre (in the map the planner saw) is within 1.0 m of the GT answer object's AABB centre. Reason (audit
  of 008): with 1.0 m alone the reference object B lies within 1.0 m on 70 of 120 find questions, so naming B
  scored correct. Applies to every 009 row (degraded, real, gt re-run); the 008 ceiling is reported under both rules.
  The G2 bar stays "± 1.0 point of 85.4 %" read on the 008 (old) rule for the re-run; the new-rule value is reported beside it.
- 2026-10-29 (run order, before any 009 number): G2 runs first as its own job on `prompts_gt.jsonl` (row `gt`,
  output `raw_gt_g2.jsonl`); the four chunk jobs start only if it is within ± 1.0 point of 85.4 % (old find rule). Chunk 0
  also carries `gt` (same batch conditions as the sweep); that run is the level-0 / fit point of §5, and it is held to
  the same G2 bar (both reported). Variant names use the level strings of §2 literally (`drop_10_s0`, `position_0.10_s0`).
