# ours v2 — pre-registered tuning objective (task 006 step 3)

Written 2026-10-28 21:32 (CET), committed before any v2 tuning number exists. Never rewritten; later changes are appended as
dated lines below with the authority named.

## Objective
Maximise **Replica-dev pooled mIoU** (OVO protocol, our scorer: `spatialmem.eval.semantic.pooled` over the
Replica-dev scenes listed in `configs/replica_dev.yaml`), **subject to** HSSD dev 5 mean per-scene object
F1@0.25 >= **0.351** (OVO). Configs failing the constraint are not eligible.
Tie (eligible configs within <= 0.3 mIoU of the best) -> higher HSSD dev 5 label accuracy (pooled, @0.25).
The 8 protocol Replica scenes (room0-2, office0-4) are never used for selection.

## Search space (<= 12 configs; everything not listed = v1 frozen, configs/ours_v1.yaml @ c4af463)
- encoder in {siglip2_so400m, clip_vith14_laion2b}
- descriptor in {blend, crop}
- label in {mean (v1: argmax of mean top-k descriptor), vote (argmax per top-k view, majority, ties -> mean)}
- tau in {0.45, 0.55}

## Plan
- Round 1 (8 configs): encoder x descriptor x label at tau 0.45 (v1 frozen tau).
- Round 2 (<= 4 configs): tau 0.55 on the 2 best eligible round-1 configs; the remaining <= 2 chosen from round-1
  evidence, each logged here with a dated line before it runs.

## Reported beside every config
Replica-dev pooled mIoU / mAcc / f-mIoU; HSSD dev 5 mean F1@0.25, pooled F1@0.25 / @0.5, label acc, rel F1;
object__outside share; CG HSSD dev 5 (pooled F1@0.25 0.409, per-scene mean 0.416) and OVO (0.355 / 0.351).

## Dated additions
- 2026-10-29 03:58 (CET): round 2, "other" config 1 of <= 2: **CLIP-H blend vote, tau 0.55**. Why: the round-2 pair was picked
  by applying the full objective (tie rule incl.) to round 1 -> CLIP-H crop vote + crop mean. Read as "2 best by pooled
  mIoU alone", the pair would have been crop mean (11.94) + blend vote (11.85). Running blend vote at tau 0.55 covers
  both readings, so the winner cannot depend on which one was meant. Selection rule unchanged; run before freeze,
  never on the protocol scenes. Round 2 total becomes 3 of <= 4.
