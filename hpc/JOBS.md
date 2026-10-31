# PERUN jobs (114)

GPU-h = job wall between the `===` start / done epoch stamps in `runs/logs/<name>_<id>.out` (queue and container
start excluded); for jobs that died before `done`, the last write of their `.err`. 1x H200, gpu_short, 8 cpus each.

| date | job | what | wall s | GPU-h | outcome |
|---|---|---|---|---|---|
| 2026-10-27 | 135703 g1_probe | G1 probe | ~5 | 0.001 | EGL on H200 OK; scene load aborted (assets still downloading) |
| 2026-10-27 | 135710 g1_probe2 | G1 probe | 11 | 0.003 | habitat_sim 0.3.3, Renderer NVIDIA H200 (G1 PASS) |
| 2026-10-27 | 135711 g0_render | gate-0 100 frames, dev 102816627 | 19 | 0.005 | G4 0.68 — GT AABB frame wrong (diagnosed by 135714) |
| 2026-10-27 | 135714 dbg_bb | AABB frame diagnosis | 6 | 0.002 | root node is COM-centred, instance scale on child node |
| 2026-10-27 | 135715 g0_render2 | gate-0 rerun | 13 | 0.004 | G2 PASS, G3 0 collisions, G4 median 1.00 |
| 2026-10-27 | 135719 hssd_dev | v1 streams, 5 dev (largest island) | 172 | 0.048 | superseded by v2 |
| 2026-10-27 | 135720 hssd_eval | v1 streams, 30 eval (largest island) | 3623 | 1.006 | 23 scenes, then crashed: output dir moved under it (operator error); superseded |
| 2026-10-27 | 135772 hssd2_dev | v2 streams, 5 dev (all islands, indoor) | 242 | 0.067 | 5/5 |
| 2026-10-27 | 135773 hssd2_ev0 | v2 streams, eval part 0/2 | 2007 | 0.558 | 15/15 |
| 2026-10-27 | 135774 hssd2_ev1 | v2 streams, eval part 1/2 | 1405 | 0.390 | 15/15 |
| | | **task 002 total** | 7503 | **2.08** | budget 6 H200-h |

Image build (the login node, no job): 3 builds, ~4 min each (`build/build_attempt1.log` pip-less venv, attempt 2 without
trimesh, final `habitat.sif` sha256 a88af2ac97e9d875c144b2f57d5172d53b7de78b2625177d90400f691dd411cc). HSSD download
(the login node): 3.3 GB, 15,044 asset files + metadata.

## Task 003 — baselines (OVO-mapping, ConceptGraphs)

GPU-h as above (1x H200 per job). CG detection ran 4 scenes per GPU (GPU-bound at ~100 %); CG mapping never creates a
CUDA context and ran on cpu_short (0 GPU-h, listed for completeness).

| date | job | what | wall s | GPU-h | outcome |
|---|---|---|---|---|---|
| 2026-10-27 | 135889 ovo_g1 | OVO G1 probe + SigLIP text embeddings | 95 | 0.026 | cuda True, SigLIP + SAM2 on cuda:0 (G1 PASS) |
| 2026-10-27 | 135891 ovo_rep0 | OVO office0 | 191 | 0.053 | failed in segment: protocol meshes not linked |
| 2026-10-27 | 135894, 135896 ovo_rep0b | OVO office0 reruns | 0 | 0 | cancelled while pending |
| 2026-10-27 | 135897 ovo_rep0b | OVO office0 rerun | 181 | 0.050 | ok |
| 2026-10-27 | 135908 ovo_repA | OVO Replica office1-3 | 669 | 0.186 | 3/3 |
| 2026-10-27 | 135909 ovo_repB | OVO Replica office4, room0-2 | 850 | 0.236 | 4/4 |
| 2026-10-27 | 135945, 135946 ovo_hsA/B | OVO HSSD, first try | ~240 | ~0.07 | stream symlinks unreadable in container (no done stamp; estimate) |
| 2026-10-27 | 135949 ovo_hsA | OVO HSSD 18 scenes | 2480 | 0.689 | 18/18 |
| 2026-10-27 | 135950 ovo_hsB | OVO HSSD 17 scenes | 2235 | 0.621 | 17/17 |
| 2026-10-28 | 136104 ovo_textemb | OVO text embeddings under CONTRACT 2026-10-27e (2) | 78 | 0.022 | 9/1064 Replica labels changed, 0 HSSD |
| 2026-10-27 | 135900 cg_g1 | CG G1 probe + CLIP text embeddings | 51 | 0.014 | cuda True, SAM / CLIP / GroundingDINO on cuda:0 (G1 PASS) |
| 2026-10-27 | 135904 cg_room0 | CG room0 detection | 1171 | 0.325 | ok; mapping died (stale cwd after push_repo) |
| 2026-10-27 | 135957 cg_room0map | CG room0 mapping (GPU node) | 533 | 0.148 | ok (mapping found to be CPU-only) |
| 2026-10-27 | 135958 cg_rep7 | CG Replica 7 scenes detection, 4/GPU | 3769 | 1.047 | 7/7 |
| 2026-10-27 | 135971 cg_hssd_det | CG HSSD 35 scenes detection, 4/GPU | 13098 | 3.638 | 34/35 (108736824_177263559: frame 545 has no SAM mask -> patch 03) |
| 2026-10-28 | 136105 cg_g1b | CG G1 rerun + text regeneration | 29 | 0.008 | embeddings bit-identical |
| 2026-10-28 | 136075 | CG map (cpu_short) | 10 | 0 | cancelled (would have been 3rd queued job) |
| 2026-10-28 | 136106 | CG Replica mapping (cpu_short) | 1082 | 0 | 8/8 |
| 2026-10-28 | 136110 | CG HSSD mapping (cpu_short) | 11 | 0 | failed: hydra parsed numeric scene id as int |
| 2026-10-28 | 136111, 136167 | CG HSSD mapping (cpu_short) | 5551 | 0 | 10 scenes done |
| | | **task 003 so far** | | **7.13** | budget line 20, cap 25 H200-h |

From 2026-10-28 ~02:55 (CET) every `sbatch -A ${PERUN_ACCOUNT}` fails in the site job_submit Lua filter (slurmdbd refusing
connections); remaining: CG HSSD mapping 24 scenes (cpu_short) + detection rerun of 108736824_177263559 (~0.25 H200-h).
Images built on the login node (no job): `ovo.sif` (sha256 19098a97…), `cg.v1.sif` (12eccc04…), `cg.sif` (4d67f419…).

## Task 005 (finish CG HSSD)

| date | job | what | wall s | H200-h | result |
|---|---|---|---|---|---|
| 2026-10-28 | 136976 cg_h824det | CG detection rerun 108736824_177263559 (cg.sif v2, patch 03), gpu_short | 573 | 0.159 | ok, 311/311 frames, peak 18,328 MiB |
| 2026-10-28 | 136977 cg_hmapA | CG HSSD mapping 12 scenes, par 8 (cpu_short) | 1387 | 0 | 12/12 |
| 2026-10-28 | 136978 cg_hmapB | CG HSSD mapping 12 scenes, par 8 (cpu_short) | 1321 | 0 | 12/12 |
| 2026-10-28 | 136979 cg_h824map | CG mapping 108736824_177263559 (cpu_short, afterok 136976) | 510 | 0 | 1/1 |
| | | **task 005** | | **0.159** | CG 43/43; 114 total 9.37 H200-h (002 2.08 + 003 7.13 + 005 0.16) |

## Task 006 (Replica-dev streams, step 1)

Habitat image `habitat.sif` (a88af2ac…), 1x H200 per job, gpu_short. Wall s from the job's own start stamp to its done
stamp / last log write (sacct unavailable on login nodes).

| date | job | what | wall s | H200-h | result |
|---|---|---|---|---|---|
| 2026-10-28 | 136998 rdev_smoke | apartment_0, 100 frames (smoke) | 56 | 0.016 | ok; render proof 0.003 cm; grid 1.0 m -> 32 deg/frame -> grid 2.0 |
| 2026-10-28 | 137027 rdev_p0 | 5 scenes, grid 2.0 + 002 indoor test | 553 | 0.154 | cancelled: FRL partial ceilings fail the indoor test (frl_apartment_0 1 waypoint) |
| 2026-10-28 | 137028 rdev_p1 | 5 scenes, same | 553 | 0.154 | cancelled, same |
| 2026-10-28 | 137035 rdev2_p0 | 5 scenes, floor test only | 784 | 0.218 | 3/5 (apartment_0, apartment_2, frl_apartment_1); Stale file handle (repo re-pushed mid-job) |
| 2026-10-28 | 137037 rdev2_p1 | 5 scenes, floor test only | 778 | 0.216 | 3/5 (apartment_1, frl_apartment_0, frl_apartment_2); same |
| 2026-10-28 | 137044 rdev3_p0 | remaining (replica_dev_render.sbatch, repo snapshot) | 523 | 0.145 | frl_apartment_3, frl_apartment_5 |
| 2026-10-28 | 137046 rdev3_p1 | remaining, same | 519 | 0.144 | frl_apartment_4, hotel_0 |
| | | **task 006 step 1** | 3766 | **1.05** | 10/10 scenes x 2,000 frames, `runs/replica_dev/<scene>/` |

## Task 006 ("ours" v2 on PERUN, step 2; deviation 11)

Image `ours.sif` (9d5c345f…, `hpc/ours.def` = `uv sync --frozen` of the lpc `uv.lock`), built on the login node (no job).
Weights `$PROJ/models` (sha256 = THIRD_PARTY.md). Runner `hpc/slurm/ours_scenes.sbatch` (per-job repo snapshot).
Wall s from the job's start stamp to its done stamp (sacct unavailable).

| date | job | what | wall s | H200-h | result |
|---|---|---|---|---|---|
| 2026-10-28 | 137042 ours_proof | v1 c11 from scratch on 102816627 (masks -> detections -> prediction), 1 H200 | 439 | 0.122 | ran; not byte-identical to lpc (SAM masks differ in 102/213 frames; 312 objects both; F1@0.25 0.3232 vs 0.3192) |
| 2026-10-28 | 137047 ours_hdev5 | HSSD dev 5 caches, 3 encoders, par 5 | ~140 | 0.039 | cancelled: v2 VRAM assert on reserved (8.21 GB on H200) -> assert on allocated |
| 2026-10-28 | 137049 ours_hdev5 | HSSD dev 5 masks + desc (siglip2_base, so400m, CLIP-H), par 5 | 418 | 0.116 | 5/5 x 3 encoders |
| 2026-10-28 | 137057 ours_smoke | association smoke 104348253_171513237, so400m + CLIP (cpu_short) | 68 | 0 | ok (so400m prediction overwritten: one encoder per tag since 920576b) |
| 2026-10-28 | 137070 ours_smoke | association smoke 104348253_171513237, so400m (cpu_short) | 24 | 0 | ok |
| 2026-10-28 | 137074 ours_v1ref | v1 c11 reference on the PERUN HSSD dev 5 caches, association only, par 5 (cpu_short) | 329 | 0 | 5/5 cache hits; 102816627 = proof 137042 (sha256 8a53ac62…) |
| | | **task 006 step 2 so far** | | **0.28** | |

### Task 006 step 1, re-render (deviation 13: PTex image + smoother path)

`habitat_ptex.sif` (habitat-sim 0.2.4 + fix 1: compile the PTex geometry shader) built on the login node, no job: attempt 0
45e9f371… (PTex shader link failure on H200), fix 1 6278f434… (used). Plans on `habitat.sif`, renders on `habitat_ptex.sif`;
all jobs run from per-job repo snapshots (`hpc/push_snapshot.sh`, `hpc/slurm/replica_dev_render.sbatch`).

| date | job | what | wall s | H200-h | result |
|---|---|---|---|---|---|
| 2026-10-29 | 137123 rdev_ptex_smoke | hotel_0 PTex smoke, image attempt 0 | 20 | 0.006 | PTex shader link failed (geometry shader never compiled) -> fix 1 |
| 2026-10-29 | 137131 rdev_ptex_smoke2 | same, fixed image | 10 | 0.003 | 0.2.4 navmesh recompute failed on the PTex stage |
| 2026-10-29 | 137135 rdev_ptex_smoke3 | same, navmesh from the vertex-colour asset | 9 | 0.003 | 0.2.4 "Could not build Detour navmesh" -> plan on 0.3.3 |
| 2026-10-29 | 137141 rdev_plan | plan 10 scenes (habitat.sif) | 146 | 0.041 | 9/10; hotel_0 no navmesh at 0.5 m -> radius fallback |
| 2026-10-29 | 137142 rdev_ptex_smoke4 | smoke afterok 137141 | 0 | 0 | cancelled while pending (dependency failed) |
| 2026-10-29 | 137163 rdev_plan2 | plan 10 scenes | 143 | 0.040 | 10/10 |
| 2026-10-29 | 137164 rdev_ptex_smoke5 | hotel_0 PTex, 200 frames | 29 | 0.008 | ok (PTex textures rendered) |
| 2026-10-29 | 137182 rdevA_p0 | PTex render 5 scenes | 1380 | 0.383 | 5/5 |
| 2026-10-29 | 137183 rdevA_p1 | PTex render 5 scenes | 1322 | 0.367 | 5/5 |
| | | **re-render** | 3059 | **0.85** | 10/10 x 2,000 frames -> `runs/replica_dev/` (v1 kept as `runs/replica_dev_v1_vertexcolour/`); step 1 total 1.90 H200-h |

## Task 006 ("ours" v2 Replica-dev caches + tuning, steps 2-3)

Snapshot `cache/tmp/repo_b52ad25` (hpc/push_snapshot.sh), runner `hpc/slurm/ours_tasks.sbatch`, image `ours.sif` (9d5c345f…).
Inputs: `runs/replica_dev/<scene>/` PTex streams (accepted 2026-10-29). Caches `runs/ours/cache/ours/replica_dev/` (17 GB).

| date | job | what | wall s | H200-h | result |
|---|---|---|---|---|---|
| 2026-10-29 | 137361 ours_rdev_smoke | Replica-dev apartment_0: masks + desc x 3 encoders, 1 H200 | 490 | 0.136 | ok; projection 10 scenes <= 1.4 H200-h |
| 2026-10-29 | 137368 ours_rdev_p0 | Replica-dev 5 scenes, par 3 on 1 H200 | 1661 | 0.461 | 5/5 |
| 2026-10-29 | 137369 ours_rdev_p1 | Replica-dev 4 scenes, par 3 on 1 H200 | 1428 | 0.397 | 4/4 |
| 2026-10-29 | 137408 ours_r1 | v1 ref (Replica-dev) + round 1 (8 configs x 15 scenes) association, par 32 (cpu_short) | 945 | 0 | 130/130 |
| 2026-10-29 | 137409 ours_r1score | round 1 scoring, serial (cpu_short) | ~1926 | 0 | 6/9 tags, cancelled (superseded by 137449) |
| 2026-10-29 | 137449 ours_r1score_b | round 1 scoring, CLIP-H 4 tags in parallel (cpu_short) | 327 | 0 | 4/4 (cliph_blend_mean row identical to 137409's) |
| 2026-10-29 | 137468 ours_r2 | round 2 (2 configs, tau 0.55) association, par 30 (cpu_short) | 251 | 0 | 30/30 |
| 2026-10-29 | 137469 ours_r2score | round 2 scoring (cpu_short) | 316 | 0 | 2/2 |
| 2026-10-29 | 137485 ours_r2c | round 2 extra config CLIP-H blend vote tau 0.55 (objective addition 0231249), association par 15 (cpu_short) | 224 | 0 | 15/15 |
| 2026-10-29 | 137486 ours_r2cscore | its scoring (cpu_short) | 304 | 0 | 1/1; winner unchanged |
| | | **this block** | | **0.994** | 006 "ours" total 1.27 H200-h (with step 2 0.28) |

## Task 006 (frozen v2 counted run, step 5-6; config 8eed799, snapshot cache/tmp/repo_6bdd821)

| date | job | what | wall s | H200-h | result |
|---|---|---|---|---|---|
| 2026-10-29 | 137498 ours_r8_smoke | Replica 8 room0: masks + desc siglip2_base + CLIP-H, 1 H200 | 647 | 0.180 | ok |
| 2026-10-29 | 137499 ours_reduct | reduction: v1 path (spatialmem.ours.run, ours_v1.yaml) on PERUN HSSD dev 5 caches (cpu_short) | 635 | 0 | 5/5 byte-identical to ours_v1ref_perun (v2 path) |
| 2026-10-29 | 137510 ours_r8_p0 | Replica 8: room1, room2, office0, office1; par 3 on 1 H200 | 1259 | 0.350 | 4/4 |
| 2026-10-29 | 137511 ours_r8_p1 | Replica 8: office2-4; par 3 on 1 H200 | 1131 | 0.314 | 3/3 |
| 2026-10-29 | 137542 ours_frozen | ours_v2 + ours_v2_abl_mean (R8 + HSSD dev 5) + v1 ref R8 (v1 path), association once (cpu_short) | 2255 | 0 | 27/27 |
| | | **this block** | | **0.844** | 006 "ours" total 2.11 H200-h |

## Task 007 (P1a close-out, step 1: frozen ours v2 on HSSD eval 30, attempt (K2 fired))

Config `configs/ours_v2.yaml` @ 8eed799 (file sha256 13551629… on PERUN, resolved 176ee20f…), snapshot
`cache/tmp/repo_a217bda` (LF export), image `ours.sif` (9d5c345f…). Caches `runs/ours/cache/ours/hssd/` (33 GB, 35 scenes).

| date | job | what | wall s | H200-h | result |
|---|---|---|---|---|---|
| 2026-10-29 | 137603 ours_e30_smoke | eval scene 102344457 (1521 frames): masks + siglip2_base + CLIP-H desc, 1 H200 | 235 | 0.065 | ok; projection 30 scenes ~1.8 H200-h |
| 2026-10-29 | 137606 ours_e30_p0 | 14 eval scenes (26,448 frames), par 4 on 1 H200 | 3348 | 0.930 | 14/14 |
| 2026-10-29 | 137607 ours_e30_p1 | 15 eval scenes (26,527 frames), par 4 on 1 H200 | 3261 | 0.906 | 15/15 |
| 2026-10-29 | 137652 ours_e30_assoc | frozen v2 (no --set) association on the 30, par 30 (cpu_short) | 1576 | 0 | 30/30 |
| | | **task 007 step 1** | | **1.90** | |

## Task 008 (P1b planner env: vLLM + Qwen2.5-VL on PERUN; dummy smokes only, no question-set prompt)

Image `vllm.sif` = `docker://vllm/vllm-openai:v0.10.1.1` (sha256 c35531a7…; vllm 0.10.1.1, torch 2.7.1+cu128, transformers
4.55.2), pulled on the login node (no job). Weights `$PROJ/models/Qwen2.5-VL-7B-Instruct` (rev cc594898…, 16.6 GB) and
`-72B-Instruct` (rev 89c86200…, 146.8 GB), downloaded on the login node (no job). Runner `hpc/p1b/vllm_run.sbatch` from per-job
snapshots. Smoke inputs synthetic (`$PROJ/runs/p1b/smoke/`). H200-h = wall between the `===` stamps x GPUs.

| date | job | what | wall s | H200-h | result |
|---|---|---|---|---|---|
| 2026-10-29 | 137704 p1b_probe | GPU node probe (nvidia-smi, internet), 1 H200 | ~1 | 0.000 | H200, driver 580.159.04 (CUDA 13.0); GPU nodes reach HF + PyPI |
| 2026-10-29 | 137710 p1b_s_text7b | 7B tp 1, 3 synthetic text prompts (~24.1k tokens) | 132 | 0.037 | ok; load 116 s, generate 2.8 s (3 prompts batched) |
| 2026-10-29 | 137711 p1b_s_img7b | 7B tp 1, 2 prompts x 8 frames of 102816627 | 133 | 0.037 | ok; 391 tokens/frame, 3,173-token prompts; load 117 s, generate 2.6 s |
| 2026-10-29 | 137734 p1b_s_text72b | 72B tp 2 | ~1 | 0.001 | failed in apptainer `--env` (commas of CUDA_VISIBLE_DEVICES=0,1) -> APPTAINERENV_ |
| 2026-10-29 | 137736 p1b_s_text72b | 72B tp 2, 2 synthetic text prompts (~24.1k tokens) | 445 | 0.247 | ok; load 421 s (weights 313 s), generate 9.2 s |
| | | **task 008 env** | 712 | **0.32** | budget 0.6 H200-h for probes |

## Task 008 (P1b planner runs on the frozen question set; snapshot repo_78868af / repo_b07b980)

Prompts `$PROJ/runs/p1b/prompts/` (sha256 in `configs/p1b_v1.yaml`), outputs `$PROJ/runs/p1b/raw/`. H200-h = wall between the
`===` stamps x GPUs.

| date | job | what | wall s | H200-h | result |
|---|---|---|---|---|---|
| 2026-10-29 | 137749 p1b_gt7b | 7B tp 1, GT map v1.1, 479 prompts | 127 | 0.035 | 479/479; 49.5 % (pre-fix) |
| 2026-10-29 | 137750 p1b_img7b | 7B tp 1, 8 frames no map, 359 prompts | 136 | 0.038 | 359/359; 24.6 % (find not asked) |
| 2026-10-29 | 137751 p1b_gt72b | 72B tp 2, GT map v1.1, 479 prompts | 373 | 0.207 | 479/479; 67.8 % (pre-fix) |
| 2026-10-29 | 137756 p1b_gt7b_v12 | 7B tp 1, GT map v1.2, 479 prompts | 80 | 0.022 | 479/479; 62.8 % (G4) |
| 2026-10-29 | 137757 p1b_gt72b_v12 | 72B tp 2, GT map v1.2, 479 prompts | 142 | 0.079 | 479/479; 85.4 % (escalation) |
| | | **task 008 runs** | | **0.38** | task 008 total 0.70 (env 0.32) of 10 |

## Task 009 (P1b sweep: 72B planner on 64 map variants + images-only; docs/P1B_SWEEP.md; snapshot repo_f75d994)

Prompts `$PROJ/runs/p1b/prompts/{prompts_gt,chunk_0..3}.jsonl` (sha256 in `data/p1b/sweep/MANIFEST.json`), outputs
`$PROJ/runs/p1b/raw/`. All jobs Qwen2.5-VL-72B tp 2 (2x H200). H200-h = wall between the `===` stamps x 2.

| date | job | what | wall s | H200-h | result |
|---|---|---|---|---|---|
| 2026-10-29 | 137778 p1b_009_g2 | G2: `gt` re-run, 479 prompts | 143 | 0.079 | 85.4 %, raw outputs = 008 on 479/479 |
| 2026-10-29 | 137784 p1b_009_c0 | chunk 0: 17 variants (gt first), 8,143 prompts | 906 | 0.503 | 8143/8143 |
| 2026-10-29 | 137785 p1b_009_c1 | chunk 1: 16 variants, 7,664 prompts | 864 | 0.480 | 7664/7664 |
| 2026-10-29 | 137790 p1b_009_c2 | chunk 2: 16 variants, 7,544 prompts | 789 | 0.438 | 7544/7544 |
| 2026-10-29 | 137791 p1b_009_c3 | chunk 3: 16 variants, 7,664 prompts | 1054 | 0.586 | 7664/7664 |
| | | **task 009** | | **2.09** | cap 15 |

## Task 010 (P1b repair attribution: 72B planner on 21 repaired real maps; docs/P1B_REPAIR.md; snapshot repo_4273172)

Prompts `$PROJ/runs/p1b/prompts/repair_chunk_0.jsonl` (sha256 d475bb98…, = lpc `data/p1b/repair/chunk_0.jsonl`), output
`$PROJ/runs/p1b/raw/raw_repair_chunk_0.jsonl`. H200-h = wall between the `===` stamps x 2.

| date | job | what | wall s | H200-h | result |
|---|---|---|---|---|---|
| 2026-10-29 | 137834 p1b_010_rep | 21 variants x 479 = 10,059 prompts, 72B tp 2 | 1781 | 0.989 | 10059/10059 |
| | | **task 010** | | **0.99** | cap 4 |
