# results/scores manifest

Inputs of `uv run python -m spatialmem.report.results` (-> `RESULTS.md`). Numbers only; no dataset content.
Evidence root: the project's run archive (outside the repo, not published). Score JSONs carry no commit
field; the scorer is `spatialmem.eval.score_baselines` (CONTRACT 2026-10-27e labels), the config / commit column is what the
scored predictions came from.

| file | source path (under evidence root) | sha256 | what it scored |
|---|---|---|---|
| scores_ovo.json | 2026-10-28-006/frozen/rescore_l2/scores_ovo.json (2026-10-29-006 absent; same numbers as 2026-10-28-005/scores) | 0ba0dc923e81d6ffbef4432591577f4464daaca0c484f9a69cad94be15104863 | OVO `tberriel/OVO@e0d16f7`, slam_module vanilla (GT poses); Replica 8, HSSD eval 30, HSSD dev 5 |
| scores_cg.json | 2026-10-28-006/frozen/rescore_l2/scores_cg.json | d89871e0675074bb9a707dc2065aaf1354f6b354179cd6d6ad66dfd29176bea2 | ConceptGraphs `93277a0` + `hpc/cg_patches/01..03` (02 = world-frame poses); same three sets |
| scores_ours_v2.json | 2026-10-28-006/frozen/rescore_l2/scores_ours_v2.json (laptop re-score, 430/430 fields = worker's) | 6fd163658dc7af09392d531813ca077f4022dc3fb5ef1999cddb4d454b2cfef9 | ours v2, `configs/ours_v2.yaml` frozen at 8eed799 (config sha256 176ee20f...), PERUN snapshot 6bdd821; Replica 8 + HSSD dev 5 |
| scores_ours_v2_hssd_eval.json | 2026-10-29-007/scores/scores_ours_v2.json | 430204edd0112da0899c598dc6f39a338f258133f917c6eff4c50c7425fff2f5 | ours v2, `configs/ours_v2.yaml` frozen at 8eed799 (file sha256 13551629..., config sha256 176ee20f...), PERUN snapshot a217bda; HSSD eval 30 (counted once, task 007; its Replica 8 / dev splits equal `scores_ours_v2.json`, which wins on merge) |
| scores_ours_v1.json | 2026-10-28-004/frozen/scores/scores_ours_v1.json | d89916f5f09d30e0fc4bf028daafcea4c349ba1312eb8dd5122295abe2642ae3 | ours v1 on lpc, `configs/ours_v1.yaml` frozen at c4af463 (config sha256 aa3d3f5d...), repo head 046208c; Replica 8 + HSSD dev 5 |
| scores_ours_v1ref_perun.json | 2026-10-28-006/frozen/scores/scores_ours_v1ref_perun.json | 42f92ca018c62281d15a538c7c99a1d66806305b67cf45b6ca59ea62b666b379 | ours v1 settings on the PERUN caches (same caches as v2), `configs/ours_v1.yaml`; Replica 8, HSSD dev 5, Replica-dev 10 |
| ours_v2_r8_build_runtime.csv | 2026-10-28-006/frozen/r8_build_runtime.csv | 92842bd4c418f34c141efb8b9a0988edef03d1c697c9df90aad0559f1b7ad3ac | ours v2 build phases A / B per Replica scene (PERUN H200) |
| runtime_ovo.csv | `data/baselines/ovo/runtime.csv` (repo, git-ignored) | df6af09cec15f2add3a07084e6c12b46c10e30634cadc020e8cb85d398c47c6b | OVO per-scene wall / s per frame / VRAM, 43 scenes |
| runtime_cg.csv | `data/baselines/cg/runtime.csv` (repo, git-ignored) | e73a5b74888a8bb2970a6a92b6528e41060000fba954b3d030e8519737f3d0f7 | CG per-scene detection + mapping wall, 43 scenes |
| ovo_own_eval_pooled_8scenes.log | 2026-10-27-003/ovo/eval_logs/pooled_8scenes.log (command: pooled_8scenes.COMMAND.md, sha256 0f4d883d6b647db459bcd02eaefdc566616bc33092b04b0cdb546c91e41f35dc) | f144828fb905b991366605ec8023becb55c606863ebd7a119aa7298c352bd6b0 | OVO's own `run_eval.py` on its Replica output: "mIoU: 27.50%; mAcc: 40.20%", "f-mIoU: 61.40%" (ran on PERUN login node, CPU) |

The runtime CSVs are copied here (they are small) so a clean checkout can regenerate `RESULTS.md` without `data/`.
