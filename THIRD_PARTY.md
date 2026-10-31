# Third-party sources

Every dataset, model and code source used by this repo, with its licence. Datasets are never committed
or redistributed (`data/` is git-ignored); derived GT (`gt/`) is regenerable and ignored too.

## Datasets

| source | used for | licence / terms | where |
|---|---|---|---|
| Replica v1 (facebookresearch/Replica-Dataset, release v1.0) | semantic meshes `habitat/mesh_semantic.ply` + `info_semantic.json` → object GT | "Replica Dataset Research Terms": non-commercial research/education only (§1); share only with people bound by the terms (§2) | `data/replica_v1/` |
| NICE-SLAM Replica sequences (cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip) | RGB-D frames, `traj.txt` poses, protocol meshes `<scene>_mesh.ply` | derived from Replica → Replica terms | `data/nice/` |
| OVO Replica per-vertex GT (`tberriel/OVO@e0d16f7` `data/input/replica_semantic_gt/*.txt`) | protocol semantic GT | derived from Replica → Replica terms (OVO's MIT covers code only) | `data/ovo_replica_gt/` |
| HSSD (`hssd/hssd-hab`, HuggingFace; revision + per-file sha256 in `data/MANIFEST.md` / `data/MANIFEST_hssd.sha256`) — from task 002 | sim scenes (35: 5 dev + 30 eval, `configs/hssd_scenes.yaml`), object GT | CC BY-NC 4.0 | PERUN `$PROJ/data/hssd-hab/`; lpc only the rendered dev/eval streams `data/hssd_runs/`, `data/hssd_streams/` |
| Replica v1 PTex assets (`mesh.ply`, `textures/`, from the official Replica v1 release, 10 non-protocol scenes) — task 006 | Replica-dev RGB renders (deviation 13) | Replica Research Terms | PERUN `$PROJ/data/replica_v1/` |
| `habitat/sorted_faces.bin` (Habitat team, `dl.fbaipublicfiles.com/habitat/sorted_faces.zip`, 40.8 MB, sha256 `a6fbdd69…`) — task 006 | PTex loader input missing from the Replica tarball; only the `.bin` files copied, the zip's script not run | derived from Replica → Replica terms | PERUN `$PROJ/data/replica_v1/<scene>/habitat/` |

## Code

| source | used for | licence | where |
|---|---|---|---|
| OVO (`tberriel/OVO@e0d16f7`) `ovo/utils/eval_utils.py`, `data/working/configs/Replica/eval_info.yaml` | Replica protocol: 51-class map; metric functions executed as the G4 reference | MIT (`third_party/OVO_LICENSE`) | `third_party/` (vendored verbatim, plus a one-line licence header) |
| Open3D 0.20 | CPU raycasting (Embree) | MIT | PyPI |
| numpy, scipy, scikit-learn, matplotlib, plyfile, Pillow, PyYAML, tqdm | numerics, IO, plots | BSD / MIT / PSF-style permissive | PyPI (`uv.lock`) |

## Models

Per method below (ConceptGraphs, OVO, ours v1/v2). No GPL detectors are used.

## Renderer (tasks 002, 006)

| source | used for | licence | where |
|---|---|---|---|
| habitat-sim v0.3.3 (`facebookresearch/habitat-sim`, built from source, headless EGL, no bullet; `hpc/habitat.def`) | HSSD RGB-D + GT-pose streams; Replica-dev navmesh path planning | MIT | PERUN `$PROJ/habitat.sif` |
| habitat-sim v0.2.4 (same, the last release with the Replica PTex mesh path; `hpc/habitat_ptex.def`, one-line geometry-shader patch) | Replica-dev PTex RGB-D renders (deviation 13) | MIT | PERUN `$PROJ/habitat_ptex.sif` (sha256 `6278f434…`) |
| Magnum / Corrade, trimesh, numpy<2, scipy, Pillow, PyYAML, plyfile (in those images) | habitat-sim dependencies, IO | MIT / BSD | built into the images |

## Media

| source | used for | licence |
|---|---|---|
| HSSD renders of house 102815859 (Habitat-sim streams) + its ConceptGraphs map | `media/114_demo.mp4`, `media/114_demo.gif` (`scripts/make_demo_video.py`); HSSD figures in `reports/figs/` | HSSD CC BY-NC 4.0 (derived renders; attribution `hssd/hssd-hab`) |
| imageio-ffmpeg 0.7.0 (bundles an ffmpeg 7.1 binary; used as a tool, not distributed) | h264 encoding | BSD-2 (wrapper) |

## Clean-room

Public code only.

## ConceptGraphs baseline (task 003; `hpc/cg.def`, image `$PROJ/cg.sif` on PERUN, never on lpc)

Class-agnostic "ConceptGraphs" variant (SAM "segment all" + CLIP), run unmodified except the patches listed.
Nothing below is vendored into this repo; the image clones each source at the pinned commit.

| source | used for | licence |
|---|---|---|
| concept-graphs `93277a02bd89171f8121e84203121cf7af9ebb5d` | detections (`scripts/generate_gsa_results.py`) + object mapping (`slam/cfslam_pipeline_batch.py`) | MIT |
| patches `hpc/cg_patches/01_matplotlib_agg.patch` (TkAgg -> Agg, headless), `02_world_frame_poses.patch` (`get_dataset` default `relative_pose=False`), `hssd.yaml` (HSSD dataconfig, 640x480, f 320) | ours | — |
| Grounded-Segment-Anything `a4d76a2b55e348943cba4cd57d7553c354296223` (segment_anything, GroundingDINO) | GSA_PATH; SAM automatic mask generator; GroundingDINO is loaded by the script but unused with `--class_set none` | Apache-2.0 |
| SAM ViT-H `sam_vit_h_4b8939.pth` (dl.fbaipublicfiles.com) | masks | Apache-2.0 |
| GroundingDINO SwinT-OGC `groundingdino_swint_ogc.pth` (+ HF `bert-base-uncased` tokenizer/encoder) | loaded only | Apache-2.0 |
| open_clip 2.20.0, CLIP ViT-H-14 `laion2b_s32b_b79k` (HF `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`) | per-mask CLIP features; text embeddings for the CONTRACT label | MIT |
| recognize-anything `7cb804a8609e9f4b1a50b7f31436d2df40bb9481` | import only (the script imports `ram` at top level; RAM is not run, no checkpoint) | Apache-2.0 |
| gradslam `conceptfusion` @ `59ca872e3d265ad09f63c4793d011fad67064452` | dataset loader dependency | MIT |
| chamferdist @ `ee753899b583bdc06bd2505dc162fd54ba0c6c5b` (built for sm_80/sm_90) | CG knn dependency | MIT |
| PyTorch 2.0.1+cu118, torchvision 0.15.2, PyTorch3D 0.7.4 (built from source for sm_80/sm_90) | runtime | BSD-3 |
| open3d 0.17.0, faiss-cpu 1.7.4, supervision 0.14.0, hydra-core 1.3.2, transformers 4.31.0, timm 0.6.13 | runtime (pins of concept-graphs `environment.yml`; full list `/opt/pip_freeze.txt` in the image) | MIT / Apache-2.0 / BSD |

Not used: ultralytics (AGPL; only for `--detector yolo`), LLaVA, OpenAI API (scene-graph captions only), wandb.

## OVO baseline (task 003)

Container `hpc/ovo.def` (ubuntu 22.04, uv python 3.11); built on PERUN, weights kept outside the image in `$PROJ/ovo/`.

| source | used for | licence | note |
|---|---|---|---|
| OVO (`tberriel/OVO@e0d16f7a459ea5451540c50ac264b00aca3aeb85`), `slam_module: vanilla` (GT poses) | open-vocabulary 3D instance mapping baseline | MIT | cloned with submodules at build time |
| SAM 2 (`thirdParty/segment-anything-2`, `c98aa6b`) + SAM2.1 hiera-large checkpoint (`dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt`) | class-agnostic masks | Apache-2.0 | sha256 `2647878d...d318`; not redistributed |
| Meta Perception Models (`thirdParty/perception_models`, installed `--no-deps`; OVO imports it at top level) | CLIP utils import only | Meta FAIR licence (see repo) | PE-Core not used (model_card SigLIP-384) |
| SigLIP `timm/ViT-SO400M-14-SigLIP-384` (HF hub) | instance features + our text embeddings | Apache-2.0 | HF cache in `$PROJ/ovo/hf_home` |
| OVO CLIP weights predictor `data/input/weights_predictor/base/model.pt` (Google Drive id `186wZ2mLES_QjUjW8l2DmVmlWDpNl2fSY`, via `gdown`) | learned view fusion of instance features | licence unstated by the authors — not redistributed here; we report evaluation numbers only, obtained with the authors' public release (deviation 18) | used locally for evaluation; sha256 `cf83bc85...1192`; loaded with `torch.load(weights_only=False)` (pickle: only from the OVO author's Drive, not from untrusted sources) |
| torch 2.5.1+cu124, torchvision 0.20.1, transformers 4.51.0, open_clip_torch 2.32.0, open3d 0.19.0, numpy 1.26.4 (README pins numpy 2.4; <2 used) | runtime | BSD / Apache-2.0 / MIT | |

## ours v1 (task 004)

Weights live under `data/models/` (git-ignored), never committed or redistributed. Checksums are also written
per run to `data/baselines/<tag>/run_meta.json`.

| source | used for | licence | where / checksum |
|---|---|---|---|
| SAM 2 (`facebookresearch/sam2` @ `2b90b9f5ceec907a1c18123530e92e794ad901a4`, PyPI name `sam-2`, built with `SAM2_BUILD_CUDA=0`) | `SAM2AutomaticMaskGenerator`, class-agnostic masks | Apache-2.0 | `uv.lock` |
| SAM 2.1 hiera-small checkpoint `sam2.1_hiera_small.pt` (`dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt`), config `configs/sam2.1/sam2.1_hiera_s.yaml` | masks | Apache-2.0 | `data/models/sam2.1_hiera_small.pt`, sha256 `6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38` |
| SigLIP 2 base `google/siglip2-base-patch16-224` (HuggingFace) | mask-crop image embeddings (descriptor) + text embeddings of the CONTRACT 2026-10-27e prompts (label) | Apache-2.0 | `data/models/siglip2-base-patch16-224/`, `model.safetensors` sha256 `612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b` |
| transformers 4.57.1, huggingface-hub, tokenizers, safetensors | SigLIP 2 loading / processing | Apache-2.0 | PyPI (`uv.lock`) |
| PyTorch 2.7.1+cu126, torchvision 0.22.1+cu126 | runtime | BSD-3 | `download.pytorch.org/whl/cu126` (`uv.lock`) |
| hydra-core, omegaconf, iopath (SAM 2 dependencies) | config / IO | MIT | PyPI (`uv.lock`) |
| *v2 (task 006)* — encoder knob (`spatialmem/ours/encoders.py`); SAM 2.1-s, SigLIP 2 base and the rows above unchanged | | | |
| SigLIP 2 so400m `google/siglip2-so400m-patch14-384` @ `e8e487298228002f3d8a82e0cd5c8ea9c567f57f` (HuggingFace; transformers `SiglipModel`, text padding `max_length` 64 as on the model card) | v2 encoder `siglip2_so400m`: mask-crop image + CONTRACT-prompt text embeddings | Apache-2.0 | `data/models/siglip2-so400m-patch14-384/`, `model.safetensors` sha256 `9f4f4a49f908ef0c979bce8ff5a5c0e88882dc6c5dc4304387cbbd152558e2c2` (= HF LFS sha256) |
| CLIP ViT-H-14 `laion2b_s32b_b79k` (open_clip; HF `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` @ `1c2b8495b28150b8a4922ee1c8edee224c284c0c`, the ConceptGraphs encoder) | v2 encoder `clip_vith14_laion2b`: mask-crop image + CONTRACT-prompt text embeddings | MIT | `data/models/CLIP-ViT-H-14-laion2B-s32B-b79K/`, `open_clip_model.safetensors` sha256 `0084e75319a50ad85ef45377bad5bc38f2f58824459eb690048d51c9f8863be5` (= HF LFS sha256) |
| open_clip_torch 3.3.0 (+ timm 1.0.30, ftfy 6.3.1) | CLIP ViT-H-14 loading / preprocessing / tokenizer | MIT / Apache-2.0 / MIT | PyPI (`uv.lock`) |
| PERUN image `hpc/ours.def` (ubuntu 22.04, uv 0.12.19, `uv sync --frozen` of this `uv.lock`) | the same stack on H200 (deviation 11) | — | `$PROJ/ours.sif` (+ `.sha256`); weights in `$PROJ/models`, same sha256s |

## P1b planner (task 008; `hpc/p1b/`, image `$PROJ/vllm.sif` on PERUN, never on lpc)

| source | used for | licence / terms | where |
|---|---|---|---|
| Qwen2.5-VL-7B-Instruct (`Qwen/Qwen2.5-VL-7B-Instruct` @ `cc594898137f460bfe9f0759e9844b3ce807cfb5`, 16.6 GB) | planner rows `gt7b`, `img7b` | Apache-2.0 | PERUN `$PROJ/models/`; lpc only its tokenizer files (`data/models/qwen25vl_tokenizer/`, token counts) |
| Qwen2.5-VL-72B-Instruct (`Qwen/Qwen2.5-VL-72B-Instruct` @ `89c86200743eec961a297729e7990e8f2ddbc4c5`, 146.8 GB) | planner row `gt72b` (P1b planner after the 008 escalation) | "Qwen LICENSE AGREEMENT" (2024-09-19): use / modify / distribute allowed; distributed copies carry the attribution "Notice"; > 100M monthly active users needs a separate licence; outputs used to build or improve a distributed model → "Built with Qwen" / "Improved using Qwen"; no trademark grant; Chinese law. Research use here; nothing distributed | PERUN `$PROJ/models/` |
| vLLM 0.10.1.1 (docker `vllm/vllm-openai:v0.10.1.1`, image sha256 `c35531a7…`; torch 2.7.1+cu128, transformers 4.55.2) | inference, temperature 0 | Apache-2.0 | PERUN `$PROJ/vllm.sif` |
| HSSD stream frames (8 per house, `runs/hssd/<scene>/rgb/`) | `img7b` input | HSSD CC BY-NC 4.0 (derived renders) | PERUN only |
