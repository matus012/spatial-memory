#!/bin/bash
#SBATCH --job-name=cg_g1
#SBATCH --partition=gpu_short
#SBATCH --account=<PERUN_ACCOUNT>
#SBATCH --chdir=<PROJECT_DIR>/runs/cg   # never cwd in $PROJ/repo (push_repo replaces it)
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=<PROJECT_DIR>/runs/cg/logs/%x_%j.out
#SBATCH --error=<PROJECT_DIR>/runs/cg/logs/%x_%j.err
# G1 (task 003): the CG image on an H200 — CUDA, pytorch3d + chamferdist CUDA ops, SAM ViT-H + CLIP ViT-H-14 on cuda;
# then the CLIP text embeddings of both vocabularies (spatialmem.baselines.cg text) -> $CGRUNS/text_{replica,hssd}.npz.
set -euo pipefail
source ${PROJECT_DIR:?set PROJECT_DIR}/repo/hpc/slurm/cg_env.sh
cg_snapshot
echo "=== cg_g1 on $(hostname) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1) repo $(cat $CODE/REV) sif $(cut -c1-12 $SIF.sha256) start $(date -Is) $(date -u +%s)"
cg_exec "$CGRUNS" - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version())
print("torch.cuda.is_available()", torch.cuda.is_available(), "device", torch.cuda.get_device_name(0),
      "capability", torch.cuda.get_device_capability(0), "arch_list", torch.cuda.get_arch_list())
dev = torch.device("cuda")
x = torch.randn(2, 4096, 3, device=dev); y = torch.randn(2, 2048, 3, device=dev)
from pytorch3d.ops import knn_points, box3d_overlap
k = knn_points(x, y, K=4)
ref = torch.cdist(x, y).topk(4, largest=False).values ** 2
print("pytorch3d knn_points on", k.dists.device, "max |d - cdist| =", float((k.dists - ref).abs().max()))
b = torch.tensor([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=torch.float32, device=dev)[None]
vol, iou = box3d_overlap(b, b + 0.5)
print("pytorch3d box3d_overlap on", iou.device, "iou(unit, +0.5 shift) =", float(iou), "(expect 0.125/1.875 = 0.0667)")
from chamferdist import ChamferDistance
from chamferdist.chamfer import knn_points as cd_knn
c = ChamferDistance()(x, y)
k2 = cd_knn(x, y, K=4)
print("chamferdist on", k2.dists.device, "chamfer", float(c), "max |knn - pytorch3d| =", float((k2.dists - k.dists).abs().max()))
from segment_anything import sam_model_registry
sam = sam_model_registry["vit_h"](checkpoint="/opt/gsa/sam_vit_h_4b8939.pth").to(dev)
print("SAM vit_h on", next(sam.parameters()).device, "params", sum(p.numel() for p in sam.parameters()))
import open_clip
clip, _, _ = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
clip = clip.to(dev)
print("CLIP ViT-H-14 laion2b_s32b_b79k on", next(clip.parameters()).device)
from groundingdino import _C
print("GroundingDINO _C (ms_deform_attn CUDA ext) importable")
print("peak allocated GB", round(torch.cuda.max_memory_allocated() / 2**30, 2))
EOF
for d in replica hssd; do cg_exec "$CGRUNS" -m spatialmem.baselines.cg text "$d" "$CGRUNS/text_$d.npz"; done
echo "=== cg_g1 done $(date -Is) $(date -u +%s)"
