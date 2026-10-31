#!/usr/bin/env bash
# Build the 114 ConceptGraphs image on the PERUN login node (--fakeroot, tmp on node-local /tmp) and fetch the
# checkpoints + CLIP weights into $PROJ/cg (bind-mounted at /cg, not baked in).
# Usage (on the login node): bash build_cg.sh [SRC]   SRC = tree holding hpc/cg.def + hpc/cg_patches (default $PROJ/repo)
#   -> $PROJ/cg.sif (+ .sha256); log to stdout
set -eu
PROJ=${PROJECT_DIR:?set PROJECT_DIR}
SRC=${1:-$PROJ/repo}
mkdir -p "/tmp/$USER-cg" "$PROJ/cg/ckpt" "$PROJ/cg/hf" "$PROJ/cg/torch"
export APPTAINER_TMPDIR=/tmp/$USER-cg APPTAINER_CACHEDIR=/tmp/$USER-cg/cache
cd "$SRC"
echo "build start $(date -u +%s) $(date -Is) src $SRC"
apptainer build --fakeroot --force "$PROJ/cg.sif.tmp" hpc/cg.def
mv "$PROJ/cg.sif.tmp" "$PROJ/cg.sif"
sha256sum "$PROJ/cg.sif" > "$PROJ/cg.sif.sha256"
echo "build end $(date -u +%s) $(date -Is)"; ls -l "$PROJ/cg.sif"; cat "$PROJ/cg.sif.sha256"
cd "$PROJ/cg/ckpt"
[ -s sam_vit_h_4b8939.pth ] || curl -LfsS -o sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
[ -s groundingdino_swint_ogc.pth ] || curl -LfsS -o groundingdino_swint_ogc.pth \
    https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
sha256sum sam_vit_h_4b8939.pth groundingdino_swint_ogc.pth | tee SHA256SUMS
# pre-cache CLIP ViT-H-14 laion2b_s32b_b79k (open_clip -> HF hub) and the GroundingDINO BERT tokenizer
apptainer exec --containall --bind "$PROJ/cg:/cg" "$PROJ/cg.sif" /opt/venv/bin/python -c '
import open_clip, transformers
open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
transformers.AutoTokenizer.from_pretrained("bert-base-uncased"); transformers.BertModel.from_pretrained("bert-base-uncased")
print("cache OK")'
echo BUILD_OK
