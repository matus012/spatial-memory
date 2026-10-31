#!/usr/bin/env bash
# Build the 114 "ours" image on the PERUN login node (--fakeroot, tmp on node-local /tmp) and fetch the model weights
# into $PROJ/models (bound at <code>/data/models; not baked in). Weight sha256s must equal THIRD_PARTY.md (lpc).
# Usage (on the login node): bash build_ours.sh [SRC]   SRC = repo tree (default $PROJ/repo)  -> $PROJ/ours.sif (+ .sha256)
set -eu
PROJ=${PROJECT_DIR:?set PROJECT_DIR}
SRC=${1:-$PROJ/repo}
mkdir -p "/tmp/$USER-ours" "$PROJ/models"
export APPTAINER_TMPDIR=/tmp/$USER-ours APPTAINER_CACHEDIR=/tmp/$USER-ours/cache
cd "$SRC"
echo "build start $(date -u +%s) $(date -Is) src $SRC rev $(cat REV 2>/dev/null)"
apptainer build --fakeroot --force "$PROJ/ours.sif.tmp" hpc/ours.def
mv "$PROJ/ours.sif.tmp" "$PROJ/ours.sif"
sha256sum "$PROJ/ours.sif" > "$PROJ/ours.sif.sha256"
echo "build end $(date -u +%s) $(date -Is)"; ls -l "$PROJ/ours.sif"; cat "$PROJ/ours.sif.sha256"
# weights: big files fetched here (pinned revisions), small tokenizer/config files copied from lpc (scp)
M=$PROJ/models; H=https://huggingface.co
get() { [ -s "$2" ] || curl -LfsS --retry 10 -C - -o "$2" "$1"; }
mkdir -p $M/siglip2-base-patch16-224 $M/siglip2-so400m-patch14-384 $M/CLIP-ViT-H-14-laion2B-s32B-b79K
get https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt $M/sam2.1_hiera_small.pt
get $H/google/siglip2-base-patch16-224/resolve/main/model.safetensors $M/siglip2-base-patch16-224/model.safetensors
get $H/google/siglip2-so400m-patch14-384/resolve/e8e487298228002f3d8a82e0cd5c8ea9c567f57f/model.safetensors \
    $M/siglip2-so400m-patch14-384/model.safetensors
get $H/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/1c2b8495b28150b8a4922ee1c8edee224c284c0c/open_clip_model.safetensors \
    $M/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_model.safetensors
cd $M && sha256sum -c <<SUMS
6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38  sam2.1_hiera_small.pt
612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b  siglip2-base-patch16-224/model.safetensors
9f4f4a49f908ef0c979bce8ff5a5c0e88882dc6c5dc4304387cbbd152558e2c2  siglip2-so400m-patch14-384/model.safetensors
0084e75319a50ad85ef45377bad5bc38f2f58824459eb690048d51c9f8863be5  CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_model.safetensors
SUMS
echo BUILD_OK
