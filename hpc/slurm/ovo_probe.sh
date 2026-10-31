#!/usr/bin/env bash
# G1: CUDA + SigLIP + SAM2 on device; also pre-caches the SigLIP weights into $ODIR/hf_home. Then text embeddings.
set -eu
source "$(dirname "$0")/ovo_env.sh"
mkdir -p "$ORUNS/textemb"
ovo_exec -m spatialmem.baselines.ovo probe
for v in replica hssd; do ovo_exec -m spatialmem.baselines.ovo textemb $v /runs/textemb_$v.npz; done
