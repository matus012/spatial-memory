#!/usr/bin/env bash
# Build the 114 OVO image on the PERUN login node (113 pattern: --fakeroot, tmp on node-local /tmp).
# Usage (on the login node): bash build_ovo.sh   -> $PROJ/ovo.sif (+ .sha256), log log: tee it
set -eu
PROJ=${PROJECT_DIR:?set PROJECT_DIR}
mkdir -p "$PROJ/build" "/tmp/$USER-apt"
export APPTAINER_TMPDIR=/tmp/$USER-apt APPTAINER_CACHEDIR=/tmp/$USER-apt/cache
cd "$PROJ/build"
echo "build start $(date -u +%s) $(date -Is)"
apptainer build --fakeroot --force "$PROJ/ovo.sif.tmp" "$PROJ/repo/hpc/ovo.def"
mv "$PROJ/ovo.sif.tmp" "$PROJ/ovo.sif"
sha256sum "$PROJ/ovo.sif" > "$PROJ/ovo.sif.sha256"
echo "build end $(date -u +%s) $(date -Is)"; ls -l "$PROJ/ovo.sif"; cat "$PROJ/ovo.sif.sha256"
echo BUILD_OK
