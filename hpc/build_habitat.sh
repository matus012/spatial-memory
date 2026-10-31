#!/usr/bin/env bash
# Build the 114 Habitat image on the PERUN login node (113 pattern: --fakeroot, tmp on node-local /tmp).
# Usage (on the login node): bash build_habitat.sh   -> $PROJ/habitat.sif (+ .sha256), log $PROJ/build/build.log
set -eu
PROJ=${PROJECT_DIR:?set PROJECT_DIR}
mkdir -p "$PROJ/build" "/tmp/$USER-apt"
export APPTAINER_TMPDIR=/tmp/$USER-apt APPTAINER_CACHEDIR=/tmp/$USER-apt/cache
cd "$PROJ/build"
echo "build start $(date -u +%s) $(date -Is)"
apptainer build --fakeroot --force "$PROJ/habitat.sif.tmp" "$PROJ/repo/hpc/habitat.def"
mv "$PROJ/habitat.sif.tmp" "$PROJ/habitat.sif"
sha256sum "$PROJ/habitat.sif" > "$PROJ/habitat.sif.sha256"
echo "build end $(date -u +%s) $(date -Is)"; ls -l "$PROJ/habitat.sif"; cat "$PROJ/habitat.sif.sha256"
echo BUILD_OK
