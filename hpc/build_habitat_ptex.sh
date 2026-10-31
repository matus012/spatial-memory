#!/usr/bin/env bash
# Build the 114 Habitat PTex image (task 006) on the PERUN login node (113 pattern: --fakeroot, tmp on node-local /tmp).
# Usage (on the login node): bash build_habitat_ptex.sh   -> $PROJ/habitat_ptex.sif (+ .sha256), log $PROJ/build/build_ptex.log
set -eu
PROJ=${PROJECT_DIR:?set PROJECT_DIR}
DEF="$(cd "$(dirname "$0")" && pwd)/habitat_ptex.def"   # def next to this script (no push_repo while jobs run)
mkdir -p "$PROJ/build" "/tmp/$USER-apt"
export APPTAINER_TMPDIR=/tmp/$USER-apt APPTAINER_CACHEDIR=/tmp/$USER-apt/cache
cd "$PROJ/build"
echo "build start $(date -u +%s) $(date -Is)"
apptainer build --fakeroot --force "$PROJ/habitat_ptex.sif.tmp" "$DEF"
mv "$PROJ/habitat_ptex.sif.tmp" "$PROJ/habitat_ptex.sif"
sha256sum "$PROJ/habitat_ptex.sif" > "$PROJ/habitat_ptex.sif.sha256"
echo "build end $(date -u +%s) $(date -Is)"; ls -l "$PROJ/habitat_ptex.sif"; cat "$PROJ/habitat_ptex.sif.sha256"
echo BUILD_OK
