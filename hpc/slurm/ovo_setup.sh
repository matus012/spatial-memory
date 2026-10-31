#!/usr/bin/env bash
# One-time (idempotent) on the login node: writable OVO data dir with configs + input symlinks (resolved inside the container).
# Usage: bash hpc/slurm/ovo_setup.sh        (needs $PROJ/ovo/{sam_ckpts,weights_predictor/base/model.pt})
set -eu
source "$(dirname "$0")/ovo_env.sh"
mkdir -p "$OWORK"
apptainer exec --containall --bind "$OWORK:/w" "$OSIF" bash -c 'cp -rn /opt/OVO/data/working /w/ && mkdir -p /w/input/Datasets/Replica /w/input/Datasets/HSSD /w/output && cp -n /opt/OVO/data/input/ReadMe.md /w/input/'
apptainer exec --containall --bind "$ODIR:/m" "$OSIF" cp -n /opt/OVO/data/input/weights_predictor/base/hparams.yaml /m/weights_predictor/base/
I=$OWORK/input
ln -sfn /ovo_models/sam_ckpts "$I/sam_ckpts"
ln -sfn /ovo_models/weights_predictor "$I/weights_predictor"
for s in office0 office1 office2 office3 office4 room0 room1 room2; do ln -sfn /data/nice/Replica/$s "$I/Datasets/Replica/$s"; ln -sfn /data/nice/Replica/${s}_mesh.ply "$I/Datasets/Replica/${s}_mesh.ply"; done  # OVO eval reads <ds>/<scene>_mesh.ply
ln -sfn /data/replica_gt/ovo_txt "$I/Datasets/Replica/semantic_gt"
for s in $(ls "$PROJ/data/hssd_nice" | grep -v json); do ln -sfn /data/hssd_nice/$s "$I/Datasets/HSSD/$s"; done
# HSSD = replica loader + HSSD intrinsics (640x480, fx=fy=320, cx 319.5, cy 239.5, depth scale 6553.5)
C=$OWORK/working/configs
mkdir -p "$C/HSSD"
cp -n "$C/slam/vanilla/replica.yaml" "$C/slam/vanilla/hssd.yaml"
cat > "$C/HSSD/hssd.yaml" <<'Y'
dataset_name: "replica"
cam:
  H: 480
  W: 640
  fx: 320.0
  fy: 320.0
  cx: 319.5
  cy: 239.5
  depth_scale: 6553.5
Y
echo SETUP_OK; ls "$I" "$I/Datasets/HSSD" | head -12
