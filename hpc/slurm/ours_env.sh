# Sourced by every 114 "ours" job (task 006, deviation 11). Image: hpc/build_ours.sh -> $PROJ/ours.sif; weights $PROJ/models.
export PROJ=${PROJECT_DIR:?set PROJECT_DIR}
export SIF=$PROJ/ours.sif                         # sha256 in $SIF.sha256
export ORUNS=$PROJ/runs/ours                      # cache/ (v1 detections, v2 masks + desc), baselines/<tag>/, logs/, code/
mkdir -p "$ORUNS/logs" "$ORUNS/cache" "$ORUNS/baselines" "$ORUNS/tmp"
# Never `sbatch --export=...` (job goes HELD): positional args only.
# Per-job copy of the repo snapshot ($CODE = the code's ROOT), so a concurrent hpc/push_repo.sh cannot pull files from
# under a run (cwd never in $PROJ/repo). ROOT/data/* are symlinks to the PERUN copies (same layout as lpc data/).
# ours_snapshot [SRC]: SRC = an uploaded snapshot (hpc/push_snapshot.sh -> $PROJ/cache/tmp/repo_<rev>), default $PROJ/repo.
ours_snapshot() {
  local src=${1:-$PROJ/repo}
  export CODE=$ORUNS/code/${SLURM_JOB_ID:-manual}
  rm -rf "$CODE"; mkdir -p "$CODE/data/hssd_runs"
  cp -r "$src/spatialmem" "$src/configs" "$src/third_party" "$src/hpc" "$src/REV" "$CODE/"
  ln -s "$PROJ/runs/hssd" "$CODE/data/hssd_streams"       # = lpc data/hssd_streams (sha256-checked 2026-10-28)
  ln -s "$PROJ/runs/hssd" "$CODE/data/hssd_runs/all"      # HSSD GT <scene>/gt.json (scoring)
  ln -s "$PROJ/runs/replica_dev" "$CODE/data/replica_dev" # Replica-dev PTex streams (accepted 2026-10-29)
  ln -s "$PROJ/data/nice" "$CODE/data/nice"                 # NICE Replica 8 (protocol; the OVO/CG inputs; traj sha256 = lpc)
  ln -s "$ORUNS/gt" "$CODE/gt"                            # Replica-dev GT uploaded from lpc gt/ (sha256 in $ORUNS/gt/SHA256SUMS)
  ln -s "$PROJ/models" "$CODE/data/models"
  ln -s "$ORUNS/cache" "$CODE/data/cache"
  ln -s "$ORUNS/baselines" "$CODE/data/baselines"
}
# ours_exec <python args...>: $PROJ bound at its own path (symlinks resolve); offline HF; cwd = $CODE.
ours_exec() {
  apptainer exec --nv --containall --bind "$PROJ:$PROJ" \
    --env PYTHONPATH="$CODE",PYTHONUNBUFFERED=1,TMPDIR=$ORUNS/tmp,MPLCONFIGDIR=$ORUNS/tmp,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,OMP_NUM_THREADS="${OURS_THREADS:-8}" \
    --pwd "$CODE" "$SIF" /opt/venv/bin/python "$@"
}
