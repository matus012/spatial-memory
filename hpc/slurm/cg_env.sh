# Sourced by every 114 ConceptGraphs sbatch (env.sh pattern). Image: hpc/build_cg.sh -> $PROJ/cg.sif.
export PROJ=${PROJECT_DIR:?set PROJECT_DIR}
export SIF=$PROJ/cg.sif                          # sha256 in $SIF.sha256
export CGRUNS=$PROJ/runs/cg                      # per-scene work dirs = CG dataset roots (<ds>/<scene>/)
mkdir -p "$CGRUNS/logs" "$CGRUNS/tmp"
# Never `sbatch --export=...` (job goes HELD): positional args only.
# The repo snapshot is copied per job ($CODE) so a concurrent hpc/push_repo.sh cannot pull files from under a run.
cg_snapshot() {
  export CODE=$CGRUNS/code/${SLURM_JOB_ID:-manual}
  rm -rf "$CODE"; mkdir -p "$CODE"
  cp -r "$PROJ/repo/spatialmem" "$PROJ/repo/configs" "$PROJ/repo/third_party" "$PROJ/repo/hpc" "$PROJ/repo/REV" "$CODE/"
}
# cg_exec <pwd> <python args...>: $PROJ bound at its own path (the NICE-layout symlinks resolve), $PROJ/cg at /cg.
cg_exec() {
  local pwd=$1; shift
  apptainer exec --nv --containall \
    --bind "$PROJ:$PROJ","$PROJ/cg:/cg" \
    --env PYTHONPATH="$CODE",PYTHONUNBUFFERED=1,TMPDIR=$CGRUNS/tmp,MPLCONFIGDIR=$CGRUNS/tmp,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,OMP_NUM_THREADS="${CG_THREADS:-8}" \
    --pwd "$pwd" "$SIF" /opt/venv/bin/python "$@"
}
