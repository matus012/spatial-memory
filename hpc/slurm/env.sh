# Sourced by every 114 sbatch (113 pattern). apptainer is /usr/bin/apptainer.
export PROJ=${PROJECT_DIR:?set PROJECT_DIR}
export SIF=$PROJ/habitat.sif             # hpc/build_habitat.sh, sha256 in $SIF.sha256
export REPO=$PROJ/repo                   # hpc/push_repo.sh (git archive HEAD), bound at /workspace
export DATA=$PROJ/data                   # data/hssd-hab (scripts/hssd_fetch.py)
export RUNS=$PROJ/runs
mkdir -p "$RUNS/logs" "$PROJ/cache/tmp"
# Never `sbatch --export=...` (job goes HELD): positional args only.
hab_exec() {
  apptainer exec --nv --containall \
    --bind "$REPO:/workspace","$DATA:/data","$RUNS:/runs","$PROJ/cache:/cache" \
    --env PYTHONPATH=/workspace,PYTHONUNBUFFERED=1,TMPDIR=/cache/tmp,MPLCONFIGDIR=/cache/tmp,OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" \
    --pwd /workspace "$SIF" /opt/venv/bin/python "$@"
}
