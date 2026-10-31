# Sourced by every 114 OVO job (task 003). Container: hpc/ovo.def -> $PROJ/ovo.sif. Models live in $PROJ/ovo (not baked in).
export PROJ=${PROJECT_DIR:?set PROJECT_DIR}
export OSIF=$PROJ/ovo.sif
export REPO=$PROJ/repo
export ODIR=$PROJ/ovo                      # sam_ckpts/, weights_predictor/base/model.pt, hf_home/
export OWORK=$PROJ/runs/ovo/work_data      # writable copy of /opt/OVO/data (configs, input symlinks, output/)
export ORUNS=$PROJ/runs/ovo
mkdir -p "$ORUNS/logs" "$PROJ/cache/tmp"
# Never `sbatch --export=...`. Container sees: /opt/OVO/data = $OWORK, /data = $PROJ/data, /ovo_models = $ODIR, /workspace = repo.
ovo_exec() {
  apptainer exec --nv --containall \
    --bind "$OWORK:/opt/OVO/data","$PROJ/data:/data:ro","$ODIR:/ovo_models","$ORUNS:/runs","$PROJ/runs/hssd:$PROJ/runs/hssd:ro","$REPO:/workspace:ro","$PROJ/cache:/cache" \
    --env PYTHONPATH=/opt/OVO:/workspace,PYTHONUNBUFFERED=1,TMPDIR=/cache/tmp,MPLCONFIGDIR=/cache/tmp,HF_HOME=/ovo_models/hf_home,DISABLE_WANDB=true,WANDB_MODE=disabled,OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" \
    --pwd /opt/OVO "$OSIF" /opt/ovo/bin/python "$@"
}
