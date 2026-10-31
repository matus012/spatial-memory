#!/bin/bash
#SBATCH --job-name=cg114
#SBATCH --partition=gpu_short
#SBATCH --account=<PERUN_ACCOUNT>
#SBATCH --chdir=<PROJECT_DIR>/runs/cg   # never cwd in $PROJ/repo (push_repo replaces it)
#SBATCH --time=01:00:00                  # SET PER JOB (sbatch --time); 003 CG budget <= 14 H200-h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --output=<PROJECT_DIR>/runs/cg/logs/%x_%j.out
#SBATCH --error=<PROJECT_DIR>/runs/cg/logs/%x_%j.err
# Usage: sbatch -J cg_<tag> --time=HH:MM:SS hpc/slurm/cg_scenes.sbatch <replica|hssd> <parallel> <det|map|all> <scene>...
# map-only jobs need no GPU: sbatch -p cpu_short --gres=none ... (mapping never creates a CUDA context)
# Runs up to <parallel> scenes at once on the one GPU (each scene: hpc/slurm/cg_run_scene.sh, idempotent).
set -euo pipefail
source ${PROJECT_DIR:?set PROJECT_DIR}/repo/hpc/slurm/cg_env.sh
cg_snapshot
DS=$1; export PAR=$2; export PHASES=${3/all/det map}; shift 3
export CG_THREADS=$(( SLURM_CPUS_PER_TASK / PAR ))
echo "=== cg $DS x$# par $PAR on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) repo $(cat $CODE/REV) sif $(cut -c1-12 $SIF.sha256) start $(date -Is) $(date -u +%s)"
fail=0
for S in "$@"; do
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do wait -n || fail=1; done
  bash "$CODE/hpc/slurm/cg_run_scene.sh" "$DS" "$S" &
done
while [ "$(jobs -rp | wc -l)" -gt 0 ]; do wait -n || fail=1; done
echo "=== cg $DS done fail=$fail $(date -Is) $(date -u +%s)"
exit $fail
