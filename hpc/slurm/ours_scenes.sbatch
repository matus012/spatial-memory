#!/bin/bash
#SBATCH --job-name=ours114
#SBATCH --partition=gpu_short
#SBATCH --account=<PERUN_ACCOUNT>
#SBATCH --chdir=<PROJECT_DIR>/runs/ours   # never cwd in $PROJ/repo (push_repo replaces it)
#SBATCH --time=01:00:00                  # SET PER JOB (sbatch --time); 006 cap 15 H200-h shared with the render worker
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --output=<PROJECT_DIR>/runs/ours/logs/%x_%j.out
#SBATCH --error=<PROJECT_DIR>/runs/ours/logs/%x_%j.err
# Usage: sbatch -A ${PERUN_ACCOUNT} -J ours_<tag> --time=HH:MM:SS hpc/slurm/ours_scenes.sbatch \
#            <hssd|replica_dev> <tag> <parallel> <build|run> <encoder[,encoder...]> <set,set,...|-> <scene>...
#   build: per scene, for each encoder in order: v2 --build-only (masks + detections once, then desc_<encoder>)
#   run:   per scene, for each encoder: v2 full (build what is missing + association + predictions in baselines/<tag>/)
# Up to <parallel> scenes at once on the one GPU (hpc/slurm/ours_run_scene.sh). Never the 8 Replica protocol scenes.
set -euo pipefail
source ${PROJECT_DIR:?set PROJECT_DIR}/repo/hpc/slurm/ours_env.sh
ours_snapshot
DS=$1; export TAG=$2 PAR=$3 MODE=$4 ENCS=$5 SETS=$6; shift 6
export OURS_THREADS=$(( SLURM_CPUS_PER_TASK / PAR ))
for S in "$@"; do case $S in room0|room1|room2|office0|office1|office2|office3|office4) echo "protocol scene $S: refused"; exit 2;; esac; done
echo "=== ours $DS $TAG $MODE enc $ENCS sets $SETS x$# par $PAR on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) repo $(cat $CODE/REV) sif $(cut -c1-12 $SIF.sha256) start $(date -Is) $(date -u +%s)"
fail=0
for S in "$@"; do
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do wait -n || fail=1; done
  bash "$CODE/hpc/slurm/ours_run_scene.sh" "$DS" "$S" &
done
while [ "$(jobs -rp | wc -l)" -gt 0 ]; do wait -n || fail=1; done
echo "=== ours $DS $TAG done fail=$fail $(date -Is) $(date -u +%s)"
exit $fail
