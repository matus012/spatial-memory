#!/bin/bash
#SBATCH --job-name=rdev
#SBATCH --partition=gpu_short
#SBATCH --account=<PERUN_ACCOUNT>
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=<PROJECT_DIR>/runs/logs/%x_%j.out
#SBATCH --error=<PROJECT_DIR>/runs/logs/%x_%j.err
# Task 006 Replica-dev render: like job.sbatch, but (a) runs from a repo copy given as an argument (git archive of a
# local commit uploaded by the caller, never $PROJ/repo: hpc/push_repo.sh replaces $REPO under running jobs -> "Stale
# file handle", 137035/137037), (b) takes the image as an argument (habitat_ptex.sif for PTex, habitat.sif otherwise).
# Usage: sbatch -A ${PERUN_ACCOUNT} -J <unit> hpc/slurm/replica_dev_render.sbatch <unit> <repo dir> <sif> <python args...>
set -euo pipefail
source ${PROJECT_DIR:?set PROJECT_DIR}/repo/hpc/slurm/env.sh
UNIT=$1; REPO=$2; SIF=$3; shift 3
[ -f "$RUNS/$UNIT/DONE" ] && { echo "$UNIT already done"; exit 0; }
mkdir -p "$RUNS/$UNIT"
echo "=== $UNIT on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) repo $(cat $REPO/REV) sif $(cut -c1-12 $SIF.sha256) start $(date -Is) $(date -u +%s)"
hab_exec "$@"
date -Is > "$RUNS/$UNIT/DONE"
echo "=== $UNIT done $(date -Is) $(date -u +%s)"
