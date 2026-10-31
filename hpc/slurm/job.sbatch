#!/bin/bash
#SBATCH --job-name=sm114
#SBATCH --partition=gpu_short
#SBATCH --account=<PERUN_ACCOUNT>
#SBATCH --time=00:20:00                  # SET PER JOB; 002 budget <= 6 H200-h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=<PROJECT_DIR>/runs/logs/%x_%j.out
#SBATCH --error=<PROJECT_DIR>/runs/logs/%x_%j.err
# Usage: sbatch -J <unit> hpc/slurm/job.sbatch <unit> <python args...>   (runs `python <args>` in the image)
# Idempotent: exits 0 if /runs/<unit>/DONE exists.
set -euo pipefail
source ${PROJECT_DIR:?set PROJECT_DIR}/repo/hpc/slurm/env.sh
UNIT=$1; shift
[ -f "$RUNS/$UNIT/DONE" ] && { echo "$UNIT already done"; exit 0; }
mkdir -p "$RUNS/$UNIT"
echo "=== $UNIT on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) repo $(cat $REPO/REV) sif $(cut -c1-12 $SIF.sha256) start $(date -Is) $(date -u +%s)"
hab_exec "$@"
date -Is > "$RUNS/$UNIT/DONE"
echo "=== $UNIT done $(date -Is) $(date -u +%s)"
