#!/bin/bash
#SBATCH --job-name=ovo114
#SBATCH --partition=gpu_short
#SBATCH --account=<PERUN_ACCOUNT>
#SBATCH --time=01:30:00                  # SET PER JOB at submit (sbatch --time=..)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=<PROJECT_DIR>/runs/ovo/logs/%x_%j.out
#SBATCH --error=<PROJECT_DIR>/runs/ovo/logs/%x_%j.err
# Usage: sbatch -J <unit> hpc/slurm/ovo_job.sbatch <script> [args]   (script = hpc/slurm/ovo_<script>.sh)
set -euo pipefail
source ${PROJECT_DIR:?set PROJECT_DIR}/repo/hpc/slurm/ovo_env.sh
S=$1; shift
echo "=== ovo job $SLURM_JOB_ID $S $* on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) repo $(cat $REPO/REV) sif $(cut -c1-12 $OSIF.sha256) start $(date -Is) $(date -u +%s)"
bash "$REPO/hpc/slurm/ovo_$S.sh" "$@"
echo "=== ovo job $SLURM_JOB_ID done $(date -Is) $(date -u +%s)"
