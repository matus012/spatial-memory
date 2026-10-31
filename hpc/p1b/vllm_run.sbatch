#!/bin/bash
#SBATCH --job-name=p1b
#SBATCH --partition=gpu_short
#SBATCH --account=<PERUN_ACCOUNT>
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --output=<PROJECT_DIR>/runs/logs/%x_%j.out
#SBATCH --error=<PROJECT_DIR>/runs/logs/%x_%j.err
# Task 008 P1b planner run: hpc/p1b/vllm_run.py in vllm.sif, from a per-job repo snapshot (hpc/push_snapshot.sh).
# GPUs = tp, given on the command line (positional args cannot set #SBATCH); the script checks they match:
#   cd $PROJ/cache/tmp/repo_<rev> && sbatch -A ${PERUN_ACCOUNT} -J <name> --gres=gpu:<tp> \
#       hpc/p1b/vllm_run.sbatch <model_dir> <prompts.jsonl> <out.jsonl> <tp>
set -euo pipefail
PROJ=${PROJECT_DIR:?set PROJECT_DIR}
MODEL=$1; PROMPTS=$2; OUT=$3; TP=$4
SIF=$PROJ/vllm.sif; REPO=$SLURM_SUBMIT_DIR
NGPU=$(nvidia-smi -L | wc -l)
[ "$NGPU" -eq "$TP" ] || { echo "tp $TP but $NGPU GPUs allocated (sbatch --gres=gpu:$TP)"; exit 1; }
mkdir -p "$(dirname "$OUT")"
echo "=== p1b $(basename "$MODEL") tp $TP on $(hostname) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1) repo $(cat $REPO/REV) sif $(cut -c1-12 $SIF.sha256) start $(date -Is) $(date -u +%s)"
mkdir -p $PROJ/cache/vllm $PROJ/cache/tmp/vllm_$SLURM_JOB_ID
# no --containall: tp>1 needs the host /dev/shm (NCCL); --cleanenv + explicit --env instead.
# CUDA_VISIBLE_DEVICES via APPTAINERENV_: its commas (tp>1) break the --env list (137734).
export APPTAINERENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
apptainer exec --nv --cleanenv -B "$PROJ" \
    --env HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,HF_HOME=$PROJ/cache/hf,VLLM_CACHE_ROOT=$PROJ/cache/vllm,TMPDIR=$PROJ/cache/tmp/vllm_$SLURM_JOB_ID,VLLM_NO_USAGE_STATS=1,DO_NOT_TRACK=1,PYTHONUNBUFFERED=1,SLURM_JOB_ID=$SLURM_JOB_ID \
    "$SIF" python3 "$REPO/hpc/p1b/vllm_run.py" "$MODEL" "$PROMPTS" "$OUT" "$TP"
echo "=== p1b done $(date -Is) $(date -u +%s)"
