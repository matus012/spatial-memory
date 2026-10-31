#!/usr/bin/env bash
# Run OVO-mapping (GT poses) on scenes of one dataset; idempotent per scene (skips if $ORUNS/done/<DS>_<scene>).
# Usage: ovo_scenes.sh <Replica|HSSD> <scene>...   Replica additionally runs --segment --eval per scene (own labels txt).
# Per scene: wall of the --run step, peak VRAM (nvidia-smi poll every 2 s) -> $ORUNS/timing/<DS>_<scene>.csv; ckpt -> npz.
set -euo pipefail
source "$(dirname "$0")/ovo_env.sh"
DS=$1; shift
mkdir -p "$ORUNS/timing" "$ORUNS/done" "$ORUNS/npz/$DS" "$ORUNS/eval_logs"
export APPTAINER_BINDPATH=   # keep ovo_exec's explicit binds only
OUTD=$OWORK/output/$DS/ovo_mapping
for s in "$@"; do
  [ -f "$ORUNS/done/${DS}_$s" ] && { echo "$DS $s already done"; continue; }
  rm -rf "$OUTD/$s"
  frames=$(ls "$PROJ/data/$([ $DS = Replica ] && echo nice/Replica || echo hssd_nice)/$s/results" | grep -c '^frame')
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 >> "$ORUNS/timing/${DS}_$s.vram"; sleep 2; done ) &
  PID=$!
  t0=$(date -u +%s)
  ovo_exec run_eval.py --dataset_name $DS --experiment_name ovo_mapping --scenes $s --run
  t1=$(date -u +%s)
  kill $PID || true
  peak=$(sort -n "$ORUNS/timing/${DS}_$s.vram" | tail -1)
  echo "$DS,$s,$frames,$((t1-t0)),$peak,${SLURM_JOB_ID:-0}" > "$ORUNS/timing/${DS}_$s.csv"
  echo "TIMING $DS,$s,frames=$frames,wall_s=$((t1-t0)),peak_vram_mib=$peak"
  if [ $DS = Replica ]; then
    t2=$(date -u +%s)
    ovo_exec run_eval.py --dataset_name $DS --experiment_name ovo_mapping --scenes $s --segment --eval 2>&1 | tee "$ORUNS/eval_logs/$s.log"
    echo "SEGEVAL $s wall_s=$(( $(date -u +%s) - t2 ))"
  fi
  ovo_exec -m spatialmem.baselines.ovo convert /opt/OVO/data/output/$DS/ovo_mapping/$s/ovo_map.ckpt /runs/npz/$DS/$s.npz
  touch "$ORUNS/done/${DS}_$s"
done
