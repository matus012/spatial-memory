#!/bin/bash
# One scene of an ours_scenes.sbatch job: for each encoder in $ENCS, spatialmem.ours.v2 on <ds> <scene>.
# Env from the sbatch: CODE TAG MODE ENCS SETS PAR. Caches are idempotent (key hits are skipped by v2 itself).
set -euo pipefail
source "$CODE/hpc/slurm/ours_env.sh"
DS=$1; S=$2
L=$ORUNS/logs/${SLURM_JOB_ID:-manual}; mkdir -p "$L"
SET=(); [ "$SETS" = "-" ] || SET=(${SETS//,/ })
FLAG=(); [ "$MODE" = "build" ] && FLAG=(--build-only)
# run mode writes predictions to baselines/$TAG/<ds>/<scene>.json: one encoder per tag, or they overwrite each other
[ "$MODE" = "build" ] || [ "${ENCS//,/}" = "$ENCS" ] || { echo "run mode takes one encoder per tag"; exit 2; }
for E in ${ENCS//,/ }; do
  t0=$(date -u +%s)
  ours_exec -m spatialmem.ours.v2 --config configs/ours_v2.yaml --dataset "$DS" --scenes "$S" --tag "$TAG" \
      --set encoder="$E" "${SET[@]}" "${FLAG[@]}" --meta "run_meta_${S}_${E}.json" > "$L/${S}_${E}.log" 2>&1 \
      || { echo "FAILED $DS/$S $E, tail:"; tail -20 "$L/${S}_${E}.log"; exit 1; }
  echo "  $DS/$S $E $MODE $(( $(date -u +%s) - t0 )) s"
done
