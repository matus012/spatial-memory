#!/bin/bash
# One ConceptGraphs scene (class-agnostic variant, README commands): detections -> mapping -> objects.npz.
# Usage (inside a job, after `source cg_env.sh; cg_snapshot`): PHASES="det map" bash cg_run_scene.sh <replica|hssd> <scene>
# Idempotent: skips when $CGRUNS/<ds>/<scene>/DONE exists. Writes time_<phase>.tsv (phase, start, end, peak MiB, scenes run concurrently, job).
set -euo pipefail
source "$CODE/hpc/slurm/cg_env.sh"
DS=$1; S=$2; PHASES=${PHASES:-det map}   # det = GPU (SAM + CLIP); map + export = CPU only (no CUDA context seen)
case $DS in
  replica) SRC=$PROJ/data/nice/Replica/$S; CFG=/opt/concept-graphs/conceptgraph/dataset/dataconfigs/replica/replica.yaml ;;
  hssd)    SRC=$PROJ/data/hssd_nice/$S;    CFG=/opt/concept-graphs/conceptgraph/dataset/dataconfigs/hssd/hssd.yaml ;;
esac
ROOT=$CGRUNS/$DS; W=$ROOT/$S
[ -f "$W/DONE" ] && { echo "$DS/$S already done"; exit 0; }
mkdir -p "$W"; ln -sfn "$SRC/results" "$W/results"; ln -sfn "$SRC/traj.txt" "$W/traj.txt"
SUF=overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub
NF=$(ls "$SRC/results" | grep -c '^frame')

# peak per-process GPU memory of this scene's processes (cmdline contains the scene id), polled every 2 s
vram_poll() {  # $1 = out file. Runs in the background: must never die on a non-matching grep (set -e + pipefail)
  set +e +o pipefail
  local peak=0
  while :; do
    local m
    m=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null |
        while IFS=', ' read -r pid mem; do tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | grep -q -- "$S" && echo "$mem"; done |
        awk '{s+=$1} END {print s+0}')
    [ "$m" -gt "$peak" ] && peak=$m && echo "$peak" > "$1"
    sleep 2
  done
}
phase() {  # phase <name> <python args...>; skipped when its time_<name>.tsv exists (written on success only)
  local name=$1; shift
  [ -f "$W/time_$name.tsv" ] && { echo "  $DS/$S $name already done"; return; }
  echo 0 > "$W/vram_$name.mib"
  vram_poll "$W/vram_$name.mib" & local vp=$!
  local t0; t0=$(date -u +%s)
  cg_exec "$W" "$@" > "$W/$name.log" 2>&1 || { kill $vp 2>/dev/null; echo "FAILED $name ($DS/$S), tail:"; tail -20 "$W/$name.log"; exit 1; }
  local t1; t1=$(date -u +%s)
  kill $vp 2>/dev/null || true; wait $vp 2>/dev/null || true
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$t0" "$t1" "$(cat "$W/vram_$name.mib")" "$PAR" "${SLURM_JOB_ID:-manual}" > "$W/time_$name.tsv"
  echo "  $DS/$S $name $((t1 - t0)) s, peak $(cat "$W/vram_$name.mib") MiB"
}
echo "== $DS/$S frames $NF stride 5 phases '$PHASES' start $(date -u +%s)"
if [[ " $PHASES " == *" det "* ]]; then
phase det /opt/concept-graphs/conceptgraph/scripts/generate_gsa_results.py --dataset_root "$ROOT" --dataset_config "$CFG" \
  --scene_id "$S" --class_set none --stride 5
fi
[[ " $PHASES " == *" map "* ]] || exit 0
# scene_id quoted for hydra: HSSD ids such as 102816756 / 104348181_171513120 parse as YAML ints
[ -f "$W/time_det.tsv" ] || { echo "FAILED map ($DS/$S): detections not done"; exit 1; }
phase map /opt/concept-graphs/conceptgraph/slam/cfslam_pipeline_batch.py dataset_root="$ROOT" dataset_config="$CFG" \
  stride=5 scene_id="'$S'" spatial_sim_type=overlap mask_conf_threshold=0.95 match_method=sim_sum sim_threshold=1.2 \
  dbscan_eps=0.1 gsa_variant=none class_agnostic=True skip_bg=True max_bbox_area_ratio=0.5 save_suffix=$SUF \
  merge_interval=20 merge_visual_sim_thresh=0.8 merge_text_sim_thresh=0.8
cg_exec "$W" -m spatialmem.baselines.cg export "$W/pcd_saves/full_pcd_none_${SUF}_post.pkl.gz" "$W/objects.npz" > "$W/export.log" 2>&1
cat "$W/export.log"
echo "$NF $(( (NF + 4) / 5 )) ${SLURM_JOB_ID:-manual}" > "$W/DONE"
echo "== $DS/$S done $(date -u +%s)"
