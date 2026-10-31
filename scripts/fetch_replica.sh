#!/usr/bin/env bash
# Fetch Replica for 114: NICE-SLAM sequences (8 scenes, RGB-D + traj) and Replica v1 (semantic meshes).
# Resume-safe: every piece is fetched by explicit byte range and appended (never `curl -C -` with `-r`:
# a resumed range restarts from zero). Re-run to resume. Usage: scripts/fetch_replica.sh [nice|v1|all]
set -u
DATA="$(cd "$(dirname "$0")/.." && pwd)/data"; DL="$DATA/_dl"; mkdir -p "$DL"
NICE=https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip; NICE_SIZE=12442855671
V1=https://github.com/facebookresearch/Replica-Dataset/releases/download/v1.0/replica_v1_0.tar.gz.part

# fetch_range <url> <out> <start> <end-inclusive>
fetch_range() {
  local url=$1 out=$2 start=$3 end=$4 want=$(( $4 - $3 + 1 )) have tries=0
  touch "$out"
  while :; do
    have=$(stat -c%s "$out")
    if (( have > want )); then truncate -s "$want" "$out"; have=$want; fi
    (( have == want )) && return 0
    (( tries++ > 50 )) && { echo "GIVE UP $out ($have/$want)"; return 1; }
    curl -sSL -r "$((start + have))-$end" --speed-time 60 --speed-limit 1024 "$url" >> "$out" || sleep 5
  done
}

nice() {
  local n=8 c=$(( (NICE_SIZE + 7) / 8 )) i s e
  for (( i = 0; i < n; i++ )); do
    s=$(( i * c )); e=$(( s + c - 1 )); (( e >= NICE_SIZE )) && e=$(( NICE_SIZE - 1 ))
    fetch_range "$NICE" "$DL/nice.part$i" "$s" "$e" && echo "nice chunk $i ok"
  done
  cat "$DL"/nice.part{0..7} > "$DL/Replica.zip" && [ "$(stat -c%s "$DL/Replica.zip")" = "$NICE_SIZE" ] \
    && echo "NICE ASSEMBLED" && rm -f "$DL"/nice.part?
}

v1() {
  local p size
  for p in a b c d e f g h i j k l m n o p q; do
    size=2000000000; [ "$p" = q ] && size=1859047808
    fetch_range "${V1}a$p" "$DL/replica_v1_0.tar.gz.parta$p" 0 $(( size - 1 )) && echo "v1 part a$p ok"
  done
  echo "V1 DONE"
}

case "${1:-all}" in
  nice) nice ;; v1) v1 ;; all) nice & v1 & wait ;;
esac
