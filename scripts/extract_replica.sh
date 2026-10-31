#!/usr/bin/env bash
# Extract what 114 uses from the fetched archives (scripts/fetch_replica.sh), then record checksums.
# NICE-SLAM zip -> data/nice/Replica/...; Replica v1 tar.gz (17 parts) -> only the 8 scenes' habitat/ dirs.
set -eu
DATA="$(cd "$(dirname "$0")/.." && pwd)/data"; DL="$DATA/_dl"
mkdir -p "$DATA/nice" "$DATA/replica_v1"
if [ ! -f "$DATA/nice/.done" ]; then
  unzip -q -o "$DL/Replica.zip" -d "$DATA/nice" && touch "$DATA/nice/.done" && echo "NICE extracted"
fi
if [ ! -f "$DATA/replica_v1/.done" ]; then
  pats=()
  for s in office_0 office_1 office_2 office_3 office_4 room_0 room_1 room_2; do pats+=("$s/habitat/*"); done
  cat "$DL"/replica_v1_0.tar.gz.parta? | gzip -dc | tar -x -C "$DATA/replica_v1" --wildcards "${pats[@]}" \
    && touch "$DATA/replica_v1/.done" && echo "V1 habitat dirs extracted"
fi
( cd "$DL" && sha256sum Replica.zip replica_v1_0.tar.gz.parta? ) > "$DATA/_dl_sha256.txt"
echo "CHECKSUMS DONE"
