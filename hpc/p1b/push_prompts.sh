#!/usr/bin/env bash
# Upload prompt JSONL files to $PROJ/runs/p1b/prompts/ and print their sha256 there (compare with lpc sha256sum).
# Refuses to overwrite an existing file with different content (prompts are frozen once a row has run).
# Code goes up separately as a per-job snapshot: hpc/push_snapshot.sh (never hpc/push_repo.sh while jobs run).
# usage: hpc/p1b/push_prompts.sh <prompts.jsonl>...
set -eu
D=${PROJECT_DIR:?set PROJECT_DIR}/runs/p1b/prompts
H="ssh -o BatchMode=yes -o LogLevel=ERROR ${LOGIN_NODE:?set LOGIN_NODE}"
$H "mkdir -p $D"
for f in "$@"; do
  b=$(basename "$f"); s=$(sha256sum "$f" | cut -c1-64)
  r=$($H "[ -f $D/$b ] && sha256sum $D/$b | cut -c1-64 || true")
  [ -z "$r" ] || [ "$r" = "$s" ] || { echo "$D/$b exists with sha256 $r != local $s; refusing"; exit 1; }
  [ "$r" = "$s" ] || $H "cat > $D/$b.part && mv $D/$b.part $D/$b" < "$f"
  echo "$s  $D/$b"
done
