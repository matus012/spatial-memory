#!/bin/bash
#SBATCH --job-name=ours114t
#SBATCH --partition=cpu_short
#SBATCH --account=<PERUN_ACCOUNT>
#SBATCH --chdir=<PROJECT_DIR>/runs/ours   # never cwd in $PROJ/repo
#SBATCH --time=01:00:00                  # SET PER JOB (sbatch --time)
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --output=<PROJECT_DIR>/runs/ours/logs/%x_%j.out
#SBATCH --error=<PROJECT_DIR>/runs/ours/logs/%x_%j.err
# Task list runner (task 006). Usage:
#   sbatch -A ${PERUN_ACCOUNT} -J ours_<unit> --time=HH:MM:SS [-p gpu_short --gres=gpu:1 ...] hpc/slurm/ours_tasks.sbatch \
#       <repo snapshot dir> <parallel> <tasks file> [protocol-ok]
# Each line of <tasks file> is one shell command run in a subshell with `ours_exec` defined ($CODE = per-job copy of the
# snapshot), up to <parallel> at once; log $ORUNS/logs/<job>/<line no>.log. '#' lines are skipped. A line that touches
# the 8 Replica protocol scenes is refused unless the 4th argument is `protocol-ok` (only after the v2 freeze, 8eed799).
set -euo pipefail
SRC=$1; PAR=$2; TASKS=$3; PROTO=${4:-}
source "$SRC/hpc/slurm/ours_env.sh"
ours_snapshot "$SRC"
export OURS_THREADS=$(( SLURM_CPUS_PER_TASK / PAR )); [ "$OURS_THREADS" -ge 1 ] || OURS_THREADS=1
export -f ours_exec
[ "$PROTO" = protocol-ok ] || ! grep -E -q '(^|[ ,])(room[0-2]|office[0-4])($|[ ,])' "$TASKS" || { echo "protocol scene in $TASKS: refused"; exit 2; }
L=$ORUNS/logs/$SLURM_JOB_ID; mkdir -p "$L"; cp "$TASKS" "$L/tasks.txt"
echo "=== ours tasks $TASKS ($(grep -c -v -e '^#' -e '^$' "$TASKS") lines) par $PAR threads $OURS_THREADS on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) repo $(cat $CODE/REV) sif $(cut -c1-12 $SIF.sha256) start $(date -Is) $(date -u +%s)"
fail=0; i=0
while IFS= read -r line; do
  i=$((i + 1)); [[ -z "$line" || "$line" == \#* ]] && continue
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do wait -n || fail=1; done
  ( t0=$(date -u +%s); cd "$CODE"; if eval "$line" > "$L/$i.log" 2>&1; then echo "  ok $i $(( $(date -u +%s) - t0 )) s: $line"; else echo "  FAILED $i: $line"; tail -5 "$L/$i.log"; exit 1; fi ) &
done < "$TASKS"
while [ "$(jobs -rp | wc -l)" -gt 0 ]; do wait -n || fail=1; done
echo "=== ours tasks done fail=$fail $(date -Is) $(date -u +%s)"
exit $fail
