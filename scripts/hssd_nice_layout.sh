#!/usr/bin/env bash
# Expose 002's HSSD streams in the NICE-SLAM Replica layout (task 003): every baseline reads the same bytes.
#   $PROJ/data/hssd_nice/<scene>/results/frame%06d.jpg -> runs/hssd/<scene>/rgb/%06d.jpg
#   $PROJ/data/hssd_nice/<scene>/results/depth%06d.png -> runs/hssd/<scene>/depth/%06d.png   (depth png / 6553.5)
#   $PROJ/data/hssd_nice/<scene>/traj.txt              =  runs/hssd/<scene>/poses.txt (c2w, OpenCV axes, 16 per line)
#   $PROJ/data/hssd_nice/cam_params.json               640 x 480, fx = fy = 320, cx 319.5, cy 239.5, scale 6553.5
# Usage (PERUN login node): bash hssd_nice_layout.sh
set -eu
PROJ=${PROJECT_DIR:?set PROJECT_DIR}
OUT=$PROJ/data/hssd_nice
mkdir -p "$OUT"
printf '{\n  "camera": {"w": 640, "h": 480, "fx": 320.0, "fy": 320.0, "cx": 319.5, "cy": 239.5, "scale": 6553.5}\n}\n' > "$OUT/cam_params.json"
for d in "$PROJ"/runs/hssd/*/; do
  s=$(basename "$d")
  [ -f "$d/check.json" ] || { echo "skip $s (incomplete)"; continue; }
  mkdir -p "$OUT/$s/results"
  n=$(ls "$d/rgb" | wc -l)
  for ((i = 0; i < n; i++)); do
    f=$(printf %06d "$i")
    ln -sfn "$d/rgb/$f.jpg" "$OUT/$s/results/frame$f.jpg"
    ln -sfn "$d/depth/$f.png" "$OUT/$s/results/depth$f.png"
  done
  cp "$d/poses.txt" "$OUT/$s/traj.txt"
  [ "$(wc -l < "$OUT/$s/traj.txt")" -eq "$n" ] || { echo "pose count mismatch $s"; exit 1; }
  echo "$s $n"
done
