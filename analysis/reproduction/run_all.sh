#!/usr/bin/env bash
# Reproduce the v1.1.0 analysis in Docker (R 4.5.0, package versions of analysis/renv.lock):
#   main_without_180/      main analysis, 53 samples (sample 180 excluded, Rmd default)
#   sensitivity_with_180/  sensitivity analysis, all 54 samples
# Each run folder receives the rendered scripts/ (HTML report, Word tables, figures),
# output/tables/*.csv, session.RData, sessionInfo.txt and key_numbers.csv (key_numbers.R).
#
# Usage (bash, from anywhere; OUT should lie outside the repository):
#   bash analysis/reproduction/run_all.sh <OUT>
# Needs Docker and network access (CRAN snapshot for the package restore, GeoSphere
# SPARTACUS API in the climate chunk). IMG and VOL may be overridden.
# Clean up afterwards: docker rmi "$IMG"; docker volume rm "$VOL".
set -euo pipefail
OUT=${1:?usage: run_all.sh <OUT>}
mixed() { if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else echo "$1"; fi; }
HERE=$(cd "$(dirname "$0")" && pwd)
A=$(cd "$HERE/.." && pwd)
IMG=${IMG:-iamp-analysis-r:4.5.0}
VOL=${VOL:-iamp-analysis-rlib}
mkdir -p "$OUT/restore"
OUT=$(cd "$OUT" && pwd)
export MSYS_NO_PATHCONV=1
Hm=$(mixed "$HERE"); Am=$(mixed "$A"); Om=$(mixed "$OUT")

docker build -t "$IMG" "$Hm"
docker run --rm -v "$VOL:/opt/rlib" -v "$Am/renv.lock:/lock/renv.lock:ro" -v "$Hm:/repro:ro" \
  -v "$Om/restore:/out" "$IMG" Rscript /repro/restore.R > "$OUT/restore/restore.log" 2>&1

run() {  # $1 = run folder, $2 = sample to exclude ("" = none)
  mkdir -p "$OUT/$1"
  local start; start=$(date -Iseconds)
  docker run --rm -e EXCLUDE_SAMPLE="$2" -v "$VOL:/opt/rlib" -v "$Am/scripts:/src:ro" \
    -v "$Am/data/raw:/data:ro" -v "$Hm:/repro:ro" -v "$Om/$1:/out" \
    "$IMG" bash /repro/run.sh > "$OUT/$1/render.log" 2>&1
  docker run --rm -v "$VOL:/opt/rlib" -v "$Hm:/repro:ro" -v "$Om/$1:/in" -v "$Om/$1:/out" \
    "$IMG" Rscript /repro/key_numbers.R > "$OUT/$1/key_numbers.log" 2>&1
  echo "$1: start $start end $(date -Iseconds)" | tee "$OUT/$1/time.txt"
  grep -E "^samples_n," "$OUT/$1/key_numbers.csv"
}

run main_without_180 180
run sensitivity_with_180 ""
