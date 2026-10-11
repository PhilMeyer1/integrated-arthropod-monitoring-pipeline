#!/usr/bin/env bash
# Render analysis/scripts/Data_Analysis.Rmd inside the container (started by run_all.sh).
# Mounts: /src   = analysis/scripts (read-only)
#         /data  = analysis/data/raw (read-only)
#         /repro = analysis/reproduction (read-only)
#         /out   = output folder of this run
# EXCLUDE_SAMPLE: zuordnung_id of the sample to exclude; "180" = main analysis (default),
# "" = sensitivity analysis with all 54 samples.
set -euo pipefail
work=/run/analysis
rm -rf "$work"
mkdir -p "$work/scripts" "$work/data/processed" "$work/output/tables" "$work/output/reports"
ln -s /data "$work/data/raw"
# A Windows checkout may carry CRLF; the patch is applied to an LF copy. The
# archived Rmd itself stays untouched (reproduction.patch explains each change).
tr -d '\r' < /src/Data_Analysis.Rmd > "$work/scripts/Data_Analysis.Rmd"
tr -d '\r' < /repro/reproduction.patch | patch --no-backup-if-mismatch "$work/scripts/Data_Analysis.Rmd"
# The YAML header names css: styles.css, which is not in the archive;
# pandoc aborts without it, so an empty stylesheet stands in.
[ -e /src/styles.css ] || : > "$work/scripts/styles.css"
cd "$work/scripts"
set +e
Rscript -e 'sessionInfo(); rmarkdown::render("Data_Analysis.Rmd", output_dir = ".", params = list(exclude_sample = Sys.getenv("EXCLUDE_SAMPLE", "180")), envir = globalenv()); save.image("/out/session.RData"); writeLines(capture.output(sessionInfo()), "/out/sessionInfo.txt")'
status=$?
set -e
rm -f "$work/data/raw"
cp -r "$work/scripts" "$work/data" "$work/output" /out/
echo "render exit status: $status (EXCLUDE_SAMPLE='${EXCLUDE_SAMPLE-180}')"
exit $status
