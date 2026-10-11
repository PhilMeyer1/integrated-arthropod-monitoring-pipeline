# Release v1.1.0 – Revision 1 (main analysis excluding sample 180)

**Release date**: 2026-10-11
**Archive**: Zenodo, DOI of all versions [10.5281/zenodo.17661709](https://doi.org/10.5281/zenodo.17661709)
(resolves to the latest version); previous version 1.0.0: [10.5281/zenodo.17661710](https://doi.org/10.5281/zenodo.17661710)
**Models and example images** (unchanged): [10.5281/zenodo.17661921](https://doi.org/10.5281/zenodo.17661921)

This version accompanies the first revision of the manuscript "Integrated arthropod monitoring
pipeline detects size-dependent responses" (manuscript under revision at Scientific Reports, 2026).

## What changed

### Analysis: main analysis without sample 180, sensitivity analysis with it

- Sample 180 was collected over two days instead of 24 hours. The main analysis
  now excludes it (53 samples); a sensitivity analysis repeats the analysis with all 54 samples,
  as in the submitted version.
- Both run from the same `analysis/scripts/Data_Analysis.Rmd`, controlled by one parameter in its
  YAML header (`exclude_sample: "180"`; empty = all samples). The parameter removes the specimens
  of that sample right after `specimen_data.xlsx` is read. Nothing else in the analysis code
  changed.
- How to run both: `analysis/README.md`, sections "Main analysis and sensitivity analysis" and
  "Reproduction in Docker".

### Reproduction in Docker (`analysis/reproduction/`)

- `run_all.sh` renders the main and the sensitivity analysis in Docker (R 4.5.0, package versions
  of `analysis/renv.lock`), each into its own folder, and extracts the key numbers of each run
  (`key_numbers.R` → `key_numbers.csv`).
- `reproduction.patch` holds the changes needed to run the archived Rmd against the archived data;
  it is applied to a scratch copy only, and every change is marked `# REPRODUCTION:` (see Known
  gaps).

### Tables S14 and S15

- `make_table_s14.py`: Table S14, orders and families of the 53 samples of the main analysis,
  counted from `specimen_data.xlsx` (37,612 specimens; checked against the main analysis).
- `make_table_s15.py`: Table S15, the key results of the main analysis next to the sensitivity
  analysis, computed from the `key_numbers.csv` of both runs.
- Both tables are in `analysis/reproduction/tables/`; `--check` regenerates and compares them.

### Data

- `analysis/data/raw/publication_data/` holds data files reduced to what the analysis reads to
  reproduce the results of the manuscript. Sites carry neutral codes (`site_01`, `site_01 b`,
  `site_02` to `site_10`), sampling dates are given as year and day of the year, and
  coordinates, object IDs and taxonomic ranks the analysis does not use are not included.
  Sample numbers (`zuordnung_id`) are those of the manuscript (sample 180 is `180`).
- `specimen_data.xlsx`: sheet `Object List`, the 40,090 specimens with an order in the 54 valid
  samples (10 columns). `biomass_data.xlsx`: 371 weighed size fractions (`zuordnung_id`,
  `size`, `biomass_size`). `model_performance.xlsx`: the two test-set sheets the analysis reads,
  unchanged.
- New: `climate_data.xlsx`, daily precipitation and minimum and maximum temperature (GeoSphere
  SPARTACUS v2.1) for each site and sampling day, as queried for the reference run. The
  analysis reads it instead of querying the API, which needs the site coordinates.
- Columns and coding: `analysis/data/raw/publication_data/README.md`.

### Metadata and documentation

- Version 1.1.0 in `.zenodo.json`, `CITATION.cff` and `analysis/DESCRIPTION`; ORCID of the
  corresponding author in `.zenodo.json`.
- The manuscript is described as "manuscript under revision at Scientific Reports (2026)" instead
  of "manuscript in review, 2025".
- The code archive is cited with the DOI of all versions (10.5281/zenodo.17661709); version 1.0.0
  is named as the earlier version.
- Removed: the reference to an older, separate Zenodo software record ("Arthropod Classification
  Pipeline", 2025-11-03), which was listed as the companion article; the article has no DOI yet.
- README files: placeholder and outdated publication notes replaced; the key findings in
  `README.md` are marked as the figures of the submitted version (54 samples).

## What is unchanged

- Pipeline code (`pipeline/`): detection, segmentation and hierarchical classification.
- Detection and segmentation models and example images (Zenodo 10.5281/zenodo.17661921).
  The hierarchical classification models remain available upon reasonable request.
- The graphical user interface (Tkinter) used for imaging is not part of this archive.
- The analysis code in `Data_Analysis.Rmd` apart from the sample parameter; commented-out lines
  that named sampling sites now use site codes or placeholders.

## Verification

Both analyses were rerun in Docker on 2026-10-11 on the reduced data files (renders without
network access) and compared value by value with runs of the same analysis on the full data
files of version 1.0.0. Main analysis (53 samples): of 933 values (key numbers and output
tables), 927 are equal (exactly or within a relative difference of 1e-9); the other 6 are
permutation P-values of Tables S8 and S10, which use no fixed seed. Sensitivity analysis (54
samples): 913 of 921 values equal, the other 8 are such permutation P-values. The rows that
feed the seeded indicator species analysis (Table S12) keep their order, and its results are
identical. Table S14 is regenerated by `--check` without differences; Table S15 is regenerated
without differences from the runs it was made from, and on the new runs only values in its
three rows of permutation P-values (Table S10) differ.

## Known gaps

- **Confidence threshold**: the classification threshold applied per taxon is 0.9 where the
  initial accuracy reached at least 95 %, otherwise 0.95. A recalculation of the
  threshold-dependent numbers at 0.95 needs the pipeline database, whose export is still
  pending; it is not part of this version.
- **Climate data**: archived as values, not as a query. `climate_data.xlsx` holds the daily
  values per site and sampling day; the climate chunk of the Rmd (`eval=FALSE` in the archived
  script) queries the GeoSphere SPARTACUS API (v2.1) with the site coordinates, which are not
  part of this version, so `reproduction.patch` keeps the query switched off.
- **Reduced data and the archived Rmd**: `Data_Analysis.Rmd` is archived as used for the
  manuscript and expects the full data files (columns by position, month and day, coordinates).
  On the reduced files it runs with `reproduction.patch`, which `run.sh` applies to a scratch
  copy.
- **Path diagram (Fig. 3h)**: the image `path_fig_20250819.png` is not included; the reproduction
  run uses a labelled placeholder for this panel.
- **Biomass sheet**: `biomass_data.xlsx` holds one prepared sheet, while the Rmd reads two raw
  sheets; `reproduction.patch` reads the prepared sheet instead.
- **Permutation tests**: P-values of the PERMANOVA and beta-diversity tests (Tables S8 to S11)
  vary slightly between runs because no seed is set there.
