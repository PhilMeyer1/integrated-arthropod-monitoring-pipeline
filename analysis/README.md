# Data Analysis

Statistical data analysis for the publication "Cheaper, Faster, More Comprehensive: AI-based image analysis pipeline for biodiversity surveys".

## Project Structure

```
data-analysis/
├── data/
│   ├── raw/              # Raw publication data
│   │   └── publication_data/
│   └── processed/        # Processed data for analysis
├── scripts/
│   └── Data_Analysis.Rmd # Main R Markdown analysis script
├── output/
│   ├── tables/           # Generated tables
│   └── reports/          # HTML/PDF reports
└── README.md
```

## Requirements

- **R version**: 4.0.0 or higher
- **Required packages**: 25 packages (see `DESCRIPTION` file)
- **System**: Windows, macOS, or Linux

## Installation

This analysis requires several R packages. We provide three installation methods:

### Method 1: Using renv (Recommended for Reproducibility)

This method installs the exact package versions used in the publication:

```r
# Install renv if not already installed
install.packages("renv")

# Restore exact package versions from renv.lock
renv::restore()
```

**When to use**: For exact reproducibility of publication results, code reviews, or when submitting to journals.

### Method 2: Using DESCRIPTION File

This method installs the latest compatible versions of all required packages:

```r
# Install remotes if not already installed
install.packages("remotes")

# Install all dependencies from DESCRIPTION
remotes::install_deps(dependencies = TRUE)
```

**When to use**: For general analysis or when you want the latest package versions.

### Method 3: Interactive Setup Script

Run the interactive setup script that guides you through the installation:

```r
source("setup.R")
```

This script will:
1. Check your R version
2. Present all three installation options
3. Install missing packages
4. Verify the installation

### Manual Package Installation

If you prefer to install packages manually:

```r
required_packages <- c(
  "DT", "tidyverse", "rio", "data.table", "lubridate",
  "ggpubr", "ggsci", "mgcv", "ggstatsplot", "patchwork",
  "funspace", "car", "broom", "glmnet", "MASS",
  "DHARMa", "codyn", "viridis", "iNEXT", "vegan",
  "cluster", "ggordiplots", "openxlsx", "svglite", "glmmTMB"
)

install.packages(required_packages)
```

### Initializing renv (For First-Time Contributors)

If you need to create a new `renv.lock` file (e.g., after updating packages):

```r
# In RStudio, open data-analysis.Rproj, then:
source("init_renv.R")

# Or manually:
renv::init()
renv::snapshot()
```

### Troubleshooting

**Issue**: `glmmTMB` installation fails
**Solution**: On Windows, you may need Rtools. Download from: https://cran.r-project.org/bin/windows/Rtools/

**Issue**: `ggstatsplot` has dependency conflicts
**Solution**: Update all packages first with `update.packages(ask = FALSE)`

**Issue**: Packages install but fail to load
**Solution**: Restart R session and try loading packages individually to identify the problematic one

## Data Sources

The raw data is located in `data/raw/publication_data/`, reduced to what the analysis reads
(columns and coding: `data/raw/publication_data/README.md`):
- `specimen_data.xlsx` - Specimen-level data (40,090 specimens of the 54 valid samples; site codes, year and day of the year)
- `biomass_data.xlsx` - Biomass per sample and size fraction (371 weighed size fractions)
- `model_performance.xlsx` - Test-set performance per classification model and per taxon (two sheets)
- `climate_data.xlsx` - Daily precipitation and temperature per site and sampling day (GeoSphere SPARTACUS v2.1)

## Analysis

The main analysis is performed in `scripts/Data_Analysis.Rmd`.

### Key Analyses:
- Arthropod diversity and abundance gradients
- Eco-treatment comparisons (Hedges vs. Control)
- Temperature effects on arthropod populations
- Focus on small arthropods (<2mm) representing 80% of diversity

## Usage

1. Open `data-analysis.Rproj` in RStudio
2. Run `scripts/Data_Analysis.Rmd` to perform the analysis (main analysis without sample 180; sensitivity analysis: see below)
3. Results will be generated in the `output/` directory

## Main analysis and sensitivity analysis (version 1.1.0)

Sample 180 was collected over two days instead of 24 hours. Since version 1.1.0
the main analysis excludes it (53 samples); a sensitivity analysis repeats the analysis with all
54 samples, as in the submitted version. Both come from the same `scripts/Data_Analysis.Rmd`,
controlled by one parameter in its YAML header:

```yaml
params:
  exclude_sample: "180"   # zuordnung_id of the excluded sample; "" = keep all samples
```

The parameter removes the specimens of that sample right after `specimen_data.xlsx` is read; with
an empty value the imported data stay unchanged.

```r
# main analysis (53 samples, default)
rmarkdown::render("scripts/Data_Analysis.Rmd")
# sensitivity analysis (54 samples, sample 180 included)
rmarkdown::render("scripts/Data_Analysis.Rmd", params = list(exclude_sample = ""),
                  output_file = "Data_Analysis_sensitivity_with_180.html")
```

Both renders write their tables and figures into the same folders (`scripts/`, `output/`); run them
one after the other into separate copies of `analysis/`, or use the Docker run below, which keeps
each run in its own folder.

## Reproduction in Docker

`reproduction/` renders both analyses in Docker (R 4.5.0, package versions of `renv.lock`,
restored into a Docker volume; nothing is installed on the host):

```bash
bash analysis/reproduction/run_all.sh <OUT>     # OUT outside the repository
```

| File | Purpose |
|---|---|
| `reproduction/Dockerfile` | `rocker/r-ver:4.5.0` plus system libraries, pandoc and patch |
| `reproduction/restore.R` | restores `renv.lock` into the volume (CRAN snapshot 2025-11-19 for 15 packages the Rmd loads but the lockfile does not pin) |
| `reproduction/run.sh` | renders the Rmd in a scratch copy inside the container, with `EXCLUDE_SAMPLE` as the parameter |
| `reproduction/reproduction.patch` | changes applied to the scratch copy only, each marked `# REPRODUCTION:`: adapts the Rmd to the reduced data files (sheets and columns selected by name instead of position, date rebuilt from year and day of the year), reads the prepared sheet of `biomass_data.xlsx` (the Rmd expects two raw sheets), reads the climate values from `climate_data.xlsx` (the SPARTACUS query of the Rmd needs the site coordinates, which are not archived; it stays switched off), keeps the confidence-interval columns of `emmeans` for glmmTMB 1.1.13, and uses a placeholder for the path diagram (Fig. 3h, not in the archive) |
| `reproduction/key_numbers.R` | reads the key numbers of a finished run from its `session.RData` into `key_numbers.csv` |
| `reproduction/make_table_s14.py` | Table S14 (orders and families of the 53 samples) from `specimen_data.xlsx` |
| `reproduction/make_table_s15.py` | Table S15 (main analysis next to the sensitivity analysis) from the `key_numbers.csv` of both runs |
| `reproduction/tables/` | the generated Tables S14 and S15 |

`run_all.sh` writes `<OUT>/main_without_180/` and `<OUT>/sensitivity_with_180/` (rendered report,
Word tables, figures, `output/tables/*.csv`, `session.RData`, `sessionInfo.txt`,
`key_numbers.csv`). It needs network access only for the package restore (CRAN snapshot); the renders
read the climate values from `climate_data.xlsx`. A run takes a few minutes per
analysis (about 2 to 3 minutes on the authors' machine) plus the one-time package restore. Clean up afterwards with
`docker rmi iamp-analysis-r:4.5.0` and `docker volume rm iamp-analysis-rlib`.

Tables S14 and S15 (Python 3.9 or newer, standard library only; from the repository root):

```bash
python analysis/reproduction/make_table_s14.py --check
python analysis/reproduction/make_table_s15.py <OUT>/main_without_180 <OUT>/sensitivity_with_180 --check
```

`--check` compares the generated table with the file in `reproduction/tables/` and exits with 1 on
any difference; without `--check` the file is written. Permutation P-values of Tables S8 to S11
vary slightly between runs because those permutation tests use no fixed seed (the indicator
analysis, Table S12, sets one), so a fresh run may differ from Table S15 in those cells.

## Related Repository

This repository contains the statistical analysis. The machine learning pipeline is in a separate repository:
- `../arthropod-classification-pipeline/` - AI-based arthropod classification pipeline
