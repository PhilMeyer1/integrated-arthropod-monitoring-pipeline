# Data Analysis - Hecken für Hühner

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

The raw data is located in `data/raw/publication_data/`:
- `specimen_data.xlsx` - Specimen-level data (46,012 specimens)
- `biomass_data.xlsx` - Biomass measurements (378 records from 54 samples × 7 size fractions)
- `model_performance.xlsx` - Model performance metrics (73 taxa)

## Analysis

The main analysis is performed in `scripts/Data_Analysis.Rmd`.

### Key Analyses:
- Arthropod diversity and abundance gradients
- Eco-treatment comparisons (Hedges vs. Control)
- Temperature effects on arthropod populations
- Focus on small arthropods (<2mm) representing 80% of diversity

## Usage

1. Open `data-analysis.Rproj` in RStudio
2. Run `scripts/Data_Analysis.Rmd` to perform the analysis
3. Results will be generated in the `output/` directory

## Related Repository

This repository contains the statistical analysis. The machine learning pipeline is in a separate repository:
- `../arthropod-classification-pipeline/` - AI-based arthropod classification pipeline
