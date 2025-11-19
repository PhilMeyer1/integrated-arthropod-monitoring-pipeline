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
