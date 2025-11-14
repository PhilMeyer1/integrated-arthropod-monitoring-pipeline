# Integrated Arthropod Monitoring Pipeline Detects Size-Dependent Responses

**Companion repository for Current Biology publication**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![R 4.2+](https://img.shields.io/badge/R-4.2+-blue.svg)](https://www.r-project.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17512687.svg)](https://doi.org/10.5281/zenodo.17512687)

---

## 📋 For Reviewers

**This repository contains all code and data for reproducing the analyses in:**

> **Meyer, P., Scharnhorst, V., Lechner, M., Haslinger, H., Gierus, M., & Meimberg, H.** (2025).
> *Integrated arthropod monitoring pipeline detects size-dependent responses.*
> Current Biology, XX(X), XXX-XXX.

**Quick Start:**
1. **Pipeline** (AI Classification): See [`pipeline/README.md`](pipeline/README.md)
2. **Analysis** (Statistics): See [`analysis/README.md`](analysis/README.md)

**Complete workflow test (~10 minutes):**
```bash
# Clone repository
git clone https://github.com/PhilMeyer1/integrated-arthropod-monitoring-pipeline.git
cd integrated-arthropod-monitoring-pipeline

# Test classification pipeline
cd pipeline
pip install -r requirements.txt
python scripts/test_core_components.py  # Expect: 5/5 tests passed

# Test statistical analysis
cd ../analysis
# Open scripts/Data_Analysis.Rmd in RStudio
```

---

## Repository Structure

```
integrated-arthropod-monitoring-pipeline/
│
├── pipeline/                   # AI-based arthropod classification
│   ├── src/                   # Core pipeline code
│   ├── scripts/               # CLI scripts for workflow steps
│   ├── data/                  # Data directories (models, examples)
│   ├── docs/                  # Technical documentation
│   ├── requirements.txt       # Python dependencies
│   └── README.md              # Detailed pipeline documentation
│
├── analysis/                  # Statistical analysis & publication data
│   ├── data/raw/              # Publication datasets (46,012 specimens)
│   ├── scripts/               # R Markdown analysis scripts
│   ├── output/                # Generated tables & figures
│   └── README.md              # Analysis documentation
│
├── README.md                  # This file
├── CITATION.cff               # Citation information
└── LICENSE                    # MIT License
```

---

## Overview

This repository provides:

1. **Automated Classification Pipeline** ([`pipeline/`](pipeline/))
   - Size-based sorting integration
   - High-resolution imaging workflow
   - Hierarchical AI classification (YOLOv11)
   - Confidence threshold calibration
   - Non-destructive, specimen-preserving approach

2. **Statistical Analysis** ([`analysis/`](analysis/))
   - GLMM models for diversity, abundance, biomass
   - PERMANOVA for community composition
   - Beta-diversity partitioning
   - Indicator species analysis
   - Size-dependent response quantification

---

## Key Findings

Our integrated approach reveals:

- **Small taxa (<2 mm) show strongest environmental response** (+8% abundance per mm decrease)
- **Hedgerows increase arthropod abundance by 60%** (size-dependent effect)
- **Community restructuring via nestedness**, not turnover
- **18 indicator taxa** associated with habitat enhancements (all small-bodied)
- **Size-biased sampling** obscures ecological patterns in conventional methods

---

## Citation

### Paper
```bibtex
@article{Meyer2025,
  title={Integrated arthropod monitoring pipeline detects size-dependent responses},
  author={Meyer, Philipp and Scharnhorst, Victor and Lechner, Michael and
          Haslinger, Hanna and Gierus, Martin and Meimberg, Harald},
  journal={Current Biology},
  year={2025},
  volume={XX},
  pages={XXX--XXX},
  doi={10.1016/j.cub.2025.XX.XXX}
}
```

### Code & Data
```bibtex
@software{Meyer2025code,
  author={Meyer, Philipp and Scharnhorst, Victor and Lechner, Michael and
          Haslinger, Hanna and Gierus, Martin and Meimberg, Harald},
  title={Integrated Arthropod Monitoring Pipeline - Code and Data},
  year={2025},
  publisher={GitHub},
  url={https://github.com/PhilMeyer1/integrated-arthropod-monitoring-pipeline},
  doi={10.5281/zenodo.17512687}
}
```

---

## Quick Links

### Documentation
- **Pipeline Documentation**: [`pipeline/README.md`](pipeline/README.md)
- **Pipeline Architecture**: [`pipeline/docs/ARCHITECTURE.md`](pipeline/docs/ARCHITECTURE.md)
- **Analysis Documentation**: [`analysis/README.md`](analysis/README.md)

### Data
- **Specimen Data**: [`analysis/data/raw/publication_data/specimen_data.xlsx`](analysis/data/raw/publication_data/specimen_data.xlsx) (46,012 specimens)
- **Biomass Data**: [`analysis/data/raw/publication_data/biomass_data.xlsx`](analysis/data/raw/publication_data/biomass_data.xlsx) (378 measurements from 54 samples × 7 size fractions)
- **Model Performance**: [`analysis/data/raw/publication_data/model_performance.xlsx`](analysis/data/raw/publication_data/model_performance.xlsx) (73 taxa)

---

## System Requirements

### Pipeline
- **OS**: Windows, macOS, or Linux
- **Python**: 3.8 or higher
- **RAM**: 8 GB minimum (16 GB recommended)
- **GPU**: CUDA-compatible GPU recommended (CPU also supported)

### Analysis
- **R**: 4.2 or higher
- **RStudio**: Recommended for .Rmd notebooks
- **RAM**: 8 GB minimum

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Catalogue of Life** for taxonomic data
- **Ultralytics** for YOLO implementation
- **Federal Ministry of Agriculture and Forestry, Climate and Environmental Protection, Regions and Water Management of Austria** for funding

---

## Contact

**Corresponding Author**: Philipp Meyer
**Email**: philipp.meyer@boku.ac.at
**Institution**: BOKU University, Vienna, Austria

---

## Reproducibility Statement

All analyses are fully reproducible using the code and data provided in this repository. The pipeline processes specimens from raw images to taxonomic classifications, and the statistical analysis scripts generate all figures and tables in the manuscript.

**Last Updated**: 2025-01-14
