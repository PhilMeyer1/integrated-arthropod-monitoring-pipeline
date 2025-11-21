# Integrated Arthropod Monitoring Pipeline Detects Size-Dependent Responses

**Automated arthropod monitoring and classification pipeline**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![R 4.2+](https://img.shields.io/badge/R-4.2+-blue.svg)](https://www.r-project.org/)
[![Code DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17661710.svg)](https://doi.org/10.5281/zenodo.17661710)
[![Models DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17661921.svg)](https://doi.org/10.5281/zenodo.17661921)

---

## About

This repository provides a complete, reproducible pipeline for automated arthropod detection, segmentation, and hierarchical classification from composite images. The system integrates size-based sorting with deep learning (YOLOv8/YOLOv11) to enable comprehensive biodiversity monitoring while preserving specimens for downstream analyses. Our integrated approach reveals size-dependent ecological responses that conventional monitoring methods miss.

**Reference:** Meyer, P., Scharnhorst, V., Lechner, M., Haslinger, H., Gierus, M., & Meimberg, H. (2025). *Integrated arthropod monitoring pipeline detects size-dependent responses.*

---

## Key Findings

Our integrated approach reveals:

- **Small taxa (<2 mm) show strongest environmental response** (+8% abundance per mm decrease)
- **Hedgerows increase arthropod abundance by 60%** (size-dependent effect)
- **Community restructuring via nestedness**, not turnover
- **18 indicator taxa** associated with habitat enhancements (all small-bodied)
- **Size-biased sampling** obscures ecological patterns in conventional methods

---

## Quick Start

### System Requirements

**Pipeline (Python)**
- **OS**: Windows, macOS, or Linux
- **Python**: 3.8 or higher
- **RAM**: 8 GB minimum (16 GB recommended)
- **GPU**: CUDA-compatible GPU recommended (CPU also supported)

**Analysis (R)**
- **R**: 4.2 or higher
- **RStudio**: Recommended for .Rmd notebooks
- **RAM**: 8 GB minimum

### Installation & Testing

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
# Open scripts/Data_Analysis.Rmd in RStudio and knit
```

### Running the Pipeline

**1. AI Classification Pipeline** ([`pipeline/README.md`](pipeline/README.md))
   - Size-based sorting integration
   - High-resolution imaging workflow
   - Hierarchical AI classification (YOLOv11)
   - Confidence threshold calibration
   - Non-destructive, specimen-preserving approach

**2. Statistical Analysis** ([`analysis/README.md`](analysis/README.md))
   - GLMM models for diversity, abundance, biomass
   - PERMANOVA for community composition
   - Beta-diversity partitioning
   - Indicator species analysis
   - Size-dependent response quantification

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

## Documentation & Data

### Documentation
- **Pipeline Documentation**: [`pipeline/README.md`](pipeline/README.md)
- **Pipeline Architecture**: [`pipeline/docs/ARCHITECTURE.md`](pipeline/docs/ARCHITECTURE.md)
- **Analysis Documentation**: [`analysis/README.md`](analysis/README.md)

### Publication Data (Included in Repository)
- **Specimen Data**: [`analysis/data/raw/publication_data/specimen_data.xlsx`](analysis/data/raw/publication_data/specimen_data.xlsx) (46,012 specimens)
- **Biomass Data**: [`analysis/data/raw/publication_data/biomass_data.xlsx`](analysis/data/raw/publication_data/biomass_data.xlsx) (378 measurements from 54 samples × 7 size fractions)
- **Model Performance**: [`analysis/data/raw/publication_data/model_performance.xlsx`](analysis/data/raw/publication_data/model_performance.xlsx) (73 taxa)

---

## Data Availability

### Trained Models and Example Data
Detection and segmentation models (YOLOv8) and example composite images are available on Zenodo:
- **DOI**: [10.5281/zenodo.17661921](https://doi.org/10.5281/zenodo.17661921)
- **Contents**: Detection models (<1mm and others), segmentation models (<1mm and others), example composite images
- **Size**: ~106 MB (compressed)

### Classification Models
The hierarchical classification models are not included in this public release but are available upon reasonable request for academic research purposes. Contact: philipp.meyer@boku.ac.at

### Complete Source Code
All source code is archived on Zenodo:
- **DOI**: [10.5281/zenodo.17661710](https://doi.org/10.5281/zenodo.17661710)
- **GitHub**: [github.com/PhilMeyer1/integrated-arthropod-monitoring-pipeline](https://github.com/PhilMeyer1/integrated-arthropod-monitoring-pipeline)

---

## Citation

### Paper
```bibtex
@article{Meyer2025,
  title={Integrated arthropod monitoring pipeline detects size-dependent responses},
  author={Meyer, Philipp and Scharnhorst, Victor and Lechner, Michael and
          Haslinger, Hanna and Gierus, Martin and Meimberg, Harald},
  year={2025},
  note={Manuscript in review}
}
```

### Code
```bibtex
@software{Meyer2025code,
  author={Meyer, Philipp and Scharnhorst, Victor and Lechner, Michael and
          Haslinger, Hanna and Gierus, Martin and Meimberg, Harald},
  title={Integrated Arthropod Monitoring Pipeline - Source Code},
  year={2025},
  publisher={Zenodo},
  doi={10.5281/zenodo.17661710},
  url={https://github.com/PhilMeyer1/integrated-arthropod-monitoring-pipeline}
}
```

### Models & Example Data
```bibtex
@dataset{Meyer2025models,
  author={Meyer, Philipp and Scharnhorst, Victor and Lechner, Michael and
          Haslinger, Hanna and Gierus, Martin and Meimberg, Harald},
  title={Detection and Segmentation Models for Arthropod Monitoring Pipeline},
  year={2025},
  publisher={Zenodo},
  doi={10.5281/zenodo.17661921}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

**Corresponding Author**: Philipp Meyer
**Email**: philipp.meyer@boku.ac.at
**Institution**: BOKU University, Vienna, Austria

---

## Acknowledgments

- **Catalogue of Life** for taxonomic data
- **Ultralytics** for YOLO implementation
- **Federal Ministry of Agriculture and Forestry, Climate and Environmental Protection, Regions and Water Management of Austria** for funding

---

## Reproducibility Statement

All analyses are fully reproducible using the code and data provided in this repository. The pipeline processes specimens from raw images to taxonomic classifications, and the statistical analysis scripts generate all figures and tables in the manuscript.

**Last Updated**: 2025-01-20
