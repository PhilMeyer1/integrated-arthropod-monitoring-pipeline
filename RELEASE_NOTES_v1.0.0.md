# Release v1.0.0 - Initial Publication Release

**Release Date**: 2025-01-20

## Publication Reference

This release is associated with the manuscript:

> **Meyer, P., Scharnhorst, V., Lechner, M., Haslinger, H., Gierus, M., & Meimberg, H.** (2025).
> *Integrated arthropod monitoring pipeline detects size-dependent responses.*
> Manuscript in review.

---

## Overview

This is the first public release of the Integrated Arthropod Monitoring Pipeline, providing a complete workflow for automated detection, segmentation, and classification of arthropod specimens from composite images.

## Key Features

### 🔬 Automated Image Processing
- **Detection**: YOLOv8-based models for size-specific arthropod detection
- **Segmentation**: Precise instance segmentation for specimen extraction
- **Classification**: Hierarchical taxonomic classification using YOLOv11

### 📊 Complete Analysis Pipeline
- Size-based sorting integration
- Hierarchical AI classification with confidence calibration
- Statistical analysis for diversity, abundance, and biomass
- Community composition analysis (PERMANOVA, beta-diversity)

### 📈 Reproducible Research
- Complete source code for all analyses
- Specimen-level data (46,012 specimens)
- Biomass measurements (378 measurements)
- Model performance metrics (73 taxa)

---

## What's Included

### Pipeline Components
- **Source Code**: Complete Python pipeline in `pipeline/`
  - Detection, segmentation, and extraction modules
  - Hierarchical classification system
  - Database integration with SQLAlchemy
  - Export functionality for results

- **Analysis Scripts**: R Markdown scripts in `analysis/`
  - GLMM models for ecological metrics
  - Community composition analyses
  - Figure and table generation
  - Size-dependent response quantification

### Data
- **Publication Data**: All specimen-level data in `analysis/data/raw/publication_data/`
  - `specimen_data.xlsx` (46,012 specimens)
  - `biomass_data.xlsx` (378 measurements)
  - `model_performance.xlsx` (73 taxa)

### Models
- **Detection & Segmentation Models**: Available separately on Zenodo
  - `detect_k1.pt` - Detection model for <1mm specimens (22 MB)
  - `detect_others.pt` - Detection model for 1-10mm specimens (6.1 MB)
  - `segment_k1.pt` - Segmentation model for <1mm specimens (23 MB)
  - `segment_others.pt` - Segmentation model for 1-10mm specimens (23 MB)
  - **Zenodo DOI**: 10.5281/zenodo.17661921

- **Classification Models**: Available upon reasonable request for academic research
  - Contact: philipp.meyer@boku.ac.at

### Documentation
- Complete README with installation and usage instructions
- Architecture documentation in `pipeline/docs/`
- Example notebooks and scripts in `pipeline/examples/`
- API documentation for all modules

---

## System Requirements

### Pipeline
- **OS**: Windows, macOS, or Linux
- **Python**: 3.8 or higher
- **RAM**: 8 GB minimum (16 GB recommended)
- **GPU**: CUDA-compatible GPU recommended (CPU also supported)
- **Dependencies**: See `pipeline/requirements.txt`

### Analysis
- **R**: 4.2 or higher
- **RStudio**: Recommended for .Rmd notebooks
- **RAM**: 8 GB minimum
- **Dependencies**: Managed via `renv`

---

## Installation

```bash
# Clone the repository
git clone https://github.com/PhilMeyer1/integrated-arthropod-monitoring-pipeline.git
cd integrated-arthropod-monitoring-pipeline

# Install Python dependencies
cd pipeline
pip install -r requirements.txt

# Download models from Zenodo
# Visit: https://doi.org/10.5281/zenodo.17661921
# Extract to: pipeline/data/models/

# Test installation
python examples/test_core_components.py
# Expected: 5/5 tests passed
```

---

## Quick Start

### Process Example Images

```bash
cd pipeline
python examples/02_process_images.py \
    --metadata data/examples/metadata_example.csv \
    --output-dir ./output/specimens
```

### Run Statistical Analysis

```R
# Open in RStudio
# analysis/scripts/Data_Analysis.Rmd
# Knit to generate all figures and tables
```

---

## Key Findings from the Publication

- **Small taxa (<2 mm) show strongest environmental response** (+8% abundance per mm decrease)
- **Hedgerows increase arthropod abundance by 60%** (size-dependent effect)
- **Community restructuring via nestedness**, not turnover
- **18 indicator taxa** associated with habitat enhancements (all small-bodied)
- **Size-biased sampling** obscures ecological patterns in conventional methods

---

## Changes Since Development

This is the initial public release. All development was conducted privately at BOKU University, Vienna.

### Core Features
- ✅ Detection and segmentation pipeline
- ✅ Hierarchical classification system
- ✅ Database integration
- ✅ Statistical analysis scripts
- ✅ Publication data included
- ✅ Complete documentation

---

## Citation

### Software
```bibtex
@software{Meyer2025code,
  author={Meyer, Philipp and Scharnhorst, Victor and Lechner, Michael and
          Haslinger, Hanna and Gierus, Martin and Meimberg, Harald},
  title={Integrated Arthropod Monitoring Pipeline - Source Code},
  year={2025},
  version={1.0.0},
  publisher={Zenodo},
  doi={10.5281/zenodo.17661710},
  url={https://github.com/PhilMeyer1/integrated-arthropod-monitoring-pipeline}
}
```

### Models
```bibtex
@dataset{Meyer2025models,
  author={Meyer, Philipp and Scharnhorst, Victor and Lechner, Michael and
          Haslinger, Hanna and Gierus, Martin and Meimberg, Harald},
  title={Detection and Segmentation Models for Arthropod Monitoring Pipeline},
  year={2025},
  version={1.0.0},
  publisher={Zenodo},
  doi={10.5281/zenodo.17661921}
}
```

### Publication
```bibtex
@article{Meyer2025,
  title={Integrated arthropod monitoring pipeline detects size-dependent responses},
  author={Meyer, Philipp and Scharnhorst, Victor and Lechner, Michael and
          Haslinger, Hanna and Gierus, Martin and Meimberg, Harald},
  year={2025},
  note={Manuscript in review}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note**: Detection and segmentation models are provided under MIT License for academic research. Classification models are available upon request. Commercial use of any models requires separate licensing.

---

## Known Limitations

1. **Classification Models**: Not included in public release (available upon request)
2. **Large Files**: Example composite images and models hosted separately on Zenodo
3. **GPU Recommended**: While CPU processing is supported, GPU significantly improves performance
4. **Windows Line Endings**: Some files may have CRLF line endings due to development on Windows

---

## Support

For questions, issues, or feature requests:
- **GitHub Issues**: [github.com/PhilMeyer1/integrated-arthropod-monitoring-pipeline/issues](https://github.com/PhilMeyer1/integrated-arthropod-monitoring-pipeline/issues)
- **Email**: philipp.meyer@boku.ac.at
- **Institution**: University of Natural Resources and Life Sciences (BOKU), Vienna, Austria

---

## Acknowledgments

- **Catalogue of Life** for taxonomic data
- **Ultralytics** for YOLO implementation
- **Federal Ministry of Agriculture and Forestry, Climate and Environmental Protection, Regions and Water Management of Austria** for funding
- **BOKU University** for institutional support

---

**Contributors**: Philipp Meyer, Victor Scharnhorst, Michael Lechner, Hanna Haslinger, Martin Gierus, Harald Meimberg

**Corresponding Author**: Philipp Meyer (philipp.meyer@boku.ac.at)
