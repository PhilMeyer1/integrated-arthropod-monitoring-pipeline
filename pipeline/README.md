# Arthropod Classification Pipeline

**Automated hierarchical classification of arthropod specimens using deep learning.**

This repository contains the complete pipeline for processing arthropod specimens from high-resolution composite images to taxonomic classification, as described in the manuscript.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/1088692587.svg)](https://doi.org/10.5281/zenodo.17512686)

---

## 📋 For Reviewers

**Quick test (5 minutes)**: See [QUICKTEST.md](QUICKTEST.md)
**Full test guide**: See [REVIEWER_GUIDE.md](REVIEWER_GUIDE.md)
**Test results**: See [TEST_REPORT.md](TEST_REPORT.md)

**TL;DR**: `pip install -r requirements.txt` → `python scripts/01_setup_database.py --example` → `python scripts/test_core_components.py` → Expect "5/5 tests passed"

---

## Overview

This pipeline processes arthropod specimens through a multi-stage workflow:

1. **Database Setup**: Initialize SQLite database with taxonomic hierarchy
2. **Image Processing**: Detection, segmentation, and extraction of individual specimens
3. **Model Training**: Train hierarchical YOLO classification models
4. **Classification**: Automated taxonomic classification with confidence tracking
5. **Export & Analysis**: Statistical analysis and publication-ready exports

### Key Features

- 🗄️ **SQLite-based** - Fully reproducible with embedded database
- 🌳 **Hierarchical classification** - Following taxonomic hierarchy (Catalogue of Life)
- 📊 **Methods-aligned** - All parameters match publication methods
- ⚡ **GPU acceleration** - CUDA support for fast processing
- 🔧 **CLI-based** - No GUI dependencies, fully scriptable
- 📖 **Comprehensive documentation** - See [QUICKSTART.md](QUICKSTART.md) for 10-minute demo

## Quick Start (10 Minutes)

### Installation

```bash
# Clone repository
git clone https://github.com/PhilMeyer1/arthropod-classification-pipeline.git
cd arthropod-classification-pipeline

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Demo Pipeline

Run the complete pipeline with example data:

```bash
# Step 1: Initialize database with taxonomy
python scripts/01_setup_database.py --example

# Step 2: Process composite images (optional - requires images)
python scripts/02_process_images.py --input data/examples/

# Step 3: Train classification models
python scripts/03_train_models.py \
    --set-number 1 \
    --min-images 7 \
    --epochs 50

# Step 4: Classify specimens
python scripts/04_classify.py \
    --set-number 1 \
    --batch-size 200

# Step 5: Export results
python scripts/05_export_results.py \
    --set-number 1 \
    --format excel \
    --output exports/results.xlsx
```

**For detailed instructions**, see [QUICKSTART.md](QUICKSTART.md).

## System Requirements

### Minimum
- **OS**: Windows, macOS, or Linux
- **Python**: 3.8 or higher
- **RAM**: 8 GB
- **Disk**: 5 GB free space
- **CPU**: Multi-core processor

### Recommended
- **RAM**: 16 GB or more
- **GPU**: CUDA-compatible GPU (NVIDIA RTX series)
- **Disk**: SSD with 10+ GB free space

## Trained Models and Example Data

### Detection and Segmentation Models

The pipeline uses specialized YOLO models for arthropod detection and segmentation:

**Detection Models** (`data/models/detection/`)
- `detect_k1.pt` (22 MB) - For <1mm specimens (k1 fraction)
  - Confidence: 0.01, IOU: 0.9
- `detect_others.pt` (6.1 MB) - For 1-10mm specimens (fractions 1-7)
  - Confidence: 0.01, IOU: 0.9

**Segmentation Models** (`data/models/segmentation/`)
- `segment_k1.pt` (23 MB) - For <1mm specimens
  - Confidence: 0.05, IOU: 0.1
- `segment_others.pt` (23 MB) - For 1-10mm specimens
  - Confidence: 0.05, IOU: 0.1

**Architecture**: YOLOv8n (detection) and YOLOv8n-seg (segmentation)

### Example Data

**Composite Images** (`data/examples/`)
- `composite_S001_k1.jpg` (42 MB) - k1 fraction, ~879 specimens
- `composite_S002_1.5mm.jpg` (48 MB) - 1-2mm fraction, ~245 specimens
- `metadata_example.csv` - Batch processing metadata

### Classification Models

The hierarchical classification models (YOLOv11) are **not included** in this public release but are available upon reasonable request for academic research purposes. Contact: philipp.meyer@boku.ac.at

### Download Models and Data

Models and example images are available on Zenodo:
- **DOI**: [10.5281/zenodo.17661921](https://doi.org/10.5281/zenodo.17661921)
- **Size**: ~106 MB (compressed)
- **Contents**: All detection/segmentation models + example images

```bash
# Download and extract to pipeline directory
wget https://zenodo.org/records/17661921/files/detection-classification-models-example-images-v1.0.0.zip
unzip detection-classification-models-example-images-v1.0.0.zip -d data/
```

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SQLite Database                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐   │
│  │ Taxonomy │  │Specimens │  │  Models  │  │ Results │   │
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘   │
└─────────────────────────────────────────────────────────────┘
         │              │               │              │
         ▼              ▼               ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────┐ ┌──────────┐
│  01_setup_   │ │ 02_process_  │ │03_train_ │ │04_classify│
│  database.py │ │  images.py   │ │models.py │ │   .py    │
└──────────────┘ └──────────────┘ └──────────┘ └──────────┘
                                                      │
                                                      ▼
                                              ┌──────────────┐
                                              │ 05_export_   │
                                              │  results.py  │
                                              └──────────────┘
```

## Repository Structure

```
arthropod-classification-pipeline/
├── scripts/                    # CLI scripts for pipeline execution
│   ├── 01_setup_database.py   # Initialize SQLite database
│   ├── 02_process_images.py   # Detection & extraction
│   ├── 03_train_models.py     # Train YOLO models
│   ├── 04_classify.py         # Hierarchical classification
│   └── 05_export_results.py   # Export to Excel/CSV
│
├── src/                        # Core pipeline code
│   ├── database/               # SQLite ORM and utilities
│   ├── classification/         # Training & inference
│   ├── image_processing/       # Detection & segmentation
│   ├── export/                 # Export utilities
│   └── utils/                  # Helper functions
│
├── data/                       # Data directory
│   ├── arthropod_pipeline.db  # SQLite database
│   ├── taxonomy/              # Catalogue of Life data
│   ├── models/                # Trained YOLO models
│   └── examples/              # Example data
│
├── config/                     # Configuration files
│   └── default_config.yaml    # Pipeline parameters
│
├── tests/                      # Unit and integration tests
├── docs/                       # Additional documentation
├── QUICKSTART.md              # 10-minute quickstart guide
├── TEST_REPORT.md             # Comprehensive test results
└── requirements.txt           # Python dependencies
```

## Methods Correspondence

This pipeline implements the methods described in the manuscript:

| Pipeline Step | Methods Section | Key Parameters |
|--------------|-----------------|----------------|
| `01_setup_database.py` | Data Management | SQLite, Catalogue of Life |
| `02_process_images.py` | Image Analysis | Tiled YOLO, NMS IoU=0.2 |
| `03_train_models.py` | Model Training | YOLOv11, 512×512, 50 epochs |
| `04_classify.py` | Inference | Batch=200, Hierarchical |
| `05_export_results.py` | Data Export | Excel/CSV, Statistics |

**All parameters match the publication exactly.**

## Database Schema

The pipeline uses SQLite with the following main tables:

- **`taxa`** - Taxonomic hierarchy (Catalogue of Life)
- **`single_images`** - Individual specimen images with metadata
- **`models`** - Trained model metadata and versioning
- **`inference_results`** - Classification results with confidence scores
- **`size_fractions`** - Size-based specimen groupings

For full schema documentation, see `docs/DATABASE_SCHEMA.md`.

## Training Parameters

Training uses the exact parameters from the Methods section:

```yaml
Model: YOLOv11 (yolo11m-cls.pt)
Image Size: 512×512 pixels
Epochs: 50
Batch Size: Auto
Data Split: 72% train / 18% val / 10% test
Augmentation:
  - Rotations: 90°, 180°, 270°
  - HSV: h=0.025, s=0.3, v=0.3
Special Classes:
  - "Other" class for low-frequency taxa
  - "Debris" class for non-specimens
```

## Classification Workflow

Hierarchical inference process:

1. Load all models for specified set number
2. Start with top-level model (e.g., Insecta)
3. For each specimen:
   - Classify with current level model
   - Record prediction and confidence
   - If child model exists, apply it
4. Repeat until reaching leaf level
5. Save results to database

**Batch processing**: 200 images per batch (Methods-compliant)

## Testing

The pipeline has been comprehensively tested:

```bash
# Run core component tests
python scripts/test_core_components.py

# Expected output: 5/5 tests passed
```

See [TEST_REPORT.md](TEST_REPORT.md) for full test results.

## Data Availability

### Example Data (Included)
- 11 taxa from Catalogue of Life
- Example database schema
- Mock specimen images for testing

### Full Dataset (Upon Request)
- 12,666 manually classified specimens
- Complete SQLite database (~100 MB)
- Trained models for all taxonomic levels

Contact corresponding author for full dataset access.

## Citation

If you use this pipeline, please cite:

```bibtex
@article{Meyer2025,
  title={Integrated arthropod monitoring pipeline detects size-dependent responses},
  author={Meyer, Philipp and Scharnhorst, Victor and Lechner, Michael and
          Haslinger, Hanna and Gierus, Martin and Meimberg, Harald},
  year={2025},
  note={Manuscript in review}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Catalogue of Life for taxonomic data
- Ultralytics for YOLO implementation
- SQLAlchemy for database ORM

## Support

- **Issues**: [GitHub Issues](https://github.com/PhilMeyer1/arthropod-classification-pipeline/issues)
- **Documentation**: See `docs/` folder
- **Publication**: Manuscript in review (link will be added upon publication)

## Contributing

This repository is primarily for publication and reproducibility. For contributions or collaborations, please contact the corresponding author.

## Version History

- **v1.0.0** (2025-10-27): Initial publication release
  - Complete SQLite-based pipeline
  - CLI scripts for all stages
  - Comprehensive test suite
  - Publication-ready code

---

**Developed for**: Scientific Publication
**Maintainer**: Philipp Meyer (philipp.meyer@boku.ac.at)
**Institution**: BOKU University, Vienna, Austria
**Last Updated**: 2025-01-03
