# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release preparation

## [1.0.0] - 2024-10-25

### Added
- **Image Processing Module** (`src/image_processing/`)
  - Tiled YOLO object detection with adaptive grid sizes
  - Instance segmentation for precise boundaries
  - Automated specimen extraction with metadata
  - Support for multiple size fractions (k1, 1-7mm, A)

- **Hierarchical Classification Module** (`src/classification/`)
  - Taxonomy hierarchy builder from Catalogue of Life
  - Training dataset creator with augmentation
  - YOLO model trainer for each taxonomic level
  - Hierarchical inference engine (top-down classification)
  - Confidence threshold optimizer

- **Export & Statistics Module** (`src/export/`)
  - Excel export with multi-sheet formatting
  - CSV export for R/Python analysis
  - Comprehensive statistics calculator
  - Publication-ready outputs

- **Configuration System**
  - YAML-based configuration with variable expansion
  - Environment variable support
  - Path resolution
  - Dot notation access

- **Database Support** (Optional)
  - SQLAlchemy models for PostgreSQL/SQLite
  - File-based alternative (CSV/JSON)
  - Abstraction layer for both modes

- **Utilities**
  - Centralized logging with progress tracking
  - Input validation functions
  - Image processing helpers

- **Scripts**
  - `02_process_images.py`: Process composite images
  - `03_train_models.py`: Train classification models (planned)
  - `04_classify.py`: Run hierarchical classification (planned)
  - `05_export_results.py`: Export results (planned)
  - `create_test_images.py`: Generate synthetic test data

- **Documentation**
  - Comprehensive README with quick start
  - Detailed installation guide (INSTALLATION.md)
  - Architecture overview (ARCHITECTURE.md)
  - Module-specific READMEs
  - Testing guide (TESTING.md)
  - Contributing guidelines (CONTRIBUTING.md)

- **Project Setup**
  - MIT License
  - Citation file (CITATION.cff)
  - Requirements.txt with GPU support
  - .gitignore for Python projects
  - Development tags documentation

### Configuration
- Default config: `config/default_config.yaml`
- Supports CUDA/CPU device selection
- Configurable detection/segmentation parameters
- Customizable training hyperparameters

### Dependencies
- Python 3.8+
- PyTorch 2.0+
- Ultralytics (YOLOv8/v11)
- OpenCV, Pillow, NumPy
- Pandas, Openpyxl
- SQLAlchemy (optional)

### Notes
- Tested on Ubuntu 20.04/22.04, Windows 10/11, macOS 12+
- GPU acceleration recommended (NVIDIA CUDA 11.8/12.1)
- Based on methods described in [Your Paper Citation]

## Version History

- **1.0.0** (2024-10-25): Initial public release
- **0.1.0** (Development): Internal pipeline version

---

## Release Notes Format

For future releases, use this format:

```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- New features

### Changed
- Changes to existing features

### Deprecated
- Features marked for removal

### Removed
- Removed features

### Fixed
- Bug fixes

### Security
- Security fixes
```

## Links

- [Repository](https://github.com/PhilMeyer1/arthropod-classification-pipeline)
- [Issues](https://github.com/PhilMeyer1/arthropod-classification-pipeline/issues)
- [Zenodo DOI](https://github.com/PhilMeyer1/arthropod-classification-pipeline) (will be added upon publication)
