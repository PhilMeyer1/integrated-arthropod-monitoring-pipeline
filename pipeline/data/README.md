# Data Directory

This directory contains all data files for the arthropod classification pipeline.

## Structure

```
data/
├── raw/              # Raw composite images (not tracked in git)
├── processed/        # Processed/extracted specimens (not tracked)
├── models/           # Pre-trained and trained models
│   ├── detection/    # YOLO detection models
│   ├── segmentation/ # YOLO segmentation models
│   └── classification/ # Hierarchical classification models
└── example/          # Example dataset for testing
    ├── images/       # Sample composite images
    └── metadata.csv  # Example metadata
```

## Download Models

Pre-trained models are not included in the repository due to size.

Download them using:
```bash
python scripts/download_models.py
```

This will download:
- YOLOv8 detection model → `models/detection/`
- YOLOv8 segmentation model → `models/segmentation/`
- YOLOv11 classification base model → `models/classification/`
- Example trained taxonomic model → `models/classification/`

## Example Data

A small example dataset is included in `example/` for testing the pipeline.
See `example/README.md` for details.

## Data Availability

Full dataset available at: [Zenodo DOI - to be added after publication]
