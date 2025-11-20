# Trained Models

This directory contains the trained YOLO models used in the publication for arthropod detection, segmentation, and classification.

## Directory Structure

```
models/
├── detection/
│   ├── detect_k1.pt              # Detection model for <1mm specimens (22 MB)
│   └── detect_others.pt          # Detection model for 1-10mm specimens (6.1 MB)
├── segmentation/
│   ├── segment_k1.pt             # Segmentation model for <1mm specimens (23 MB)
│   └── segment_others.pt         # Segmentation model for 1-10mm specimens (23 MB)
└── classification/
    ├── 1.pt                       # Classification model (Level 1)
    └── 2.pt                       # Classification model (Level 2)
```

## Model Details

### Detection Models

**Purpose**: Detect individual arthropod specimens on composite images

**Architecture**: YOLOv8-based detection models

**Models**:

#### `detect_k1.pt` (22 MB)
- **Target**: Very small arthropods (<1mm, k1 fraction)
- **Optimized**: High sensitivity for tiny specimens
- **Training**: Specialized dataset of <1mm specimens

#### `detect_others.pt` (6.1 MB)
- **Target**: Medium to large arthropods (1-10mm)
- **Optimized**: Balanced detection across size fractions 1, 2, 3, 5, 7
- **Training**: Combined dataset of 1-10mm specimens

**Training Details**:
- Trained on high-resolution composite images from the study
- Detects class "Area" (arthropod regions)
- Confidence threshold: 0.01 (very low for maximum sensitivity)
- IOU threshold: 0.9 (high for detection phase)

**Usage**:
```python
from ultralytics import YOLO
# For k1 specimens
model = YOLO('data/models/detection/detect_k1.pt')
# For other size fractions
model = YOLO('data/models/detection/detect_others.pt')
results = model.predict('composite_image.jpg', conf=0.01, iou=0.9)
```

### Segmentation Models

**Purpose**: Segment individual arthropod specimens for precise extraction

**Architecture**: YOLOv8-based segmentation models

**Models**:

#### `segment_k1.pt` (23 MB)
- **Target**: Very small arthropods (<1mm, k1 fraction)
- **Optimized**: Precise masks for tiny specimens
- **Training**: Specialized dataset of <1mm specimens

#### `segment_others.pt` (23 MB)
- **Target**: Medium to large arthropods (1-10mm)
- **Optimized**: Accurate segmentation across size fractions 1, 2, 3, 5, 7
- **Training**: Combined dataset of 1-10mm specimens

**Training Details**:
- Trained on annotated arthropod specimens
- Generates pixel-level segmentation masks
- Segments class "Area" (arthropod regions)
- Confidence threshold: 0.05
- IOU threshold: 0.1 (low for segmentation NMS)
- Used in Step 2 of the pipeline for precise specimen extraction

**Usage**:
```python
from ultralytics import YOLO
# For k1 specimens
model = YOLO('data/models/segmentation/segment_k1.pt')
# For other size fractions
model = YOLO('data/models/segmentation/segment_others.pt')
results = model.predict('composite_image.jpg', conf=0.05, iou=0.1)
```

### Classification Models: `1.pt`, `2.pt`

**Purpose**: Hierarchical taxonomic classification of arthropod specimens

**Architecture**: YOLOv11m-cls based classification models

**Training Details**:
- Hierarchical structure following Catalogue of Life taxonomy
- Trained with 72%/18%/10% train/val/test split
- Data augmentation: 90°, 180°, 270° rotations
- 50 epochs, 512×512 image size
- HSV augmentation enabled

**Note**: The classification models in this repository (1.pt, 2.pt) are example models demonstrating the pipeline structure. The actual models used in the study were trained on the complete dataset and can be obtained from the lead contact upon reasonable request for academic research purposes.

## Model Performance

Performance metrics for the models are documented in the publication and in:
- `analysis/data/raw/publication_data/model_performance.xlsx`

## Reproducing the Models

To train your own models using the pipeline:

```bash
# 1. Setup database with taxonomy
python scripts/01_setup_database.py --example

# 2. Process your images
python scripts/02_process_images.py \
    --metadata data/examples/metadata_example.csv \
    --output-dir ./output/processed

# 3. Train classification models
python scripts/03_train_models.py \
    --set-number 1 \
    --min-images 7 \
    --epochs 50
```

## License

These models are provided under the MIT License for academic and research purposes. Commercial use requires separate licensing.

## Citation

If you use these models in your research, please cite:

```bibtex
@article{Meyer2025,
  title={Integrated arthropod monitoring pipeline detects size-dependent responses},
  author={Meyer, Philipp and Scharnhorst, Victor and Lechner, Michael and
          Haslinger, Hanna and Gierus, Martin and Meimberg, Harald},
  journal={Current Biology},
  year={2025},
  doi={10.5281/zenodo.17512687}
}
```

## Contact

For questions about the models or requests for the full trained models:

**Philipp Meyer**
Email: philipp.meyer@boku.ac.at
Institution: BOKU University, Vienna, Austria
