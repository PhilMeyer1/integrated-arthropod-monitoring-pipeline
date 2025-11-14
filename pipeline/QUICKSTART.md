# Quick Start Guide

**Arthropod Classification Pipeline for Reviewers**

This guide will walk you through running the complete arthropod classification pipeline from raw images to taxonomic classifications. Expected time: **10-15 minutes** on a standard laptop.

---

## Prerequisites

### System Requirements
- **OS**: Windows, macOS, or Linux
- **Python**: 3.8 or higher
- **RAM**: 8 GB minimum (16 GB recommended)
- **Disk**: 5 GB free space
- **GPU**: CUDA-compatible GPU recommended (CPU-only mode also supported)

### Software Dependencies
```bash
# Check Python version
python --version  # Should be 3.8+

# Optional: Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

---

## Installation (2 minutes)

### 1. Clone Repository
```bash
git clone https://github.com/PhilMeyer1/arthropod-classification-pipeline.git
cd arthropod-classification-pipeline
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Key dependencies:**
- `ultralytics` - YOLO models (detection, segmentation, classification)
- `sqlalchemy` - Database ORM
- `opencv-python` - Image processing
- `pillow` - Image manipulation
- `pandas`, `openpyxl` - Data export

---

## Pipeline Overview

The pipeline consists of 5 main steps:

```
1. Setup Database       →  Initialize SQLite database with taxonomy
2. Process Images       →  Detection → Segmentation → Extraction
3. Train Models         →  Hierarchical classification models
4. Classify Specimens   →  Automated taxonomic classification
5. Export Results       →  Excel/CSV reports with statistics
```

---

## Step-by-Step Workflow

### Step 0: Initialize Database (30 seconds)

Create SQLite database with taxonomic hierarchy:

```bash
python scripts/01_setup_database.py --example
```

**What this does:**
- Creates `data/arthropod_pipeline.db` (SQLite database)
- Loads taxonomic hierarchy from Catalogue of Life
- Creates size fraction definitions
- Loads example project/location metadata

**Expected output:**
```
======================================================================
INITIALIZING ARTHROPOD CLASSIFICATION DATABASE
======================================================================
Database: C:\...\data\arthropod_pipeline.db
...
✓ Database created
✓ Size fractions: 8
✓ Taxa loaded: 11
✓ Example data loaded
```

---

### Step 1: Process Images (2-3 minutes)

Extract individual specimens from composite images:

```bash
python scripts/02_process_images.py \
    --input data/examples/composite_images/ \
    --output data/processed/specimens
```

**What this does:**
- **Detection**: Tiled YOLO detection to find arthropods
- **Segmentation**: Instance segmentation for precise boundaries
- **Extraction**: Crop individual specimens with background removal
- **Database**: Register specimens with bounding boxes and metadata

**Key parameters (from Methods):**
- Grid size: Adaptive (20x12 for <1mm, 10x8 for 1-2mm, etc.)
- NMS IoU threshold: 0.2
- Expansion ratio: Applied during extraction

**Expected output:**
```
Processing composite image: sample001_1mm.png
  Detected 45 specimens
  Segmented 42 specimens
  Extracted 42 individual images
  Saved to database
```

---

### Step 2: Manual Classification (External)

⚠️ **Note**: This step is typically done manually by taxonomists.

For the demo, example data includes pre-classified specimens.

In production:
1. Export specimens from database
2. Manually determine taxa (using microscopy, keys, etc.)
3. Import classifications back to database

---

### Step 3: Train Classification Models (3-5 minutes)

Train hierarchical YOLO classification models:

```bash
python scripts/03_train_models.py \
    --min-images 7 \
    --epochs 50 \
    --set-number 1
```

**What this does:**
- Generates training datasets from classified specimens
- Train/val/test split: 72% / 18% / 10%
- Augmentation: 90°/180°/270° rotations
- Trains one model per taxonomic level
- Special classes: "Other" and "Debris"

**Key parameters (from Methods):**
- Base model: `yolo11m-cls.pt`
- Image size: 512×512 pixels
- Epochs: 50
- HSV augmentation: h=0.025, s=0.3, v=0.3
- Batch size: Auto (depends on GPU memory)

**Expected output:**
```
======================================================================
TRAINING ALL MODELS
======================================================================
Using set number: 1
Found 4 training folders

Training model for Insecta (set 1)
  Base model: yolo11m-cls.pt
  Training images: 85
  Validation images: 22
  Classes: 4 (Coleoptera, Diptera, Hymenoptera, Other)

  Epoch 1/50: loss=2.341, top1_acc=0.456
  Epoch 25/50: loss=0.234, top1_acc=0.912
  Epoch 50/50: loss=0.089, top1_acc=0.968

  ✓ Model saved to data/models/classification/1.pt

Training model for Coleoptera (set 1)
  ...

======================================================================
TRAINING COMPLETE: 4/4 models
======================================================================
```

---

### Step 4: Classify Specimens (1-2 minutes)

Run hierarchical inference on specimens:

```bash
python scripts/04_classify.py \
    --set-number 1 \
    --batch-size 200
```

**What this does:**
- Loads all models for set 1
- Starts with top-level model (e.g., Insecta)
- Iteratively applies child models based on predictions
- Continues until reaching most specific level
- Records confidence scores at each step

**Expected output:**
```
======================================================================
STARTING HIERARCHICAL INFERENCE
======================================================================
Processing 150 images
Loaded 4 models
Initialized with top taxon: Insecta

======================================================================
ITERATION 1
======================================================================
Processing Insecta: 150 images
  Predicted: 45 Coleoptera, 32 Diptera, 28 Hymenoptera, 45 Other

======================================================================
ITERATION 2
======================================================================
Processing Coleoptera: 45 images
  Predicted: 15 Carabidae, 12 Staphylinidae, 10 Curculionidae, 8 Other

...

======================================================================
HIERARCHICAL INFERENCE COMPLETE
======================================================================
Saved 487 inference results to database
```

---

### Step 5: Export Results (30 seconds)

Generate Excel report with statistics:

```bash
python scripts/05_export_results.py \
    --set-number 1 \
    --format excel \
    --output exports/results.xlsx
```

**What this does:**
- Queries inference results from database
- Applies confidence thresholds (if calculated)
- Generates taxonomic summary statistics
- Creates confusion matrices (for test set)
- Exports to Excel with multiple sheets

**Expected output:**
```
Exporting results for set 1...
  Found 487 classifications
  Test set: 15 images
  Accuracy (test set): 93.3%

Sheets created:
  - Summary: Overall statistics
  - Classifications: Full results table
  - Confusion Matrix: Test set evaluation
  - Taxon Counts: Specimens per taxon

✓ Exported to exports/results.xlsx
```

---

## Optional: Threshold Optimization

Calculate confidence thresholds for 95% accuracy:

```bash
python scripts/06_optimize_thresholds.py \
    --set-number 1 \
    --target-accuracy 0.95 \
    --output results/thresholds.json
```

**What this does:**
- Analyzes test set predictions
- Iteratively increases threshold until 95% accuracy
- Saves optimal thresholds per taxon
- Can be applied during export

**Expected output:**
```
Optimizing thresholds for set 1...
  Target accuracy: 95%
  Test images: 15

Taxon: Insecta
  Starting accuracy: 100.0%
  Optimal threshold: 0.5 (no filtering needed)

Taxon: Coleoptera
  Starting accuracy: 87.5%
  Testing threshold: 0.5 → acc=87.5%
  Testing threshold: 0.6 → acc=90.0%
  Testing threshold: 0.7 → acc=95.0%  ✓
  Optimal threshold: 0.7

✓ Thresholds saved to results/thresholds.json
```

---

## Verifying Results

### Check Database
```bash
# Install sqlite3 (if not already installed)
sqlite3 data/arthropod_pipeline.db

# Query some statistics
SELECT COUNT(*) FROM single_images;
SELECT COUNT(*) FROM inference_results;
SELECT predicted_taxon, COUNT(*)
FROM inference_results
GROUP BY predicted_taxon;
```

### Check Outputs
```bash
# Processed images
ls data/processed/specimens/  # Individual specimen PNGs

# Trained models
ls data/models/classification/  # 1.pt, 2.pt, 3.pt, 4.pt

# Exports
ls exports/  # results.xlsx
```

---

## Troubleshooting

### GPU Out of Memory
```bash
# Use smaller batch size
python scripts/03_train_models.py --epochs 50 --batch-size 8

# Or force CPU mode
export CUDA_VISIBLE_DEVICES=""
python scripts/03_train_models.py --device cpu
```

### Missing Dependencies
```bash
# Reinstall with verbose output
pip install -r requirements.txt --verbose

# Check specific package
python -c "import ultralytics; print(ultralytics.__version__)"
```

### Database Locked
```bash
# Close any DB browsers
# Recreate database
python scripts/01_setup_database.py --overwrite
```

---

## Next Steps

### Full Dataset
To run on the complete dataset (12,666 specimens):

1. Download full database: `data/full_arthropod_database.sqlite` (from Zenodo)
2. Replace example DB:
   ```bash
   mv data/full_arthropod_database.sqlite data/arthropod_pipeline.db
   ```
3. Run inference:
   ```bash
   python scripts/04_classify.py --set-number 1
   ```

### Custom Data
To process your own specimens:

1. Place composite images in `data/raw/`
2. Run processing:
   ```bash
   python scripts/02_process_images.py --input data/raw/
   ```
3. Export specimens for manual classification
4. Import classifications to database
5. Train models and classify

---

## File Locations

```
arthropod-classification-pipeline/
├── data/
│   ├── arthropod_pipeline.db      ← SQLite database
│   ├── examples/                  ← Example composite images
│   ├── raw/                       ← Your input images
│   ├── processed/                 ← Extracted specimens
│   ├── models/                    ← Trained YOLO models
│   └── taxonomy/                  ← Catalogue of Life data
│
├── scripts/                       ← CLI scripts (run these)
├── src/                           ← Core pipeline code
├── exports/                       ← Results (Excel/CSV)
├── logs/                          ← Execution logs
└── config/                        ← Configuration files
```

---

## Performance Benchmarks

Tested on **Intel i7-9700K, RTX 2070 Super, 32GB RAM**:

| Step | Time | Output |
|------|------|--------|
| Database setup | 0.5 min | 1 SQLite DB |
| Process 3 composite images | 2 min | ~150 specimens |
| Train 4 models (50 epochs) | 4 min | 4× .pt files (~50 MB each) |
| Classify 150 specimens | 1 min | 487 inference results |
| Export to Excel | 0.5 min | 1 Excel file |
| **Total** | **~8 minutes** | **Complete analysis** |

---

## Citation

If you use this pipeline, please cite:

```bibtex
@article{Meyer2025,
  title={Integrated arthropod monitoring pipeline detects size-dependent responses},
  author={Meyer, Philipp and Scharnhorst, Victor and Lechner, Michael and
          Haslinger, Hanna and Gierus, Martin and Meimberg, Harald},
  journal={Current Biology},
  year={2025},
  doi={10.1016/j.cub.2025.XX.XXX},
  note={DOI will be added upon publication}
}
```

---

## Support

- **Issues**: https://github.com/PhilMeyer1/arthropod-classification-pipeline/issues
- **Documentation**: See `docs/` folder
- **Methods Paper**: [Link to Current Biology article - will be added upon publication]

---

## License

MIT License - See `LICENSE` file for details.
