# Architecture Overview

System design and components of the Arthropod Classification Pipeline.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT: Composite Images                      │
│              (High-resolution, multiple specimens)               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                 STAGE 1: Image Processing                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌────────────────┐  ┌──────────────────┐   │
│  │  Detection   │─▶│  Segmentation  │─▶│   Extraction     │   │
│  │  (Tiled YOLO)│  │  (YOLO-seg)    │  │  (PNG + metadata)│   │
│  └──────────────┘  └────────────────┘  └──────────────────┘   │
│                                                                  │
│  Outputs: Individual specimen images + bounding boxes           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 2: Hierarchical Classification                │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐                                           │
│  │ Taxonomy Builder │  Builds hierarchical tree                 │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │ Dataset Creator  │  Train/val/test splits + augmentation     │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │ Model Trainer    │  Train YOLO models per taxonomic level    │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │ Inference Engine │  Hierarchical top-down classification     │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │ Threshold Optim. │  Optimize confidence thresholds           │
│  └──────────────────┘                                           │
│                                                                  │
│  Outputs: Classifications + confidences + hierarchy              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  STAGE 3: Export & Analysis                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────┐  ┌───────────┐  ┌────────────────┐             │
│  │   Excel   │  │    CSV    │  │   Statistics   │             │
│  │  Exporter │  │ Exporter  │  │   Calculator   │             │
│  └───────────┘  └───────────┘  └────────────────┘             │
│                                                                  │
│  Outputs: Publication-ready tables, statistics, figures          │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Image Processing (`src/image_processing/`)

#### Detection (`detection.py`)

**Purpose:** Detect specimen bounding boxes in composite images

**Algorithm:**
```python
1. Divide image into overlapping tiles (grid size depends on size fraction)
2. Run YOLO object detection on each tile
3. Transform detections back to full image coordinates
4. Apply global NMS to remove duplicates
5. Filter border detections
```

**Key Parameters:**
- Grid size: Adaptive (k1: 20×12, 1mm: 10×8, 7mm: 1×1)
- Overlap: 0-80% depending on size fraction
- IoU threshold: 0.2 for NMS

**Corresponds to:** Methods "Tiled Detection"

#### Segmentation (`segmentation.py`)

**Purpose:** Extract precise specimen boundaries

**Algorithm:**
```python
1. For each detected bounding box:
   a. Crop image to detection + margin
   b. Run YOLO instance segmentation
   c. Dilate mask by factor (e.g., 1.02x)
   d. Extract specimen with near-white background
```

**Key Parameters:**
- Mask dilation: 1.02 (2% expansion)
- Background color: RGB(248, 248, 248)

**Corresponds to:** Methods "Segmentation"

#### Extraction (`extraction.py`)

**Purpose:** Save individual specimens as files

**Output Format:**
```
{sample_id}_{size_fraction}_{index:04d}.png
Example: S001_1_0001.png
```

**Metadata:**
```json
{
  "image_id": 1,
  "sample_id": "S001",
  "size_fraction": "1",
  "bounding_box": [x1, y1, x2, y2],
  "confidence": 0.95,
  "width": 512,
  "height": 384
}
```

### 2. Classification (`src/classification/`)

#### Taxonomy Hierarchy (`taxonomy.py`)

**Purpose:** Build hierarchical tree from Catalogue of Life

**Data Structure:**
```python
{
  "RT": {  # Root (Arthropoda)
    "children": {"Insecta", "Arachnida"},
    "images": [],
    "parent_id": None
  },
  "Insecta": {
    "children": {"Coleoptera", "Diptera"},
    "images": [1, 2, 3, ...],
    "parent_id": "RT"
  },
  ...
}
```

**Features:**
- Time filtering (e.g., only 2020-2023)
- Recursive ancestor addition
- Image aggregation per taxon

#### Training Dataset Creator (`training_data.py`)

**Purpose:** Create train/val/test splits

**Process:**
```python
1. Split each taxon's images:
   - 10% test (excluded from training)
   - 15% validation
   - 75% training

2. For each suitable taxon (≥ min_images children):
   a. Create class folders for each child
   b. Add "additional_class" for rare taxa
   c. Add "Debris" class for non-specimens

3. Apply augmentation (training only):
   - Rotation: 0°, 90°, 180°, 270°
```

**Directory Structure:**
```
training_data/
├── Insecta/           # Model for Insecta level
│   ├── train/
│   │   ├── Coleoptera/
│   │   ├── Diptera/
│   │   ├── additional_class/
│   │   └── Debris/
│   └── val/
│       └── ...
└── Coleoptera/        # Model for Coleoptera level
    ├── train/
    └── val/
```

#### Model Trainer (`training.py`)

**Purpose:** Train YOLO classification models

**Training Configuration:**
```yaml
epochs: 50
image_size: 512
batch_size: -1  # Auto
patience: 10
base_model: "yolo11m-cls.pt"

# Minimal augmentation (already rotated)
mosaic: 0.0
degrees: 0.0
translate: 0.0
scale: 0.0
flipud: 0.0
fliplr: 0.0
```

**Model Versioning:**
- Set numbers group related models (e.g., Set 1, Set 2)
- All models in a set trained together
- Allows A/B testing of different configurations

#### Inference Engine (`inference.py`)

**Purpose:** Hierarchical classification

**Algorithm:**
```python
1. Load all models for set_number
2. Find top-level model (e.g., Arthropoda)
3. Initialize: all images → top taxon

4. LOOP until no changes:
   For each model_taxon:
     a. Get images currently at this level
     b. Run batch inference
     c. Update image → predicted child taxon
     d. Record confidence + correctness
     e. Save results

5. Final: each image classified to leaf level
```

**Example Flow:**
```
Image 123: RT (start)
  ↓ [Insecta, conf=0.95]
Image 123: Insecta
  ↓ [Coleoptera, conf=0.88]
Image 123: Coleoptera
  ↓ [Carabidae, conf=0.92]
Image 123: Carabidae (final - no child model)
```

#### Threshold Optimizer (`thresholds.py`)

**Purpose:** Find optimal confidence thresholds

**Optimization:**
```python
For each taxon:
  For threshold in [0.1, 0.15, 0.2, ..., 1.0]:
    Calculate F1 score (or accuracy, precision)
  Select threshold with best F1

Output: Per-taxon thresholds
```

**Metrics:**
- F1 score (default)
- Accuracy
- Precision
- Recall

### 3. Export (`src/export/`)

#### Excel Exporter (`excel_export.py`)

**Features:**
- Multi-sheet workbooks
- Formatted headers (colored, bold)
- Auto-adjusted column widths
- Frozen header rows

**Sheets:**
1. Classification Results
2. Summary Statistics
3. Per-Taxon Statistics
4. Taxonomy Hierarchy

#### CSV Exporter (`csv_export.py`)

**Exports:**
- Results (full data)
- Statistics (summary)
- Confusion matrix data
- Hierarchical paths
- Optimized thresholds

**Format:** Standard CSV (comma-separated)

#### Statistics Calculator (`statistics.py`)

**Calculations:**
- Overall metrics (accuracy, precision, recall, F1)
- Per-taxon metrics
- Confidence distributions
- Error analysis (top confusion pairs)
- Test set vs all data comparison

## Data Flow

### 1. Image Processing Flow

```
Composite Image (4000×3000 px)
    │
    ├─▶ Tiled Detection
    │   ├─ Tile 1 (400×375 px) ─▶ YOLO ─▶ Detections
    │   ├─ Tile 2 (400×375 px) ─▶ YOLO ─▶ Detections
    │   └─ ... (80 tiles for 10×8 grid)
    │
    ├─▶ Global NMS ─▶ Filtered Detections (30 specimens)
    │
    ├─▶ Segmentation
    │   ├─ Crop 1 ─▶ YOLO-seg ─▶ Mask 1
    │   ├─ Crop 2 ─▶ YOLO-seg ─▶ Mask 2
    │   └─ ...
    │
    └─▶ Extraction
        ├─ S001_1_0001.png (specimen 1)
        ├─ S001_1_0002.png (specimen 2)
        └─ ... + metadata.json
```

### 2. Classification Flow

```
Specimen Images (30 specimens)
    │
    ├─▶ Taxonomy Hierarchy
    │   └─▶ Tree structure with image counts
    │
    ├─▶ Dataset Creation
    │   ├─ Train/val/test split (75/15/10)
    │   ├─ Augmentation (rotations)
    │   └─▶ Training directories
    │
    ├─▶ Model Training
    │   ├─ Model 1: Arthropoda level
    │   ├─ Model 2: Insecta level
    │   ├─ Model 3: Coleoptera level
    │   └─▶ Trained models (.pt files)
    │
    ├─▶ Inference
    │   ├─ Iteration 1: All → Insecta
    │   ├─ Iteration 2: Insecta → Coleoptera
    │   ├─ Iteration 3: Coleoptera → Carabidae
    │   └─▶ Final classifications
    │
    ├─▶ Threshold Optimization
    │   └─▶ Optimal thresholds per taxon
    │
    └─▶ Export
        ├─ results.xlsx
        ├─ statistics.csv
        └─ publication_stats.json
```

## Configuration System

### Config File (`config/default_config.yaml`)

**Structure:**
```yaml
paths:
  data_root: "./data"
  models: "${paths.data_root}/models"  # Variable expansion

image_processing:
  detection:
    model_path: "${paths.models}/detection/scale_cutout2.pt"
    confidence_threshold: 0.25

  size_fractions:
    "1":
      grid_size: [10, 8]
      overlap: 0.3

classification:
  device: "cuda"
  training:
    epochs: 50

database:
  use_database: false  # Optional
  url: "sqlite:///arthropod.db"
```

**Features:**
- Variable expansion: `${section.key}`
- Environment variables: `${HOME}`
- Path resolution
- Dot notation access in Python

## Module Dependencies

```
┌─────────────────────┐
│  Scripts (scripts/) │
└──────────┬──────────┘
           │
           ▼
┌──────────────────────────────────────┐
│  Image Processing (src/image_proc/)  │
│  - Detection                          │
│  - Segmentation                       │
│  - Extraction                         │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│  Classification (src/classification/)│
│  - Taxonomy                           │
│  - Training Data                      │
│  - Training                           │
│  - Inference                          │
│  - Thresholds                         │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│  Export (src/export/)                 │
│  - Excel Export                       │
│  - CSV Export                         │
│  - Statistics                         │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│  Utilities (src/utils/)               │
│  - Logging                            │
│  - Validation                         │
│  - Image Utils                        │
│  - Config                             │
└───────────────────────────────────────┘
```

## External Dependencies

- **Ultralytics YOLO:** Detection, segmentation, classification models
- **PyTorch:** Deep learning backend
- **OpenCV:** Image processing
- **Pandas:** Data manipulation
- **Openpyxl:** Excel export
- **Matplotlib:** Plotting (optional visualization)
- **SQLAlchemy:** Database ORM (optional)

## Performance Considerations

### Bottlenecks

1. **Image Processing:** Tiled detection on large images
   - **Solution:** GPU acceleration, parallel processing

2. **Model Training:** 50 epochs × multiple models
   - **Solution:** GPU, larger batch size, fewer epochs

3. **Inference:** Sequential model application
   - **Solution:** Batch processing, GPU

### Optimization Strategies

1. **Multiprocessing:**
   - Dataset creation uses `multiprocessing.Pool`
   - Configurable worker count

2. **Batch Processing:**
   - Inference processes 200 images per batch
   - Reduces YOLO overhead

3. **GPU Acceleration:**
   - Automatic GPU detection
   - Configurable device (cuda/cpu)

4. **Memory Management:**
   - Models loaded/unloaded per iteration
   - Results saved periodically

## Extensibility

### Adding New Modules

1. Create module in `src/your_module/`
2. Add `__init__.py` with exports
3. Update `src/__init__.py`
4. Add documentation README
5. Create corresponding script in `scripts/`

### Custom Data Sources

Replace `DataManager` with custom implementation:

```python
from src.database.utils import DataManager

class CustomDataManager(DataManager):
    def get_specimens(self, **kwargs):
        # Your implementation
        pass
```

### Alternative Models

Replace YOLO with custom models:

```python
from src.image_processing.detection import SpecimenDetector

class CustomDetector(SpecimenDetector):
    def detect_specimens(self, image_path):
        # Your detection algorithm
        pass
```

## Testing Architecture

See [`TESTING.md`](TESTING.md) for:
- Unit tests
- Integration tests
- End-to-end tests
- Performance benchmarks

---

**Architecture designed for:**
- ✅ Reproducibility
- ✅ Scalability
- ✅ Modularity
- ✅ Extensibility
- ✅ Performance
