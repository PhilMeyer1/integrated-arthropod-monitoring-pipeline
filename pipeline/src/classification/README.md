# Classification Module

This module implements hierarchical taxonomic classification using YOLO models.

## Overview

The classification system follows the natural taxonomic hierarchy:

```
Arthropoda (RT)
├── Insecta
│   ├── Coleoptera
│   │   ├── Carabidae
│   │   └── Staphylinidae
│   └── Diptera
│       └── ...
└── ...
```

Each level has a separate YOLO classification model. During inference, the system:
1. Starts at the top level (e.g., Arthropoda)
2. Classifies specimen into child taxon
3. Applies child model if available
4. Repeats until reaching leaf level

## Modules

### `taxonomy.py` - Taxonomy Hierarchy

Builds and manages the taxonomic tree from the Catalogue of Life database.

```python
from src.classification import TaxonomyHierarchy

# Build hierarchy
hierarchy = TaxonomyHierarchy(start_year=2020, end_year=2023)
tree = hierarchy.build()

# Print tree
hierarchy.print_tree(tree)

# Get statistics
stats = hierarchy.get_hierarchy_statistics(tree)
print(f"Total taxa: {stats['total_taxa']}")
print(f"Max depth: {stats['max_depth']}")

# Save to file
hierarchy.save_hierarchy(tree, Path('./taxonomy.json'))
```

**Key Features:**
- Time filtering (e.g., only 2020-2023 specimens)
- Recursive ancestor addition
- Image aggregation per taxon
- Hierarchy statistics

### `training_data.py` - Training Dataset Creation

Creates train/val/test splits with data augmentation.

```python
from src.classification import TrainingDatasetCreator
from pathlib import Path

creator = TrainingDatasetCreator(
    base_directory=Path('./training_data'),
    start_year=2020,
    end_year=2023
)

creator.generate_training_data(
    top_taxon_id='RT',  # Arthropoda
    min_images_per_class=7,
    additional_class_train_images=10,
    additional_class_val_images=2,
    num_processes=8,
    test_ratio=0.10,
    val_ratio=0.15
)
```

**Output Structure:**
```
training_data/
├── RT/  # Arthropoda level
│   ├── train/
│   │   ├── Insecta/
│   │   ├── Debris/
│   │   └── additional_class/
│   └── val/
│       └── ...
├── Insecta/  # Insecta level
│   ├── train/
│   │   ├── Coleoptera/
│   │   ├── Diptera/
│   │   └── additional_class/
│   └── val/
│       └── ...
└── ...
```

**Key Features:**
- 10% test, 15% val, 75% train split
- Rotation augmentation (90°, 180°, 270°) for training only
- "Additional class" for rare taxa
- Debris class for non-specimens
- Multiprocessing for speed

### `training.py` - Model Training

Trains YOLO classification models for each taxonomic level.

```python
from src.classification import ModelTrainer
from pathlib import Path

trainer = ModelTrainer(
    trainingsets_path=Path('./training_data'),
    models_path=Path('./models')
)

# Train all models
trained_models = trainer.train_all_models(
    set_number=1,  # Auto-increments if None
    delete_datasets_after=True  # Save disk space
)

print(f"Trained {len(trained_models)} models")
```

**Training Configuration** (from `config/default_config.yaml`):
```yaml
classification:
  training:
    epochs: 50
    image_size: 512
    batch_size: -1  # Auto
    patience: 10
    base_model: "yolo11m-cls.pt"
```

**Augmentation:**
- Minimal augmentation (specimens already rotated during data prep)
- Slight HSV variations only
- No geometric transformations

**Output:**
- Models saved to `./models/{model_id}.pt`
- Metadata in `./models/{model_id}.json` or database
- Model set numbers for versioning

### `inference.py` - Hierarchical Inference

Runs hierarchical classification on specimens.

```python
from src.classification import InferenceEngine

engine = InferenceEngine(
    set_number=1,  # Which model set to use
    models_path=Path('./models')
)

# Run on specific images
results = engine.run_inference(
    image_ids=[1, 2, 3, 4, 5],
    batch_size=200
)

# Or run on all images
results = engine.run_inference()

# Results are automatically saved to database or JSON
```

**How It Works:**
1. Load all models for the set
2. Find top-level model (e.g., Arthropoda)
3. Initialize all images at top level
4. **Iteration loop:**
   - For each model/taxon:
     - Find images currently at that level
     - Run batch inference
     - Update determinations to child taxa
     - Record confidence at each step
   - Repeat until no changes (all images at leaf level)

**Special Handling:**
- Debris detection marks images as non-specimens
- Confidence tracking at each level
- Test set filtering for evaluation
- Comparison with manual determinations

**Results Format:**
```json
[
  {
    "image_id": 123,
    "model_taxon_id": "Insecta",
    "predicted_taxon_id": "Coleoptera",
    "correct_taxon_id": "Coleoptera",
    "confidence": 0.9523,
    "is_excluded": true,
    "set_number": 1
  },
  ...
]
```

### `thresholds.py` - Threshold Optimization

Optimizes confidence thresholds to maximize accuracy.

```python
from src.classification import ThresholdOptimizer

optimizer = ThresholdOptimizer()

# Load inference results
optimizer.load_results(set_number=1)

# Optimize thresholds (per taxon, using F1 score)
thresholds = optimizer.optimize_thresholds(
    method='f1',  # or 'accuracy', 'precision'
    min_confidence=0.1,
    per_taxon=True,
    only_test_set=True
)

# Save thresholds
optimizer.save_thresholds(thresholds, Path('./thresholds.json'))

# Get summary
summary = optimizer.get_threshold_summary(thresholds)
print(summary.to_string())
```

**Threshold Format:**
```json
{
  "Insecta": {
    "threshold": 0.45,
    "metrics": {
      "f1": 0.923,
      "accuracy": 0.945,
      "precision": 0.967,
      "recall": 0.901
    },
    "n_samples": 234
  },
  "Coleoptera": {
    "threshold": 0.55,
    ...
  }
}
```

**Applying Thresholds:**
```python
# Filter results by optimal thresholds
filtered_results = optimizer.apply_thresholds(
    results_df,
    thresholds,
    default_threshold=0.5
)
```

## Complete Workflow Example

```python
from pathlib import Path
from src.classification import (
    TaxonomyHierarchy,
    TrainingDatasetCreator,
    ModelTrainer,
    InferenceEngine,
    ThresholdOptimizer
)

# Step 1: Build taxonomy hierarchy
hierarchy = TaxonomyHierarchy(start_year=2020, end_year=2023)
tree = hierarchy.build()
hierarchy.save_hierarchy(tree, Path('./taxonomy.json'))

# Step 2: Create training datasets
creator = TrainingDatasetCreator(
    base_directory=Path('./training_data'),
    start_year=2020,
    end_year=2023
)

trainingset_structure, excluded_ids = creator.generate_training_data(
    top_taxon_id='RT',
    min_images_per_class=7
)

# Step 3: Train models
trainer = ModelTrainer(
    trainingsets_path=Path('./training_data'),
    models_path=Path('./models')
)

trained_models = trainer.train_all_models(
    set_number=1,
    delete_datasets_after=True
)

# Step 4: Run inference
engine = InferenceEngine(
    set_number=1,
    models_path=Path('./models')
)

results = engine.run_inference(batch_size=200)

# Step 5: Optimize thresholds
optimizer = ThresholdOptimizer()
optimizer.load_results(set_number=1)

thresholds = optimizer.optimize_thresholds(
    method='f1',
    per_taxon=True,
    only_test_set=True
)

optimizer.save_thresholds(thresholds, Path('./thresholds.json'))

# Get summary
summary = optimizer.get_threshold_summary(thresholds)
print(summary.to_string())
```

## Configuration

All parameters can be configured in `config/default_config.yaml`:

```yaml
classification:
  device: "cuda"  # or "cpu"

  training:
    epochs: 50
    image_size: 512
    batch_size: -1
    patience: 10
    workers: 10
    base_model: "yolo11m-cls.pt"

  inference:
    batch_size: 200
    default_threshold: 0.5

  thresholds:
    method: "f1"  # or "accuracy", "precision"
    min_confidence: 0.1
```

## Methods Section Reference

This module corresponds to these Methods sections:

- **Hierarchical Classification Architecture**: `taxonomy.py`, overall structure
- **Training Dataset Creation**: `training_data.py`
- **Model Training**: `training.py`
- **Classification Workflow**: `inference.py`
- **Threshold Optimization**: `thresholds.py`

## Size Fractions

The system can be used for all size fractions:
- k1 (<1mm)
- 1 (1-2mm)
- 2 (2-3mm)
- 3 (3-5mm)
- 5 (5-7mm)
- 7 (7-10mm)
- A (>10mm)

Each size fraction may have different taxonomic composition, leading to different model structures.

## Model Set Numbers

Model sets allow version tracking:
- **Set 1**: Initial training (2020-2023 data)
- **Set 2**: Retrained with more data
- **Set 3**: Updated taxonomy
- etc.

All models in a set are trained together and should be used together during inference.

## Performance Tips

1. **GPU Acceleration**: Set `device: "cuda"` in config
2. **Batch Size**: Larger batches = faster inference (if GPU memory allows)
3. **Multiprocessing**: Use `num_processes=8` or more for dataset creation
4. **Disk Space**: Use `delete_datasets_after=True` to save space
5. **Test Set**: Always use `only_test_set=True` for threshold optimization

## Troubleshooting

**Issue: Out of Memory During Training**
- Reduce `batch_size` (try 16, 32)
- Use smaller model (`yolo11n-cls.pt` instead of `yolo11m-cls.pt`)
- Reduce `image_size` (try 384 or 256)

**Issue: Poor Classification Accuracy**
- Check if enough training data (min 7 specimens per taxon)
- Optimize thresholds using test set
- Try larger model (`yolo11l-cls.pt`)
- Increase training epochs

**Issue: Inference Very Slow**
- Increase `batch_size`
- Use GPU (`device: "cuda"`)
- Process fewer images at once

**Issue: Too Many Taxa (models)**
- Increase `min_images_per_class` (try 10 or 15)
- This creates fewer, more robust models
- Rare taxa go to "additional_class"

## References

- Original GUI implementation: `gui/Tabs/AI/hirarchical_classification_architecture/full_process.py`
- YOLOv11 Classification: https://docs.ultralytics.com/tasks/classify/
- Catalogue of Life: https://www.catalogueoflife.org/
