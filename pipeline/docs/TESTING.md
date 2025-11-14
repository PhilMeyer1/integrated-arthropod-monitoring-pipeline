# Testing Guide

This document describes how to test the arthropod classification pipeline.

## Prerequisites

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download YOLO Models:**
   ```bash
   python scripts/download_models.py
   ```

3. **Create Test Data:**
   ```bash
   python scripts/create_test_images.py
   ```

## Test Data Structure

After running `create_test_images.py`, you should have:

```
data/examples/
├── composite_S001_1mm.png      # 30 medium specimens (1-2mm fraction)
├── composite_S002_2mm.png      # 20 larger specimens (2-3mm fraction)
├── composite_S003_k1.png       # 50 small specimens (<1mm fraction)
└── metadata_example.csv        # Metadata for batch processing
```

## Running Tests

### 1. Image Processing Tests

#### Single Image Mode
Test processing a single composite image:

```bash
python scripts/02_process_images.py \
    --input data/examples/composite_S001_1mm.png \
    --sample-id S001 \
    --size-fraction 1 \
    --output-dir ./output/test_single \
    --verbose
```

**Expected Output:**
```
output/test_single/
├── S001/
│   └── 1/
│       ├── S001_1_0001.png
│       ├── S001_1_0002.png
│       ├── ...
│       └── metadata.json
└── processing_results.csv
```

**Verification:**
- Check that individual specimen PNGs are created
- Verify metadata.json contains detection info
- Review processing_results.csv for statistics
- Check logs in `./logs/pipeline.log`

#### Metadata CSV Mode (Recommended)
Test batch processing from metadata file:

```bash
python scripts/02_process_images.py \
    --metadata data/examples/metadata_example.csv \
    --output-dir ./output/test_batch \
    --verbose
```

**Expected Output:**
```
output/test_batch/
├── S001/
│   └── 1/
│       ├── S001_1_0001.png
│       └── metadata.json
├── S002/
│   └── 2/
│       ├── S002_2_0001.png
│       └── metadata.json
├── S003/
│   └── k1/
│       ├── S003_k1_0001.png
│       └── metadata.json
└── processing_results.csv
```

**Verification:**
- All three samples processed
- Each has correct size fraction directory
- processing_results.csv shows summary for all images
- No errors in log file

#### Directory Batch Mode
Test processing all images in a directory:

```bash
python scripts/02_process_images.py \
    --input-dir data/examples \
    --pattern "composite_*.png" \
    --output-dir ./output/test_directory \
    --verbose
```

**Expected Output:**
Similar to metadata mode, but extracts sample_id and size_fraction from filenames.

**Verification:**
- Automatic filename parsing works correctly
- All matching files processed
- Sample IDs and size fractions correctly extracted

### 2. Individual Module Tests

#### Detection Module
```python
from pathlib import Path
from src.image_processing.detection import SpecimenDetector

detector = SpecimenDetector(size_fraction='1')
detections = detector.detect_specimens(
    Path('data/examples/composite_S001_1mm.png')
)

print(f"Detected {len(detections)} specimens")
for i, det in enumerate(detections[:5]):  # Show first 5
    print(f"  {i+1}. BBox: {det['bounding_box']}, Conf: {det['confidence']:.3f}")
```

**Expected:**
- Number of detections close to expected (may vary with YOLO model)
- Confidence scores between 0.0 and 1.0
- Bounding boxes within image dimensions

#### Segmentation Module
```python
from pathlib import Path
from src.image_processing.detection import SpecimenDetector
from src.image_processing.segmentation import SpecimenSegmenter

# First detect
detector = SpecimenDetector(size_fraction='1')
detections = detector.detect_specimens(
    Path('data/examples/composite_S001_1mm.png')
)

# Then segment
segmenter = SpecimenSegmenter(size_fraction='1')
segmentations = segmenter.segment_specimens(
    Path('data/examples/composite_S001_1mm.png'),
    detections
)

print(f"Segmented {len(segmentations)} specimens")
for i, seg in enumerate(segmentations[:5]):
    print(f"  {i+1}. Has mask: {seg['mask'] is not None}")
```

**Expected:**
- Number of segmentations ≤ detections (some may be filtered)
- Each segmentation has a mask
- Image data is extracted

#### Complete Pipeline
```python
from pathlib import Path
from src.image_processing.extraction import SpecimenExtractor

extractor = SpecimenExtractor(output_dir=Path('./output/test_pipeline'))

results, metadata = extractor.extract_from_composite(
    composite_image_path=Path('data/examples/composite_S001_1mm.png'),
    sample_id='S001',
    size_fraction='1',
    verbose=True
)

print(f"\nExtracted {len(results)} specimens")

# Get statistics
stats = extractor.get_statistics(results)
print(f"Average confidence: {stats['avg_confidence']:.3f}")
print(f"Size range: {stats['min_size']}px - {stats['max_size']}px")
```

**Expected:**
- Complete pipeline runs without errors
- Specimen files created in output directory
- Metadata JSON created
- Statistics calculated correctly

### 3. Configuration Tests

#### Test Config Loading
```python
from src.config import config

# Test path expansion
print(f"Data root: {config.get('paths.data_root')}")
print(f"Models path: {config.get('paths.models')}")

# Test detection config
print(f"Confidence threshold: {config.get('image_processing.detection.confidence_threshold')}")

# Test size fraction config
k1_config = config.get('image_processing.size_fractions.k1')
print(f"k1 grid size: {k1_config['grid_size']}")
print(f"k1 overlap: {k1_config['overlap']}")
```

**Expected:**
- All paths resolved correctly
- Nested config values accessible
- Default values present

#### Test Custom Config
```python
from src.config import Config

# Load custom config
custom_config = Config('config/custom_config.yaml')

# Verify overrides work
print(f"Custom device: {custom_config.get('image_processing.device')}")
```

### 4. Validation Tests

```python
from pathlib import Path
from src.utils.validation import (
    validate_image_file,
    validate_size_fraction,
    validate_sample_id,
    validate_bounding_box,
    check_gpu_available
)

# Test image validation
try:
    validate_image_file('data/examples/composite_S001_1mm.png')
    print("✓ Image file validation passed")
except Exception as e:
    print(f"✗ Image file validation failed: {e}")

# Test size fraction validation
try:
    validate_size_fraction('1')
    validate_size_fraction('k1')
    validate_size_fraction('invalid')  # Should raise
except ValueError as e:
    print(f"✓ Size fraction validation working: {e}")

# Test sample ID validation
try:
    validate_sample_id('S001')
    validate_sample_id('Sample-123_ABC')
    print("✓ Sample ID validation passed")
except Exception as e:
    print(f"✗ Sample ID validation failed: {e}")

# Test bbox validation
try:
    validate_bounding_box((10, 20, 100, 200))
    print("✓ BBox validation passed")
except Exception as e:
    print(f"✗ BBox validation failed: {e}")

# Check GPU
if check_gpu_available():
    print("✓ GPU (CUDA) available")
else:
    print("ℹ GPU not available, will use CPU")
```

### 5. Logging Tests

```python
from src.utils.logging_config import setup_logging, LogContext, ProgressLogger

# Test basic logging
logger = setup_logging('test_logger', level='DEBUG')
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")

# Test LogContext
with LogContext('Test operation', logger):
    # Simulate work
    import time
    time.sleep(1)
# Should log: "Test operation - Completed in 1.0s"

# Test ProgressLogger
progress = ProgressLogger(total=100, name='Test progress', logger=logger)
for i in range(100):
    progress.update()
    # Simulate work
    time.sleep(0.01)
progress.finish()
# Should log progress at intervals
```

## Common Issues and Solutions

### Issue: Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size in config
2. Use smaller YOLO model (n instead of m/l)
3. Switch to CPU: `device: "cpu"` in config
4. Process smaller images or fewer tiles

### Issue: Low Detection Rate

**Symptoms:**
- Very few or no detections
- Expected ~30 specimens, got 5

**Solutions:**
1. Lower confidence threshold:
   ```yaml
   detection:
     confidence_threshold: 0.1  # Try lower
   ```
2. Verify correct size fraction selected
3. Check if model is appropriate for specimen size
4. Enable verbose mode to see detection details

### Issue: Duplicate Detections

**Symptoms:**
- Same specimen detected multiple times
- Overlapping bounding boxes

**Solutions:**
1. Increase IoU threshold for NMS:
   ```yaml
   detection:
     iou_threshold: 0.4  # Try higher (default: 0.2)
   ```
2. Reduce tile overlap in size fraction config
3. Check border filtering is enabled

### Issue: Poor Segmentation Quality

**Symptoms:**
- Segmentation masks too small
- Missing appendages or fine details

**Solutions:**
1. Increase mask dilation:
   ```yaml
   segmentation:
     mask_dilation_factor: 1.05  # Default: 1.02
   ```
2. Try different segmentation model
3. Verify detection bounding boxes are correct

### Issue: Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'ultralytics'
```

**Solutions:**
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Verify virtual environment is activated
3. Check Python version (requires 3.8+)

## Performance Benchmarks

Expected processing times (approximate, will vary by hardware):

| Image Size | Size Fraction | GPU (RTX 3080) | CPU (i7) |
|-----------|---------------|----------------|----------|
| 4000×3000 | k1 (20×12)   | ~30s           | ~3min    |
| 4000×3000 | 1 (10×8)     | ~15s           | ~90s     |
| 4000×3000 | 2 (6×5)      | ~8s            | ~45s     |
| 4000×3000 | 7 (1×1)      | ~2s            | ~10s     |

## Next Steps After Testing

Once basic tests pass:

1. **Test with Real Data:**
   - Use actual composite images from your dataset
   - Compare results with manual counts/identifications
   - Adjust thresholds if needed

2. **Optimize Parameters:**
   - Tune confidence and IoU thresholds
   - Adjust tile overlap if needed
   - Test different YOLO model sizes

3. **Run Full Pipeline:**
   - Process complete dataset
   - Monitor for errors or edge cases
   - Review quality of extracted specimens

4. **Document Results:**
   - Record processing statistics
   - Note any issues encountered
   - Update config with optimal parameters

## Automated Testing

### Test Suite Overview

The project includes a comprehensive test suite using **pytest**:

```
tests/
├── conftest.py                    # Shared fixtures and pytest configuration
├── test_image_processing.py       # Unit tests for detection, segmentation, extraction
├── test_classification.py         # Unit tests for taxonomy, training, inference
├── test_export.py                 # Unit tests for Excel, CSV, statistics
├── test_config.py                 # Unit tests for configuration system
└── test_integration_e2e.py        # End-to-end integration tests
```

### Running Tests

#### Prerequisites

Install pytest and coverage tools:
```bash
pip install pytest pytest-cov
```

#### Run All Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=src --cov-report=html tests/

# View coverage report
open htmlcov/index.html  # macOS/Linux
start htmlcov/index.html  # Windows
```

#### Run Specific Test Files

```bash
# Test image processing only
pytest tests/test_image_processing.py -v

# Test classification only
pytest tests/test_classification.py -v

# Test export functionality
pytest tests/test_export.py -v

# Run integration tests
pytest tests/test_integration_e2e.py -v
```

#### Run by Test Markers

Tests are marked with custom markers for selective execution:

```bash
# Skip slow tests
pytest -m "not slow"

# Run only integration tests
pytest -m integration

# Skip tests requiring models
pytest -m "not requires_model"

# Skip tests requiring GPU
pytest -m "not requires_gpu"
```

### Test Categories

#### 1. Unit Tests

Test individual functions and classes:

**Image Processing** (`test_image_processing.py`):
- Detector initialization and grid size mapping
- Tile creation from image dimensions
- Mask dilation calculation
- Specimen filename generation
- Statistics calculation

**Classification** (`test_classification.py`):
- Taxonomy hierarchy construction
- Parent-child relationships
- Training dataset split ratios
- Model naming conventions
- Threshold optimization

**Export** (`test_export.py`):
- Excel file creation and structure
- CSV content verification
- Statistics aggregation
- Per-taxon and per-sample metrics

**Configuration** (`test_config.py`):
- YAML loading
- Variable expansion (${var})
- Environment variable support
- Dot notation access
- Path resolution

#### 2. Integration Tests

Test component interactions:

**Detection → Segmentation → Extraction** (`test_integration_e2e.py`):
- Complete image processing flow
- Batch processing with metadata
- Data consistency across stages

**Statistics → Export** (`test_integration_e2e.py`):
- Calculate statistics and export to Excel/CSV
- Multiple export formats from same data

#### 3. End-to-End Tests

Test complete workflows:

**Single Sample Workflow** (`test_integration_e2e.py`):
- Process composite image
- Classify specimens
- Export results
- Verify all outputs

**Multiple Samples Workflow** (`test_integration_e2e.py`):
- Process multiple samples
- Combined statistics
- Per-sample aggregation

### Example Test Usage

#### Running a Quick Test

```bash
# Quick smoke test (fast tests only)
pytest -m "not slow and not requires_model" -v
```

#### Running Full Test Suite with Coverage

```bash
# Complete test suite with HTML coverage report
pytest --cov=src --cov-report=html --cov-report=term tests/

# Expected output:
# ===== test session starts =====
# tests/test_config.py ............ [ 12%]
# tests/test_export.py ............. [ 25%]
# tests/test_classification.py ..... [ 38%]
# tests/test_image_processing.py ... [ 50%]
# tests/test_integration_e2e.py .... [100%]
#
# ===== 48 passed in 12.34s =====
```

### Fixtures and Test Data

Common fixtures provided in `conftest.py`:

- `sample_composite_image`: Realistic composite image with specimens
- `sample_specimen_image`: Single specimen image
- `sample_mask`: Binary segmentation mask
- `sample_detections`: Mock detection results
- `sample_metadata_csv`: Sample metadata for batch processing
- `sample_config_dict`: Complete test configuration
- `mock_yolo_model`: Mock YOLO model for testing without trained models

### Continuous Integration

To set up automated testing with GitHub Actions, create `.github/workflows/tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.8'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest --cov=src tests/
```

### Test Development Guidelines

When adding new features:

1. **Write tests first** (TDD approach)
2. **Use descriptive test names**: `test_detection_handles_empty_image`
3. **Test edge cases**: empty inputs, invalid data, boundary conditions
4. **Use fixtures** for shared test data
5. **Mock external dependencies** (models, APIs)
6. **Add docstrings** to explain what's being tested
7. **Mark slow tests**: `@pytest.mark.slow`
8. **Mark tests requiring resources**: `@pytest.mark.requires_model`

### Example Jupyter Notebook

For interactive testing and exploration:

```bash
jupyter notebook notebooks/01_quickstart_example.ipynb
```

This notebook demonstrates:
- Complete pipeline workflow
- Individual module usage
- Result visualization
- Export examples
