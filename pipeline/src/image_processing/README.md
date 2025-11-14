# Image Processing Module

This module handles specimen detection, segmentation, and extraction from high-resolution composite images.

## Overview

The image processing pipeline consists of four main steps:

1. **Focus Stacking** (optional): Combine z-stack images into sharp composite
2. **Detection**: Detect specimens using tiled YOLO object detection
3. **Segmentation**: Segment specimens using YOLO instance segmentation
4. **Extraction**: Extract individual specimens as PNG files

## Modules

### `focus_stacking.py` - Focus Stacking

Open-source alternative to Helicon Focus using OpenCV.

```python
from src.image_processing.focus_stacking import FocusStacker

stacker = FocusStacker(method='laplacian')
result = stacker.stack_from_directory(
    directory='./z_stacks/tray_001',
    pattern='*.jpg',
    output_path='composite_001.jpg'
)
```

**Methods:**
- `laplacian`: Fast Laplacian variance method
- `pyramid`: Higher quality Laplacian pyramid blending

**Note:** Original pipeline used Helicon Focus Pro 8.3.563. See `EXTERNAL_TOOLS.md`.

### `detection.py` - Specimen Detection

Detects arthropod specimens using adaptive tiled detection with NMS.

```python
from src.image_processing.detection import SpecimenDetector

detector = SpecimenDetector(size_fraction='1')
detections = detector.detect_specimens('composite.png')

print(f"Found {len(detections)} specimens")
```

**Key Features:**
- Adaptive grid size based on size fraction (k1: 20×12, 1: 10×8, 7: 1×1)
- Configurable overlap (0-80%)
- Global NMS to remove duplicates from tiling
- Border filtering to avoid edge artifacts

**Corresponds to Methods:** "Image Analysis and Specimen Extraction - Tiled Detection"

### `segmentation.py` - Specimen Segmentation

Segments specimens within detected bounding boxes.

```python
from src.image_processing.segmentation import SpecimenSegmenter

segmenter = SpecimenSegmenter(size_fraction='1')
segmentations = segmenter.segment_specimens(
    image_path='composite.png',
    detections=detections
)
```

**Key Features:**
- YOLO instance segmentation
- Dynamic mask dilation based on image size
- Second NMS step to remove overlapping segments
- Near-white background (RGB: 248, 248, 248)

**Corresponds to Methods:** "Image Analysis and Specimen Extraction - Segmentation"

### `extraction.py` - Specimen Extraction

Extracts and saves individual specimen images.

```python
from src.image_processing.extraction import SpecimenExtractor

extractor = SpecimenExtractor(output_dir='./output/specimens')
results = extractor.extract_specimens(
    segmentations=segmentations,
    sample_id='S001',
    size_fraction='1'
)

# Or run complete pipeline:
results, metadata = extractor.extract_from_composite(
    composite_image_path='composite.png',
    sample_id='S001',
    size_fraction='1'
)
```

**Key Features:**
- Standardized naming: `{sample_id}_{size_fraction}_{index:04d}.png`
- Metadata tracking (bounding box, confidence, dimensions)
- Complete pipeline integration

**Corresponds to Methods:** "Image Analysis and Specimen Extraction - Extraction"

## Complete Pipeline Example

```python
from pathlib import Path
from src.image_processing.extraction import SpecimenExtractor

# Process a single composite image
extractor = SpecimenExtractor(output_dir='./output/S001')

results, metadata = extractor.extract_from_composite(
    composite_image_path=Path('composite_S001_1mm.png'),
    sample_id='S001',
    size_fraction='1',
    verbose=True
)

print(f"Extracted {len(results)} specimens")
print(f"Saved to: {extractor.output_dir}")

# Save metadata
extractor.save_metadata(results)

# Get statistics
stats = extractor.get_statistics(results)
print(f"Average confidence: {stats['avg_confidence']:.2f}")
print(f"Total size: {stats['total_size_mb']:.1f} MB")
```

## Configuration

All parameters can be configured in `config/default_config.yaml`:

```yaml
image_processing:
  detection:
    model_path: "./data/models/detection/yolov8n.pt"
    confidence_threshold: 0.25
    iou_threshold: 0.2

  segmentation:
    model_path: "./data/models/segmentation/yolov8n-seg.pt"
    mask_dilation_factor: 1.02

  size_fractions:
    k1:  # <1 mm
      grid_size: [20, 12]
      overlap: 0.4
    "1":  # 1-2 mm
      grid_size: [10, 8]
      overlap: 0.3
    # ... more fractions
```

## Size Fractions

| Code | Size Range | Grid Size | Overlap |
|------|-----------|-----------|---------|
| k1   | <1 mm     | 20×12     | 40%     |
| 1    | 1-2 mm    | 10×8      | 30%     |
| 2    | 2-3 mm    | 6×5       | 25%     |
| 7    | 7-10 mm   | 1×1       | 0%      |

## Models

Pre-trained YOLO models required:
- Detection: `yolov8n.pt` (object detection)
- Segmentation: `yolov8n-seg.pt` (instance segmentation)

Download using:
```bash
python scripts/download_models.py
```

Different models can be used for different size fractions (configured in YAML).

## Performance Tips

1. **GPU Acceleration**: Set `device: "cuda"` in config
2. **Batch Processing**: Process multiple images in parallel
3. **Memory**: Large images may require chunking or downsampling
4. **Model Size**: Use smaller YOLO models (n, s) for speed, larger (m, l, x) for accuracy

## Troubleshooting

**Out of Memory:**
- Reduce image size or use smaller YOLO model
- Process fewer tiles at once
- Use CPU instead of GPU

**Low Detection Rate:**
- Adjust confidence threshold (try 0.1-0.3)
- Check if correct size fraction is selected
- Verify model is trained for your specimens

**Duplicate Detections:**
- Increase IoU threshold for NMS (try 0.3-0.5)
- Reduce tile overlap

**Segmentation Quality:**
- Adjust mask dilation factor (default: 10)
- Try different segmentation models

## Testing

Run image processing tests:
```bash
python -m pytest tests/test_detection.py
python -m pytest tests/test_segmentation.py
```

## References

- Methods section: "Image Analysis and Specimen Extraction"
- YOLOv8: https://github.com/ultralytics/ultralytics
- Original detection/segmentation: `gui/Tabs/Bildverarbeitung/`
