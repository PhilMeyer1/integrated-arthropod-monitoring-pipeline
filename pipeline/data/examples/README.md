# Example Data

This directory contains **real example composite images** from the study for testing and demonstrating the arthropod classification pipeline.

## Directory Structure

```
examples/
├── README.md                          # This file
├── metadata_example.csv               # Metadata for example images
├── composite_S001_k1.jpg              # Real composite image (<1mm fraction, 42 MB)
└── composite_S002_1.5mm.jpg           # Real composite image (1-2mm fraction, 48 MB)
```

## Example Images

### composite_S001_k1.jpg

**Properties**:
- **Size Fraction**: k1 (<1mm)
- **File Size**: 42 MB
- **Source**: Sample 331, k1 fraction from the study
- **Resolution**: High-resolution composite image
- **Content**: Multiple small arthropod specimens on white background

**Expected Processing**:
- Detection and segmentation of small arthropods
- Suitable for testing the pipeline's ability to handle tiny specimens

### composite_S002_1.5mm.jpg

**Properties**:
- **Size Fraction**: 1.5mm (1-2mm)
- **File Size**: 48 MB
- **Source**: Sample 331, 1.5mm fraction from the study
- **Resolution**: High-resolution composite image
- **Content**: Multiple arthropod specimens in the 1-2mm size range

**Expected Processing**:
- Detection and segmentation of medium-sized arthropods
- Demonstrates pipeline performance across different size fractions

## Metadata CSV Format

The metadata CSV requires the following columns:

- **image_path**: Relative or absolute path to composite image
- **sample_id**: Sample identifier (e.g., S001, S002)
- **size_fraction**: Size fraction code (k1, 1, 2, 3, 5, 7, A)

Example (current metadata):
```csv
image_path,sample_id,size_fraction
data/examples/composite_S001_k1.jpg,S001,k1
data/examples/composite_S002_1.5mm.jpg,S002,1
```

## Running the Example

### Single Image Mode
```bash
python scripts/02_process_images.py \
    --input data/examples/composite_S001_k1.jpg \
    --sample-id S001 \
    --size-fraction k1 \
    --output-dir ./output/test_run
```

### Metadata CSV Mode (Recommended)
```bash
python scripts/02_process_images.py \
    --metadata data/examples/metadata_example.csv \
    --output-dir ./output/test_run
```

### Directory Batch Mode
```bash
python scripts/02_process_images.py \
    --input-dir data/examples \
    --pattern "composite_*.jpg" \
    --output-dir ./output/test_run
```

## Expected Output

After processing the example images, you should see extracted specimen cutouts:

```
output/test_run/
├── S001/
│   └── k1/
│       ├── S001_k1_0001.png
│       ├── S001_k1_0002.png
│       ├── S001_k1_0003.png
│       └── metadata.json
├── S002/
│   └── 1/
│       ├── S002_1_0001.png
│       ├── S002_1_0002.png
│       └── metadata.json
└── processing_results.csv
```

The number of extracted specimens will depend on the actual arthropod count in each image.

## Notes

- **Models Included**: The detection and segmentation models (`scale_cutout2.pt`, `scale_segmentation2.pt`) are included in `data/models/`. These are the actual models used in the study.

- **Image Size**: The example images are high-resolution (42-48 MB) and representative of actual study data. Processing may take several minutes depending on hardware.

- **GPU vs CPU**: GPU is highly recommended for these large images. Processing time:
  - With GPU: ~2-5 minutes per image
  - With CPU: ~10-30 minutes per image

- **Expected Detections**:
  - composite_S001_k1.jpg: Multiple small arthropods (<1mm)
  - composite_S002_1.5mm.jpg: Medium-sized arthropods (1-2mm)

- **Troubleshooting**: Check logs in `./logs/pipeline.log` for detailed processing information

## Performance Testing

To verify the pipeline works correctly with these examples:

```bash
# Quick test (just detection and segmentation)
python scripts/test_core_components.py

# Full pipeline test
python scripts/02_process_images.py \
    --metadata data/examples/metadata_example.csv \
    --output-dir ./output/example_test
```

## Using Your Own Images

To test with your own composite images:

1. Place images in this directory
2. Update `metadata_example.csv` with your image paths and metadata
3. Run the processing script as shown above
