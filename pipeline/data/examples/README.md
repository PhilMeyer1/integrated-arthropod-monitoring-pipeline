# Example Data

This directory contains example data for testing the arthropod classification pipeline.

## Directory Structure

```
examples/
├── README.md                          # This file
├── metadata_example.csv               # Example metadata for batch processing
├── composite_S001_1mm.png            # Example composite image (1-2mm fraction)
├── composite_S002_2mm.png            # Example composite image (2-3mm fraction)
└── composite_S003_k1.png             # Example composite image (<1mm fraction)
```

## Creating Test Images

Since we cannot include actual specimen images in the repository, you need to:

1. **For testing with your own data:**
   - Place composite images in this directory
   - Update `metadata_example.csv` with correct paths and metadata

2. **For creating synthetic test images:**
   ```python
   # Create a simple test composite image
   import cv2
   import numpy as np

   # Create blank white image (similar to actual composite backgrounds)
   img = np.ones((3000, 4000, 3), dtype=np.uint8) * 248

   # Add some simple specimen-like shapes (circles/ellipses)
   for i in range(10):
       x = np.random.randint(100, 3900)
       y = np.random.randint(100, 2900)
       size = np.random.randint(20, 100)
       color = (np.random.randint(50, 150), np.random.randint(50, 150), np.random.randint(50, 150))
       cv2.circle(img, (x, y), size, color, -1)

   cv2.imwrite('data/examples/composite_S001_1mm.png', img)
   ```

3. **Using actual data (recommended):**
   - Copy a few real composite images from your dataset
   - Rename them to match the naming convention
   - Update the metadata CSV

## Metadata CSV Format

The metadata CSV requires the following columns:

- **image_path**: Relative or absolute path to composite image
- **sample_id**: Sample identifier (e.g., S001, S002)
- **size_fraction**: Size fraction code (k1, 1, 2, 3, 5, 7, A)

Example:
```csv
image_path,sample_id,size_fraction
data/examples/composite_S001_1mm.png,S001,1
data/examples/composite_S002_2mm.png,S002,2
```

## Running the Example

### Single Image Mode
```bash
python scripts/02_process_images.py \
    --input data/examples/composite_S001_1mm.png \
    --sample-id S001 \
    --size-fraction 1 \
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
    --pattern "composite_*.png" \
    --output-dir ./output/test_run
```

## Expected Output

After processing, you should see:

```
output/test_run/
├── S001/
│   └── 1/
│       ├── S001_1_0001.png
│       ├── S001_1_0002.png
│       └── metadata.json
├── S002/
│   └── 2/
│       ├── S002_2_0001.png
│       └── metadata.json
└── processing_results.csv
```

## Notes

- **YOLO Models Required**: Ensure you have downloaded the detection and segmentation models first:
  ```bash
  python scripts/download_models.py
  ```

- **Image Size**: Example images should be reasonably sized (e.g., 2000-5000px) to test tiling properly

- **GPU vs CPU**: First run will be slower as YOLO models are loaded. GPU is recommended for larger images.

- **Troubleshooting**: Check logs in `./logs/pipeline.log` for detailed processing information
