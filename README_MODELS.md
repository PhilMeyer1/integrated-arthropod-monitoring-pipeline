# Trained Models and Example Data

This archive contains the trained YOLO models and example composite images used in the publication:

**Meyer, P., Scharnhorst, V., Lechner, M., Haslinger, H., Gierus, M., & Meimberg, H. (2025)**
*Integrated arthropod monitoring pipeline detects size-dependent responses*
Current Biology. https://doi.org/10.5281/zenodo.17512687

## Contents

### Detection Models (`models/detection/`)

**Purpose**: Detect individual arthropod specimens on composite images

**Architecture**: YOLOv8-based object detection

#### `detect_k1.pt` (22 MB)
- **Target**: Very small arthropods (<1mm, k1 size fraction)
- **Training**: Specialized dataset of <1mm specimens
- **Detection class**: "Area" (arthropod regions)
- **Usage parameters**:
  - Confidence threshold: 0.01 (very low for maximum sensitivity)
  - IOU threshold: 0.9 (high for detection phase)

#### `detect_others.pt` (6.1 MB)
- **Target**: Medium to large arthropods (1-10mm)
- **Size fractions**: 1, 2, 3, 5, 7 mm
- **Training**: Combined dataset across multiple size fractions
- **Detection class**: "Area" (arthropod regions)
- **Usage parameters**:
  - Confidence threshold: 0.01
  - IOU threshold: 0.9

### Segmentation Models (`models/segmentation/`)

**Purpose**: Segment individual arthropod specimens for precise extraction

**Architecture**: YOLOv8-based instance segmentation

#### `segment_k1.pt` (23 MB)
- **Target**: Very small arthropods (<1mm, k1 size fraction)
- **Training**: Specialized dataset with pixel-level annotations
- **Segmentation class**: "Area" (arthropod regions)
- **Usage parameters**:
  - Confidence threshold: 0.05
  - IOU threshold: 0.1 (low for segmentation NMS)
  - Mask dilation factor: 1.02

#### `segment_others.pt` (23 MB)
- **Target**: Medium to large arthropods (1-10mm)
- **Size fractions**: 1, 2, 3, 5, 7 mm
- **Training**: Combined dataset with pixel-level annotations
- **Segmentation class**: "Area" (arthropod regions)
- **Usage parameters**:
  - Confidence threshold: 0.05
  - IOU threshold: 0.1
  - Mask dilation factor: 1.02

### Example Images (`examples/`)

#### `composite_S001_k1.jpg` (42 MB)
- Sample ID: S001
- Size fraction: k1 (<1mm specimens)
- Dimensions: High-resolution composite image
- Specimens: ~879 arthropods detected
- Purpose: Demonstrates detection and segmentation on very small specimens

#### `composite_S002_1.5mm.jpg` (48 MB)
- Sample ID: S002
- Size fraction: 1-2mm specimens
- Dimensions: High-resolution composite image
- Specimens: ~245 arthropods detected
- Purpose: Demonstrates detection and segmentation on medium-sized specimens

#### `metadata_example.csv`
- Example metadata file for batch processing
- Contains sample IDs, size fractions, and file paths
- Required format for running the pipeline scripts

## Usage

### Basic Detection and Segmentation

```python
from ultralytics import YOLO

# For k1 specimens
detector = YOLO('models/detection/detect_k1.pt')
segmenter = YOLO('models/segmentation/segment_k1.pt')

# Run detection
detections = detector.predict('examples/composite_S001_k1.jpg', conf=0.01, iou=0.9)

# Run segmentation
segments = segmenter.predict('examples/composite_S001_k1.jpg', conf=0.05, iou=0.1)
```

### Using the Pipeline

```bash
# Process example images with the full pipeline
python pipeline/examples/02_process_images.py \
    --metadata examples/metadata_example.csv \
    --output-dir ./output/specimens
```

See the main repository for complete documentation:
https://github.com/PhilMeyer1/integrated-arthropod-monitoring-pipeline

## Classification Models

The hierarchical classification models used in the publication are **not included** in this archive to protect commercial interests.

These models are available upon reasonable request for academic research purposes. Please contact the lead author:

**Philipp Meyer**
Email: philipp.meyer@boku.ac.at
Institution: University of Natural Resources and Life Sciences (BOKU), Vienna, Austria

## Model Performance

Detailed performance metrics for all models are documented in the publication and in the analysis data files in the main repository.

## Training Details

All models were trained using the Ultralytics YOLOv8/YOLOv11 framework with the following specifications:

- **Detection models**: YOLOv8n architecture
- **Segmentation models**: YOLOv8n-seg architecture
- **Image size**: Variable (optimized per size fraction)
- **Data augmentation**: HSV augmentation, 90°/180°/270° rotations
- **Hardware**: CUDA-enabled GPU

Training code and configuration are available in the main repository.

## License

These models are provided under the MIT License for academic and research purposes.

**Commercial use requires separate licensing.** Please contact the authors for commercial licensing inquiries.

## Citation

If you use these models in your research, please cite:

```bibtex
@software{meyer2025_arthropod_models,
  author = {Meyer, Philipp and Scharnhorst, Victor and Lechner, Michael and
            Haslinger, Hanna and Gierus, Martin and Meimberg, Harald},
  title = {Detection and Segmentation Models for Arthropod Monitoring Pipeline},
  year = {2025},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.XXXXXXX},
  url = {https://zenodo.org/records/XXXXXXX}
}

@article{meyer2025_arthropod_pipeline,
  title = {Integrated arthropod monitoring pipeline detects size-dependent responses},
  author = {Meyer, Philipp and Scharnhorst, Victor and Lechner, Michael and
            Haslinger, Hanna and Gierus, Martin and Meimberg, Harald},
  journal = {Current Biology},
  year = {2025},
  doi = {10.5281/zenodo.17512687}
}
```

## Contact

For questions about the models or data:

**Philipp Meyer**
Email: philipp.meyer@boku.ac.at
Institution: University of Natural Resources and Life Sciences (BOKU)
Vienna, Austria

## Acknowledgments

This research was conducted at BOKU University, Vienna, Austria. We thank all contributors to the arthropod monitoring project.

---

**Version**: 1.0.0
**Last Updated**: 2025-01-20
**Archive Size**: ~165 MB
