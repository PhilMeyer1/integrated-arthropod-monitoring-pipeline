# Publication Data

This directory contains the raw data underlying the manuscript figures and analyses, following best practices for data availability.

---

## Files Overview

| File | Records | Size | Description |
|------|---------|------|-------------|
| `specimen_data.xlsx` | 46,012 specimens | 1.5 MB | Individual specimen classifications |
| `biomass_data.xlsx` | 378 measurements | 23 KB | Biomass and abundance (54 samples × 7 size fractions) |
| `model_performance.xlsx` | 73 taxa | 15 KB | Per-taxon classification performance |

**Total:** ~1.54 MB

---

## File Descriptions

### specimen_data.xlsx

**Description:** Individual specimen classifications from the automated hierarchical classification pipeline

**Records:** 46,012 specimens

**Columns:**
- `specimen_id` - Unique specimen identifier
- `sample_id` - Sample identifier (links to biomass_data.xlsx)
- `size_fraction` - Size fraction category
- `phylum` - Phylum-level classification
- `class` - Class-level classification
- `order` - Order-level classification
- `family` - Family-level classification
- `hedges_present (Eco Treatment)` - Ecological treatment (hedgerow presence)

**Source:** Output from YOLOv11-based hierarchical classification pipeline

**Corresponds to:**
- Figure 2: Model performance across taxonomic levels
- Figure 3: Classification accuracy and confidence distributions
- Table S1: Complete specimen classifications
- Supplementary Data

**Usage:**
```python
import pandas as pd
specimens = pd.read_excel('specimen_data.xlsx')
print(f"Total specimens: {len(specimens)}")
print(specimens.groupby('family').size())
```

---

### biomass_data.xlsx

**Description:** Biomass and abundance data from field sampling, organized by sample and size fraction

**Records:** 378 measurements (54 field samples × 7 size fractions)

**Columns:**
- `sample_id` - Unique sample identifier
- `biomass_total` - Total biomass (mg)
- `year` - Sampling year
- `size` - Size fraction category
- `biomass_size` - Biomass for specific size fraction (mg)
- `abundance` - Specimen count

**Source:** Field sampling and laboratory measurements

**Corresponds to:**
- Figure 4: Ecological analyses (biomass, abundance)
- Manuscript Results section: Biodiversity patterns
- Ecological treatment comparisons

**Usage:**
```python
import pandas as pd
biomass = pd.read_excel('biomass_data.xlsx')
print(biomass.groupby('size')['biomass_total'].sum())
```

---

### model_performance.xlsx

**Description:** Per-taxon classification performance metrics from the hierarchical classification models

**Records:** 73 taxa (families and orders)

**Columns:**
- `Actual_Scientific_Name` - Taxon scientific name
- `Actual_Rank` - Taxonomic rank (order or family)
- `val_taxon_count` - Validation set size
- `train_taxon_count` - Training set size
- `test_count` - Test set size
- `average_confidence` - Mean confidence score
- `correct_classifications` - Number of correct predictions
- `incorrect_classifications` - Number of incorrect predictions
- `accuracy_without_threshold` - Baseline accuracy (%)
- `threshold_for_accuracy` - Calibrated confidence threshold

**Source:** Model evaluation on test set (10% of data, see Methods)

**Corresponds to:**
- Figure 2a-c: Accuracy distributions across taxa
- Table S3: Per-taxon thresholds and accuracies
- Supplementary Methods: Threshold calibration procedure
- Methods section: "Threshold Calculation"

**Usage:**
```python
import pandas as pd
performance = pd.read_excel('model_performance.xlsx')
valid_taxa = performance[performance['accuracy_without_threshold'] >= 95]
print(f"Taxa achieving ≥95% accuracy: {len(valid_taxa)}/{len(performance)}")
```

---

## Data Provenance

**Study:** Hedgerow effects on arthropod biodiversity (manuscript title)

**Methods:**
- Image acquisition: High-resolution digital photography of arthropod specimens
- Detection: YOLOv8 object detection with adaptive tiling
- Segmentation: YOLOv8 instance segmentation
- Classification: YOLOv11 hierarchical classification following Catalogue of Life taxonomy
- Threshold calibration: 95% accuracy criterion (see Methods and THRESHOLD_CALIBRATION.md)

**Data Collection Period:** 2023-2024 (field sampling)

**Processing:** Automated pipeline (see main README.md)

---

## Data Quality

### Specimen Data
- **Validation:** All specimens manually verified during model training (10% test set)
- **Coverage:** All arthropod orders present in samples
- **Size Fractions:** 7 fractions from 1.5 mm to >10 mm

### Biomass Data
- **Measurement:** Laboratory-grade scales (±0.001 mg precision)
- **Quality Control:** Replicate measurements for samples >100 mg

### Model Performance
- **Test Set:** Independent 10% holdout (never seen during training)
- **Evaluation:** Per-taxon accuracy, confidence, and threshold calibration
- **Criteria:** ≥95% accuracy with calibrated confidence thresholds

---

## Citation

If you use this data, please cite:

> Meyer, P., Scharnhorst, V., Lechner, M., Haslinger, H., Gierus, M., & Meimberg, H. (2025). Integrated arthropod monitoring pipeline detects size-dependent responses. Manuscript in review.

**Data Repository:** Zenodo DOI: 10.5281/zenodo.17512687

---

## License

This data is released under [CC-BY 4.0 License](https://creativecommons.org/licenses/by/4.0/).

You are free to:
- Share: Copy and redistribute the material
- Adapt: Remix, transform, and build upon the material

Under the following terms:
- Attribution: You must give appropriate credit and indicate if changes were made

---

## Contact

For questions about the data:

**Corresponding Author:** Philipp Meyer
**Email:** philipp.meyer@boku.ac.at
**Institution:** University of Natural Resources and Life Sciences, Vienna (BOKU)

---

## Related Files

- **Code Repository:** See parent directory for complete classification pipeline
- **Models:** Trained YOLO models available upon request (large files)
- **Documentation:**
  - `../../README.md` - Pipeline overview and usage
  - `../../THRESHOLD_CALIBRATION.md` - Threshold methodology

---

**Last Updated:** 2025-11-11
**Version:** 1.0
