# Confidence Threshold Calibration

## Overview

This document describes the confidence threshold calibration method implemented in this pipeline, corresponding to the **"Threshold Calculation"** section of the Methods.

## Methodology

### Algorithm

For each taxonomic class (or globally), thresholds are calibrated on the test set using the following algorithm:

1. **Calculate baseline accuracy** (without threshold filtering)
2. **If baseline accuracy ≥ 95%**: Set threshold to **0.95**
3. **Otherwise**:
   - Start at threshold = **0.95**
   - Increment by **0.0001** (0.01%)
   - Continue until accuracy ≥ 95% OR threshold = 1.0
4. **Exclude classes** that cannot achieve 95% accuracy

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Target accuracy | 95% | Minimum accuracy required to retain a class |
| Start threshold | 0.95 | Initial confidence threshold |
| Increment | 0.0001 | Step size for threshold search |
| Maximum threshold | 1.0 | Upper limit (even if 95% not achieved) |

### Rationale

**Why 95% accuracy?**
- Ensures high reliability of classifications
- Balances precision (minimizing false positives) with recall (retaining true positives)
- Established standard in taxonomic classification studies

**Why start at 0.95?**
- Most deep learning models produce well-calibrated confidence scores
- Starting high reduces computational cost
- Taxa with good baseline accuracy retain more specimens

**Why 0.0001 increments?**
- Fine-grained search ensures optimal threshold
- Minimizes undetermined specimens while maintaining accuracy
- Corresponds to 0.01% steps in confidence

## Implementation

### Code Structure

```
src/classification/thresholds.py
└── ThresholdCalibrator
    ├── load_results()              # Load inference results
    ├── calibrate_thresholds()       # Main calibration function
    ├── calibrate_threshold_for_taxon()  # Per-taxon calibration
    ├── calculate_baseline_accuracy()    # Baseline accuracy calculation
    ├── calculate_accuracy_at_threshold()  # Accuracy at specific threshold
    ├── get_valid_taxa()             # Taxa passing criterion
    ├── get_invalid_taxa()           # Taxa failing criterion
    ├── save_thresholds()            # Save to JSON
    └── apply_thresholds()           # Apply to filter results
```

### Usage

#### 1. Calibrate Thresholds

```bash
# Basic usage (uses Methods defaults)
python scripts/calibrate_thresholds.py --set-number 1

# Custom parameters
python scripts/calibrate_thresholds.py \
    --set-number 1 \
    --target-accuracy 95.0 \
    --start-threshold 0.90 \
    --increment 0.0001

# Save to custom path
python scripts/calibrate_thresholds.py \
    --set-number 1 \
    --output ./results/my_thresholds.json
```

#### 2. Programmatic Usage

```python
from src.classification.thresholds import ThresholdCalibrator
from src.database.utils import get_data_manager

# Initialize
data_manager = get_data_manager(use_database=True)
calibrator = ThresholdCalibrator(
    target_accuracy=95.0,
    start_threshold=0.95,
    increment=0.0001
)

# Load inference results
calibrator.load_results(set_number=1, only_test_set=True)

# Calibrate per-taxon thresholds
thresholds = calibrator.calibrate_thresholds(per_taxon=True)

# Get valid taxa
valid_taxa = calibrator.get_valid_taxa(thresholds)
print(f"Valid taxa: {len(valid_taxa)}")

# Save results
calibrator.save_thresholds(thresholds, Path('thresholds.json'))

# Apply to filter results
filtered_df = calibrator.apply_thresholds(
    results_df,
    thresholds,
    taxon_column='predicted_taxon_id'
)
```

## Output Format

### Threshold JSON Structure

```json
{
  "Formicidae": {
    "taxon_id": "Formicidae",
    "threshold": 0.95,
    "baseline_accuracy": 98.5,
    "accuracy_with_threshold": 98.5,
    "correct_with_threshold": 197,
    "incorrect_with_threshold": 3,
    "undetermined": 0,
    "test_count": 200,
    "passes_criterion": true
  },
  "Braconidae": {
    "taxon_id": "Braconidae",
    "threshold": 0.9863,
    "baseline_accuracy": 83.0,
    "accuracy_with_threshold": 95.2,
    "correct_with_threshold": 40,
    "incorrect_with_threshold": 2,
    "undetermined": 8,
    "test_count": 50,
    "passes_criterion": true
  },
  "Proctotrupidae": {
    "taxon_id": "Proctotrupidae",
    "threshold": 1.0,
    "baseline_accuracy": 14.3,
    "accuracy_with_threshold": 66.7,
    "correct_with_threshold": 2,
    "incorrect_with_threshold": 1,
    "undetermined": 4,
    "test_count": 7,
    "passes_criterion": false
  }
}
```

### Summary CSV Columns

| Column | Description |
|--------|-------------|
| `taxon_id` | Taxonomic identifier |
| `threshold` | Calibrated confidence threshold |
| `baseline_accuracy` | Accuracy without threshold (%) |
| `accuracy_with_threshold` | Accuracy at calibrated threshold (%) |
| `correct` | Correct predictions above threshold |
| `incorrect` | Incorrect predictions above threshold |
| `undetermined` | Predictions below threshold |
| `test_count` | Total test set size |
| `passes_criterion` | Whether ≥95% accuracy achieved |

## Interpretation

### Valid Taxa (passes_criterion = true)

Taxa that achieve ≥95% accuracy are **retained for downstream analysis**.

**Example: Formicidae**
- Baseline accuracy: 98.5%
- Threshold: 0.95 (baseline ≥ 95%)
- Retained: 200/200 specimens (100%)
- High confidence, no specimens excluded

**Example: Braconidae**
- Baseline accuracy: 83.0%
- Threshold: 0.9863 (incremented to achieve 95%)
- Retained: 42/50 specimens (84%)
- 8 specimens below threshold → "undetermined"

### Invalid Taxa (passes_criterion = false)

Taxa that **cannot achieve 95% accuracy** are **excluded** from downstream analysis.

**Example: Proctotrupidae**
- Baseline accuracy: 14.3%
- Threshold: 1.0 (maxed out)
- Final accuracy: 66.7% (still < 95%)
- **Excluded** from further analysis

**Reasons for exclusion:**
- Insufficient training data
- Morphologically cryptic (similar to other taxa)
- High intra-taxon variation
- Poor image quality in this size fraction

## Comparison with Original Implementation

### Changes from GUI Code

The original GUI implementation had a discrepancy:

```python
# OLD (GUI code - INCORRECT)
if baseline_accuracy >= 95%:
    threshold = 0.9  # ❌ Too low!

# NEW (Repository code - CORRECT)
if baseline_accuracy >= 95%:
    threshold = 0.95  # ✓ Matches Methods
```

**Impact:**
- Old code used **0.9** for high-performing taxa
- This retained more specimens but was inconsistent with Methods
- New code uses **0.95** as specified in manuscript

### Changes from Old Repository Code

The old repository implementation used F1-score optimization:

```python
# OLD (Repository - DIFFERENT APPROACH)
method = 'f1'  # F1-score optimization
step = 0.05    # Coarse steps
# No 95% accuracy criterion

# NEW (Repository - MATCHES METHODS)
target_accuracy = 95.0  # Accuracy-based
increment = 0.0001      # Fine steps
# Explicit 95% criterion
```

**Impact:**
- Old code optimized for F1-score, not accuracy
- Used coarse 5% steps instead of 0.01% steps
- Did not enforce 95% accuracy requirement
- New code exactly implements Methods description

## Validation

### Expected Results

When calibrating on typical arthropod data:

- **~70-80% of families** achieve ≥95% accuracy
- **~15-25% of families** excluded due to <95% accuracy
- **Threshold distribution:**
  - 40-50% use baseline threshold (0.95)
  - 30-40% increment to 0.95-0.99
  - 10-20% max out at 1.0 (excluded)

### Quality Checks

After calibration, verify:

1. **Valid taxa count**: Should retain majority of taxa
2. **Undetermined rate**: Typically 5-15% of specimens
3. **Excluded taxa**: Check for known cryptic groups (e.g., Braconidae, Ichneumonidae, Cecidomyiidae)
4. **Threshold range**: Most between 0.95-0.99

## Correspondence with Methods

This implementation directly corresponds to the manuscript Methods section:

> "We calibrated confidence thresholds per class using test-set predictions, starting at 95% and incrementing by 0.01% until achieving ≥95% accuracy or reaching 100% confidence (Table S3). After thresholding, accuracies rose substantially: order-level to 99.2% (standard deviation 1.4%), family-level to 99.7% (standard deviation 0.9%), with all retained classes exceeding 96% and misclassifications falling 66% (from 47 to 16 images; Fig. 2a–c)."

**Key points:**
- ✓ Start at 95% (0.95 threshold)
- ✓ Increment by 0.01% (0.0001 steps)
- ✓ Target ≥95% accuracy
- ✓ Maximum 100% (1.0 threshold)
- ✓ Per-class calibration
- ✓ Classes excluded if <95% not achievable

## References

- Methods section: "Threshold Calculation"
- Figure 2: Model performance across taxa
- Table S3: Per-class thresholds and accuracies
- Supplementary Methods: Detailed threshold calibration procedure

---

**Last updated:** 2025-01-11
**Authors:** Philipp Meyer
**Correspondence:** philipp.meyer@boku.ac.at
