# Export Module

Export classification results and statistics in various formats.

## Quick Start

```python
from pathlib import Path
import pandas as pd
from src.export import ExcelExporter, CSVExporter, StatisticsCalculator

# Load results
results_df = pd.read_csv('inference_results.csv')

# 1. Export to Excel (formatted, multi-sheet)
excel_exporter = ExcelExporter()
excel_exporter.export_classification_results(
    results_df,
    Path('./results.xlsx'),
    include_hierarchy=True,
    include_statistics=True
)

# 2. Export to CSV (for R/Python analysis)
csv_exporter = CSVExporter()
csv_exporter.export_batch(
    results_df,
    Path('./exports'),
    prefix='set1'
)

# 3. Calculate and export statistics
calculator = StatisticsCalculator()
stats = calculator.calculate_comprehensive_statistics(results_df)
calculator.export_for_publication(stats, Path('./publication_stats.json'))
```

## Modules

### Excel Export (`excel_export.py`)

Multi-sheet Excel files with formatting:

```python
exporter = ExcelExporter()

# Full export with all sheets
exporter.export_classification_results(
    results_df,
    Path('./results.xlsx'),
    include_hierarchy=True,  # Add taxonomy sheet
    include_statistics=True   # Add stats sheets
)

# Compare multiple model sets
exporter.export_model_comparison(
    set_numbers=[1, 2, 3],
    output_path=Path('./comparison.xlsx')
)

# Export specimen details
exporter.export_specimen_details(
    image_ids=[1, 2, 3, 4, 5],
    output_path=Path('./specimens.xlsx')
)
```

**Excel Output:**
- Sheet 1: Classification Results (all predictions)
- Sheet 2: Summary Statistics (overall metrics)
- Sheet 3: Per-Taxon Statistics (metrics per model)
- Sheet 4: Taxonomy Hierarchy (full tree)

**Features:**
- Colored headers
- Auto-adjusted column widths
- Frozen header row
- Professional formatting

### CSV Export (`csv_export.py`)

Simple CSV files for analysis:

```python
exporter = CSVExporter()

# Single file export
exporter.export_results(results_df, Path('./results.csv'))

# Statistics export
exporter.export_statistics(results_df, Path('./stats.csv'))

# Confusion matrix data
exporter.export_confusion_matrix_data(
    results_df,
    Path('./confusion.csv'),
    model_taxon='Insecta'  # Optional: specific taxon
)

# Hierarchical classification paths
exporter.export_hierarchical_path(
    results_df,
    Path('./paths.csv')
)

# Batch export (all files at once)
exporter.export_batch(
    results_df,
    Path('./exports'),
    prefix='set1'
)
```

**Batch Export Creates:**
- `set1_results.csv` - Full results
- `set1_statistics.csv` - Summary statistics
- `set1_confusion.csv` - Confusion matrix data
- `set1_paths.csv` - Hierarchical paths

### Statistics (`statistics.py`)

Comprehensive statistical analysis:

```python
calculator = StatisticsCalculator()

# Calculate all statistics
stats = calculator.calculate_comprehensive_statistics(results_df)

# Individual metric calculations
overall_metrics = calculator.calculate_multiclass_metrics(results_df)
per_taxon = calculator.calculate_per_taxon_metrics(results_df)
errors = calculator.analyze_errors(results_df, top_n=10)
conf_dist = calculator.calculate_confidence_distribution(results_df)

# Publication-ready table
performance_table = calculator.create_performance_table(results_df)
print(performance_table.to_latex())

# Export for publication
calculator.export_for_publication(stats, Path('./pub_stats.json'))
```

**Statistics Included:**
- Overall accuracy, precision, recall, F1
- Per-taxon metrics
- Confidence distributions
- Error analysis (top confusion pairs)
- Test set vs all data comparison
- Hierarchical accuracy

## Usage Examples

### For Publications

```python
# Get publication-ready statistics
calculator = StatisticsCalculator()
stats = calculator.calculate_comprehensive_statistics(results_df)

# Create LaTeX table
table = calculator.create_performance_table(results_df)
with open('performance.tex', 'w') as f:
    f.write(table.to_latex(index=False))

# Export comprehensive stats as JSON
calculator.export_for_publication(stats, Path('./supplementary_stats.json'))
```

### For Analysis in R

```python
# Export everything to CSV for R
csv_exporter = CSVExporter()
csv_exporter.export_batch(
    results_df,
    Path('./r_analysis'),
    prefix='arthropod'
)

# In R:
# results <- read.csv('r_analysis/arthropod_results.csv')
# stats <- read.csv('r_analysis/arthropod_statistics.csv')
```

### Compare Multiple Models

```python
# Export comparison of model sets 1, 2, 3
excel_exporter = ExcelExporter()
excel_exporter.export_model_comparison(
    set_numbers=[1, 2, 3],
    output_path=Path('./model_comparison.xlsx')
)
```

## Output Formats

### Excel (.xlsx)
- **Use for:** Viewing, sharing with collaborators
- **Pros:** Formatted, multiple sheets, easy to browse
- **Cons:** Larger file size, harder to version control

### CSV (.csv)
- **Use for:** Analysis in R/Python, version control
- **Pros:** Universal format, easy to process
- **Cons:** No formatting, single file per dataset

### JSON (.json)
- **Use for:** Metadata, structured data, web APIs
- **Pros:** Hierarchical structure, easy to parse
- **Cons:** Not human-friendly for large datasets

## Configuration

No specific configuration needed. All exporters work standalone.

## Tips

1. **Use batch export for complete analysis:**
   ```python
   csv_exporter.export_batch(results_df, Path('./analysis'))
   ```

2. **Always include test set filtering:**
   ```python
   test_results = results_df[results_df['is_excluded'] == True]
   calculator.calculate_comprehensive_statistics(test_results)
   ```

3. **Export thresholds with results:**
   ```python
   csv_exporter.export_thresholds(thresholds, Path('./thresholds.csv'))
   ```

4. **For publications, use both Excel and JSON:**
   - Excel for supplementary tables
   - JSON for exact numerical values

## Methods Section Reference

Corresponds to:
- "Data Export and Analysis"
- "Statistical Analysis"
- "Performance Evaluation"
