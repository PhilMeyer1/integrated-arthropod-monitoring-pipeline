#!/usr/bin/env python3
"""
Export Classification Results

This script exports classification results to publication-ready formats (Excel, CSV).
It applies confidence thresholds, calculates statistics, and formats data for analysis.

Usage:
    python scripts/05_export_results.py \\
        --results ./results/classification_results_set_1.json \\
        --output ./exports/results.xlsx \\
        --format excel

Methods correspondence:
    - Data Export (Methods section: Data Export)
    - Threshold Calculation (Methods section: Threshold Calculation)
"""

import argparse
import sys
from pathlib import Path
import logging
import json
from typing import List, Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.export.excel_export import ExcelExporter
from src.export.csv_export import CSVExporter
from src.export.statistics import StatisticsCalculator
from src.classification.thresholds import ThresholdOptimizer
from src.utils.logging_config import setup_logging, LogContext
from src.utils.validation import validate_file_exists


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Export classification results to publication-ready formats',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input arguments
    parser.add_argument(
        '--results',
        type=Path,
        required=True,
        help='Path to classification results JSON file'
    )
    parser.add_argument(
        '--thresholds',
        type=Path,
        help='Path to thresholds JSON file (optional, for filtering)'
    )

    # Output arguments
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output file path (extension determines format if not specified)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['excel', 'csv', 'both'],
        help='Output format (auto-detected from --output extension if not specified)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory (alternative to --output for batch export)'
    )

    # Export options
    parser.add_argument(
        '--include-per-taxon',
        action='store_true',
        help='Include per-taxon statistics in Excel export'
    )
    parser.add_argument(
        '--include-per-sample',
        action='store_true',
        help='Include per-sample statistics in Excel export'
    )
    parser.add_argument(
        '--include-hierarchy',
        action='store_true',
        help='Include hierarchical path in export'
    )
    parser.add_argument(
        '--include-confidence',
        action='store_true',
        default=True,
        help='Include confidence scores in export'
    )

    # Filtering options
    parser.add_argument(
        '--apply-thresholds',
        action='store_true',
        help='Apply confidence thresholds to filter results (Methods: 95%% accuracy target)'
    )
    parser.add_argument(
        '--min-confidence',
        type=float,
        help='Minimum confidence threshold (0.0-1.0)'
    )
    parser.add_argument(
        '--exclude-below-threshold',
        action='store_true',
        help='Exclude (not just flag) results below threshold'
    )

    # Metadata options
    parser.add_argument(
        '--sample-metadata',
        type=Path,
        help='CSV file with additional sample metadata to merge'
    )
    parser.add_argument(
        '--hedge-metadata',
        type=Path,
        help='CSV file with hedge metadata from sampling sites'
    )

    # Processing options
    parser.add_argument(
        '--group-by',
        type=str,
        choices=['sample', 'taxon', 'size_fraction', 'none'],
        default='none',
        help='Group results by field'
    )
    parser.add_argument(
        '--sort-by',
        type=str,
        default='confidence',
        choices=['confidence', 'taxon', 'sample_id', 'image_path'],
        help='Sort results by field'
    )
    parser.add_argument(
        '--sort-descending',
        action='store_true',
        help='Sort in descending order'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    return parser.parse_args()


def validate_arguments(args, logger):
    """Validate command line arguments."""
    # Check input file
    validate_file_exists(args.results, 'Results JSON')

    if args.thresholds:
        validate_file_exists(args.thresholds, 'Thresholds JSON')

    if args.sample_metadata:
        validate_file_exists(args.sample_metadata, 'Sample metadata CSV')

    if args.hedge_metadata:
        validate_file_exists(args.hedge_metadata, 'Hedge metadata CSV')

    # Determine output format
    if not args.format:
        # Auto-detect from extension
        suffix = args.output.suffix.lower()
        if suffix in ['.xlsx', '.xls']:
            args.format = 'excel'
        elif suffix == '.csv':
            args.format = 'csv'
        else:
            logger.error(f"Cannot determine format from extension: {suffix}")
            logger.error("Please specify --format explicitly")
            sys.exit(1)

    # Create output directory
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"✓ Arguments validated")
    logger.info(f"  Results: {args.results}")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  Format: {args.format}")
    if args.apply_thresholds:
        logger.info(f"  Applying thresholds: Yes")


def load_results(args, logger):
    """Load classification results from JSON."""
    logger.info(f"Loading classification results...")

    with open(args.results, 'r') as f:
        data = json.load(f)

    results = data.get('results', [])
    metadata = {
        'model_set': data.get('model_set'),
        'specimens_dir': data.get('specimens_dir'),
        'models_dir': data.get('models_dir'),
        'parameters': data.get('classification_parameters', {})
    }

    logger.info(f"✓ Results loaded")
    logger.info(f"  Total results: {len(results):,}")
    logger.info(f"  Model set: {metadata['model_set']}")

    return results, metadata


def load_thresholds(args, logger):
    """Load confidence thresholds if provided."""
    if not args.thresholds:
        return None

    logger.info(f"Loading thresholds from: {args.thresholds}")

    with open(args.thresholds, 'r') as f:
        thresholds = json.load(f)

    logger.info(f"✓ Thresholds loaded for {len(thresholds)} taxa")

    return thresholds


def apply_threshold_filtering(results, thresholds, args, logger):
    """Apply confidence thresholds to filter results."""
    if not args.apply_thresholds:
        return results

    logger.info("Applying confidence thresholds...")

    # Global minimum threshold
    min_conf = args.min_confidence if args.min_confidence is not None else 0.0

    filtered_results = []
    excluded_count = 0

    for result in results:
        taxon = result.get('predicted_taxon')
        confidence = result.get('confidence', 0.0)

        # Get taxon-specific threshold
        threshold = min_conf
        if thresholds and taxon in thresholds:
            threshold = max(threshold, thresholds[taxon]['threshold'])

        # Check if passes threshold
        passes = confidence >= threshold

        if args.exclude_below_threshold and not passes:
            excluded_count += 1
            continue

        # Add threshold info
        result['threshold'] = threshold
        result['passes_threshold'] = passes

        filtered_results.append(result)

    logger.info(f"✓ Thresholds applied")
    logger.info(f"  Original results: {len(results)}")
    logger.info(f"  Filtered results: {len(filtered_results)}")
    if args.exclude_below_threshold:
        logger.info(f"  Excluded: {excluded_count}")

    return filtered_results


def merge_metadata(results, args, logger):
    """Merge additional metadata into results."""
    if not args.sample_metadata and not args.hedge_metadata:
        return results

    import pandas as pd

    # Convert results to DataFrame
    df_results = pd.DataFrame(results)

    # Merge sample metadata
    if args.sample_metadata:
        logger.info(f"Merging sample metadata from: {args.sample_metadata}")
        df_samples = pd.read_csv(args.sample_metadata)

        # Merge on sample_id if available
        if 'sample_id' in df_results.columns and 'sample_id' in df_samples.columns:
            df_results = df_results.merge(df_samples, on='sample_id', how='left')
            logger.info(f"  ✓ Sample metadata merged")

    # Merge hedge metadata
    if args.hedge_metadata:
        logger.info(f"Merging hedge metadata from: {args.hedge_metadata}")
        df_hedge = pd.read_csv(args.hedge_metadata)

        # Merge on location or site ID
        merge_key = None
        for key in ['location', 'site_id', 'sampling_location']:
            if key in df_results.columns and key in df_hedge.columns:
                merge_key = key
                break

        if merge_key:
            df_results = df_results.merge(df_hedge, on=merge_key, how='left')
            logger.info(f"  ✓ Hedge metadata merged on '{merge_key}'")
        else:
            logger.warning(f"  ⚠ No matching key found for hedge metadata")

    # Convert back to list of dicts
    results = df_results.to_dict('records')

    return results


def calculate_comprehensive_statistics(results, logger):
    """Calculate comprehensive statistics."""
    logger.info("Calculating statistics...")

    calculator = StatisticsCalculator()

    with LogContext('Statistics calculation', logger):
        # Basic statistics
        basic_stats = calculator.calculate_basic_stats(results)

        # Per-taxon statistics
        per_taxon_stats = calculator.calculate_per_taxon_stats(results)

        # Per-sample statistics if sample_id available
        per_sample_stats = {}
        if results and 'sample_id' in results[0]:
            per_sample_stats = calculator.calculate_per_sample_stats(results)

        # Confidence statistics
        confidence_stats = calculator.calculate_confidence_stats(results)

    stats = {
        'basic': basic_stats,
        'per_taxon': per_taxon_stats,
        'per_sample': per_sample_stats,
        'confidence': confidence_stats
    }

    logger.info(f"✓ Statistics calculated")
    logger.info(f"  Total specimens: {basic_stats['total_specimens']:,}")
    logger.info(f"  Unique taxa: {basic_stats['unique_taxa']}")
    logger.info(f"  Mean confidence: {confidence_stats['mean']:.3f}")

    return stats


def sort_and_group_results(results, args, logger):
    """Sort and group results as specified."""
    import pandas as pd

    df = pd.DataFrame(results)

    # Sort
    if args.sort_by in df.columns:
        logger.info(f"Sorting by: {args.sort_by} ({'desc' if args.sort_descending else 'asc'})")
        df = df.sort_values(
            by=args.sort_by,
            ascending=not args.sort_descending
        )

    # Group if requested
    if args.group_by != 'none' and args.group_by in df.columns:
        logger.info(f"Grouping by: {args.group_by}")
        # For export, we keep all rows but sort by group
        df = df.sort_values(by=args.group_by)

    results = df.to_dict('records')

    return results


def export_results(results, stats, metadata, args, logger):
    """Export results to specified format(s)."""
    logger.info("Exporting results...")

    output_files = []

    # Determine output directory
    output_dir = args.output_dir if args.output_dir else args.output.parent

    # Excel export
    if args.format in ['excel', 'both']:
        logger.info("  Exporting to Excel...")

        exporter = ExcelExporter(output_dir=output_dir)

        excel_file = exporter.export(
            results=results,
            filename=args.output.name if args.format == 'excel' else f"{args.output.stem}.xlsx",
            statistics=stats['basic'],
            per_taxon_stats=stats['per_taxon'] if args.include_per_taxon else None,
            per_sample_stats=stats['per_sample'] if args.include_per_sample else None,
            include_hierarchy=args.include_hierarchy,
            metadata=metadata
        )

        logger.info(f"  ✓ Excel exported: {excel_file}")
        output_files.append(excel_file)

    # CSV export
    if args.format in ['csv', 'both']:
        logger.info("  Exporting to CSV...")

        exporter = CSVExporter(output_dir=output_dir)

        csv_file = exporter.export(
            results=results,
            filename=args.output.name if args.format == 'csv' else f"{args.output.stem}.csv",
            include_path=args.include_hierarchy,
            include_confidence=args.include_confidence
        )

        logger.info(f"  ✓ CSV exported: {csv_file}")
        output_files.append(csv_file)

        # Export statistics separately
        stats_file = exporter.export_summary(
            summary=stats['basic'],
            filename=f"{args.output.stem}_summary.csv"
        )

        logger.info(f"  ✓ Summary exported: {stats_file}")
        output_files.append(stats_file)

    return output_files


def main():
    """Main execution function."""
    args = parse_arguments()

    # Setup logging
    log_file = Path('./logs') / 'export.log'
    log_file.parent.mkdir(exist_ok=True)
    logger = setup_logging(
        log_file=log_file,
        level='DEBUG' if args.verbose else 'INFO'
    )

    logger.info("="*60)
    logger.info("EXPORT CLASSIFICATION RESULTS")
    logger.info("="*60)

    # Validate arguments
    validate_arguments(args, logger)

    # Load results
    results, metadata = load_results(args, logger)

    # Load thresholds if provided
    thresholds = load_thresholds(args, logger)

    # Apply threshold filtering
    results = apply_threshold_filtering(results, thresholds, args, logger)

    # Merge additional metadata
    results = merge_metadata(results, args, logger)

    # Sort and group
    results = sort_and_group_results(results, args, logger)

    # Calculate statistics
    stats = calculate_comprehensive_statistics(results, logger)

    # Export to files
    output_files = export_results(results, stats, metadata, args, logger)

    # Final summary
    logger.info("\n" + "="*60)
    logger.info("EXPORT COMPLETE")
    logger.info("="*60)
    logger.info(f"Results exported: {len(results):,}")
    logger.info(f"Output files:")
    for f in output_files:
        logger.info(f"  - {f}")
    logger.info("="*60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
