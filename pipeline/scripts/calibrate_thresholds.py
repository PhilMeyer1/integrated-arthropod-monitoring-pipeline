#!/usr/bin/env python3
"""
Calibrate Confidence Thresholds

This script calibrates confidence thresholds for each taxonomic class to achieve
≥95% accuracy, as described in the Methods section "Threshold Calculation".

The algorithm:
1. For each class, calculate baseline accuracy on test set
2. If baseline ≥95%, set threshold to 0.95
3. Otherwise, start at 0.95 and increment by 0.0001 until accuracy ≥95% or threshold = 1.0
4. Classes that cannot achieve 95% are excluded from downstream analysis

Usage:
    # Calibrate thresholds for model set 1
    python scripts/calibrate_thresholds.py --set-number 1

    # Use different target accuracy
    python scripts/calibrate_thresholds.py --set-number 1 --target-accuracy 97.0

    # Save results to custom path
    python scripts/calibrate_thresholds.py \\
        --set-number 1 \\
        --output ./results/thresholds_set1.json

Methods correspondence:
    - Threshold Calculation (Methods section)
    - Start threshold: 0.95
    - Increment: 0.0001
    - Target accuracy: 95%
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.utils import get_data_manager
from src.classification.thresholds import ThresholdCalibrator
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Calibrate confidence thresholds for hierarchical classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model selection
    parser.add_argument(
        '--set-number',
        type=int,
        required=True,
        help='Model set number to calibrate thresholds for'
    )

    # Calibration parameters (defaults from Methods)
    parser.add_argument(
        '--target-accuracy',
        type=float,
        default=95.0,
        help='Target accuracy percentage (Methods: 95.0)'
    )
    parser.add_argument(
        '--start-threshold',
        type=float,
        default=0.95,
        help='Starting confidence threshold (Methods: 0.95)'
    )
    parser.add_argument(
        '--increment',
        type=float,
        default=0.0001,
        help='Threshold increment step (Methods: 0.0001)'
    )

    # Grouping options
    parser.add_argument(
        '--group-by',
        type=str,
        choices=['predicted_taxon_id', 'model_taxon_id'],
        default='predicted_taxon_id',
        help='Column to group by for per-taxon calibration'
    )
    parser.add_argument(
        '--global-threshold',
        action='store_true',
        help='Calculate single global threshold instead of per-taxon'
    )

    # Database path
    parser.add_argument(
        '--db-path',
        type=Path,
        default=Path('data/arthropod_pipeline.db'),
        help='Path to SQLite database (default: data/arthropod_pipeline.db)'
    )

    # Output options
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output path for thresholds JSON (default: data/thresholds/thresholds_setN.json)'
    )
    parser.add_argument(
        '--summary',
        type=Path,
        default=None,
        help='Output path for summary table (default: data/thresholds/summary_setN.csv)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    return parser.parse_args()


def validate_arguments(args, data_manager):
    """Validate command line arguments."""
    # Check database exists
    if not args.db_path.exists():
        logger.error(f"Database not found: {args.db_path}")
        logger.error("Run scripts/01_setup_database.py first")
        sys.exit(1)

    # Check if models exist for this set
    models = data_manager.get_models_by_set(args.set_number)
    if not models:
        logger.error(f"No models found for set number {args.set_number}")
        sys.exit(1)

    # Check if inference results exist
    results = data_manager.get_inference_results(args.set_number)
    if not results:
        logger.error(f"No inference results found for set {args.set_number}")
        logger.error(f"Run classification first: python scripts/04_classify.py --set-number {args.set_number}")
        sys.exit(1)

    logger.info("Arguments validated:")
    logger.info(f"  Database: {args.db_path}")
    logger.info(f"  Model set: {args.set_number}")
    logger.info(f"  Models: {len(models)}")
    logger.info(f"  Inference results: {len(results)}")
    logger.info(f"  Target accuracy: {args.target_accuracy}%")
    logger.info("")


def setup_output_paths(args):
    """Setup output file paths."""
    if args.output is None:
        output_dir = Path('data/thresholds')
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = output_dir / f'thresholds_set{args.set_number}.json'

    if args.summary is None:
        summary_dir = Path('data/thresholds')
        summary_dir.mkdir(parents=True, exist_ok=True)
        args.summary = summary_dir / f'summary_set{args.set_number}.csv'

    logger.info(f"Output paths:")
    logger.info(f"  Thresholds: {args.output}")
    logger.info(f"  Summary: {args.summary}")
    logger.info("")


def run_calibration(args, data_manager):
    """
    Run threshold calibration.

    Args:
        args: Command line arguments
        data_manager: Database manager

    Returns:
        Dictionary with calibration results
    """
    logger.info("="*70)
    logger.info("THRESHOLD CALIBRATION")
    logger.info("="*70)
    logger.info("")

    # Initialize calibrator
    calibrator = ThresholdCalibrator(
        target_accuracy=args.target_accuracy,
        start_threshold=args.start_threshold,
        increment=args.increment,
        data_manager=data_manager
    )

    # Load inference results from database
    calibrator.load_results(
        set_number=args.set_number,
        only_test_set=True  # Only use test set for calibration
    )

    # Calibrate thresholds
    thresholds = calibrator.calibrate_thresholds(
        per_taxon=not args.global_threshold,
        group_by_column=args.group_by
    )

    return thresholds, calibrator


def save_results(thresholds, calibrator, args):
    """
    Save calibration results to files.

    Args:
        thresholds: Calibration results dictionary
        calibrator: ThresholdCalibrator instance
        args: Command line arguments
    """
    logger.info("Saving results...")

    # Save thresholds JSON
    calibrator.save_thresholds(thresholds, args.output)

    # Save summary table
    summary_df = calibrator.get_threshold_summary(thresholds)
    summary_df.to_csv(args.summary, index=False)
    logger.info(f"Saved summary table to {args.summary}")

    logger.info("")


def print_summary(thresholds, calibrator):
    """
    Print calibration summary.

    Args:
        thresholds: Calibration results
        calibrator: ThresholdCalibrator instance
    """
    valid_taxa = calibrator.get_valid_taxa(thresholds)
    invalid_taxa = calibrator.get_invalid_taxa(thresholds)

    logger.info("="*70)
    logger.info("CALIBRATION SUMMARY")
    logger.info("="*70)
    logger.info(f"Total taxa: {len(thresholds)}")
    logger.info(f"Valid taxa (≥95% accuracy): {len(valid_taxa)}")
    logger.info(f"Invalid taxa (<95% accuracy): {len(invalid_taxa)}")
    logger.info("")

    if invalid_taxa:
        logger.warning("The following taxa did not achieve 95% accuracy:")
        for taxon in invalid_taxa:
            result = thresholds[taxon]
            logger.warning(
                f"  - {taxon}: {result['accuracy_with_threshold']:.1f}% "
                f"(threshold={result['threshold']:.4f}, n={result['test_count']})"
            )
        logger.warning("")
        logger.warning("These taxa will be excluded from downstream analysis.")

    logger.info("="*70)
    logger.info("")


def main():
    """Main execution function."""
    args = parse_arguments()

    logger.info("="*70)
    logger.info("THRESHOLD CALIBRATION PIPELINE")
    logger.info("="*70)
    logger.info("")

    # Initialize database connection
    logger.info("Connecting to database...")
    db_url = f"sqlite:///{args.db_path}"
    data_manager = get_data_manager(use_database=True)
    data_manager.db_url = db_url
    data_manager.engine = None  # Force reconnect
    data_manager.__init__(use_database=True, database_url=db_url)
    logger.info(f"Connected to: {args.db_path}")
    logger.info("")

    # Validate arguments
    validate_arguments(args, data_manager)

    # Setup output paths
    setup_output_paths(args)

    # Run calibration
    thresholds, calibrator = run_calibration(args, data_manager)

    # Save results
    save_results(thresholds, calibrator, args)

    # Print summary
    print_summary(thresholds, calibrator)

    # Close database
    data_manager.close()

    logger.info("Threshold calibration completed successfully!")
    logger.info("")

    return 0


if __name__ == '__main__':
    sys.exit(main())
